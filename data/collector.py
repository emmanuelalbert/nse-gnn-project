"""
data/collector.py
-----------------
Downloads and caches historical daily closing prices for all 50 Nifty 50
entities (50 stocks + ^NSEI index) from Yahoo Finance.

Design decisions vs. the paper:
  - Paper pulled S&P 500 data from Yahoo Finance via API for 498 entities.
    We do the same for 50 NSE entities using yfinance.
  - NSE tickers require the ".NS" suffix (e.g. "INFY.NS"); the index is "^NSEI".
  - yfinance batch download is used for efficiency; individual fallback is used
    for any ticker that fails the batch (common with NSE tickers on some days).
  - auto_adjust=True: accounts for corporate actions (splits, dividends).
    This is critical for long history (2005–present) given frequent NSE splits.
  - Raw closing prices are stored as Parquet (columnar, fast I/O for 20yr data).

Usage:
    from data.collector import DataCollector
    collector = DataCollector()
    prices = collector.fetch_all()          # downloads or loads from cache
    collector.update()                      # incremental update from last date
"""

import logging
import random
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as CFG
from config.nifty50_universe import get_tickers, get_universe

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# SESSION NOTE
# ─────────────────────────────────────────────
# yfinance >=0.2.40 bundles curl_cffi impersonation internally and manages
# its own session object. Passing a raw curl_cffi.Session() into
# yf.download(session=...) breaks — yfinance expects attributes that only
# exist on its own internal session wrapper, not a bare curl_cffi session,
# which surfaces as `'str' object has no attribute 'name'`. So: don't pass
# a custom session at all. Just make sure yfinance itself is >=0.2.40 and
# it will use curl_cffi impersonation under the hood automatically.
_SESSION = None


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Sequential, single-ticker, jittered downloads. Batched multi-ticker
# yf.download() calls hit a more aggressively rate-limited Yahoo endpoint
# than single-ticker calls, so batching is disabled by default — one
# ticker at a time, with randomized spacing, is what actually survives.
BATCH_SIZE           = 1    # tickers per yfinance call (kept at 1 — see above)
BATCH_DELAY_MIN_SEC  = 8    # min jittered delay between tickers
BATCH_DELAY_MAX_SEC  = 15   # max jittered delay between tickers
RETRY_ATTEMPTS       = 3    # retries per failed ticker
RETRY_DELAY_SEC       = 10  # base seconds to wait between retries (multiplied by attempt)
MIN_ROWS_REQUIRED = 100     # ticker is considered failed if fewer rows returned


def _jittered_delay(min_sec: float = None, max_sec: float = None) -> None:
    """Sleep a random duration to avoid a detectable fixed-interval pattern."""
    min_sec = BATCH_DELAY_MIN_SEC if min_sec is None else min_sec
    max_sec = BATCH_DELAY_MAX_SEC if max_sec is None else max_sec
    delay = random.uniform(min_sec, max_sec)
    logger.debug(f"Sleeping {delay:.1f}s before next request.")
    time.sleep(delay)


# ─────────────────────────────────────────────
# COLLECTOR
# ─────────────────────────────────────────────

class DataCollector:
    """
    Fetches, validates, and caches daily closing prices for the full
    Nifty 50 universe from Yahoo Finance.

    All data is stored as Parquet at CFG.RAW_PRICES_FILE. On subsequent
    calls, the file is loaded from disk unless force_refresh=True.
    """

    def __init__(self, force_refresh: bool = False):
        self.force_refresh  = force_refresh
        self.tickers        = get_tickers()        # 50 symbols incl. ^NSEI
        self.universe       = get_universe()
        self.output_path    = CFG.RAW_PRICES_FILE

        CFG.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────

    def fetch_all(self) -> pd.DataFrame:
        """
        Main entry point. Returns a DataFrame of shape (trading_days, n_tickers)
        containing raw adjusted closing prices.

        Loads from disk if cache exists and force_refresh=False.
        """
        if self.output_path.exists() and not self.force_refresh:
            logger.info(f"Loading cached prices from {self.output_path}")
            return self._load()

        logger.info(
            f"Downloading closing prices for {len(self.tickers)} entities "
            f"[{CFG.START_DATE} → {CFG.END_DATE}]"
        )

        prices = self._download_all()
        prices = self._validate(prices)
        self._save(prices)

        logger.info(
            f"Saved {prices.shape[0]} trading days × {prices.shape[1]} tickers "
            f"to {self.output_path}"
        )
        return prices

    def update(self) -> pd.DataFrame:
        """
        Incremental update: appends new trading days since the last cached date.
        Falls back to full download if no cache exists.
        """
        if not self.output_path.exists():
            logger.warning("No cache found — running full download instead.")
            return self.fetch_all()

        existing = self._load()
        last_date = existing.index.max()
        today     = pd.Timestamp.today().normalize()

        if last_date >= today:
            logger.info("Cache is already up to date.")
            return existing

        update_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info(f"Incremental update from {update_start} → {CFG.END_DATE}")

        new_data = self._download_batch(
            tickers=self.tickers,
            start=update_start,
            end=CFG.END_DATE,
        )

        if new_data.empty:
            logger.info("No new data available.")
            return existing

        combined = pd.concat([existing, new_data], axis=0)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        combined = self._validate(combined)
        self._save(combined)

        logger.info(f"Updated cache: {combined.shape[0]} total trading days.")
        return combined

    # ─────────────────────────────────────────
    # DOWNLOAD INTERNALS
    # ─────────────────────────────────────────

    def _download_all(self) -> pd.DataFrame:
        """
        Downloads all tickers ONE AT A TIME, sequentially, with a randomized
        jittered delay between each request. This is deliberately not batched:
        multi-ticker yf.download() calls hit a more aggressively rate-limited
        Yahoo endpoint than single-ticker calls, and the fixed-interval batch
        pattern is itself a detectable fingerprint. Tickers that fail outright
        fall into a second, more patient retry pass at the end.
        """
        all_frames = []
        failed     = []

        for idx, ticker in enumerate(self.tickers, start=1):
            logger.info(f"[{idx}/{len(self.tickers)}] Fetching {ticker} ...")
            try:
                frame = self._download_batch([ticker], CFG.START_DATE, CFG.END_DATE)
                if frame.empty or len(frame) < MIN_ROWS_REQUIRED:
                    logger.warning(f"{ticker}: only {len(frame)} rows on first pass — queued for retry.")
                    failed.append(ticker)
                else:
                    all_frames.append(frame)
            except Exception as exc:
                logger.warning(f"{ticker}: fetch failed — {exc} — queued for retry.")
                failed.append(ticker)

            # Jittered pacing between every single request, not just batches.
            if idx < len(self.tickers):
                _jittered_delay()

        # Second, more patient retry pass for anything that failed above.
        if failed:
            logger.info(f"Retrying {len(failed)} failed tickers individually (patient pass).")
            for ticker in failed:
                frame = self._download_single_with_retry(ticker)
                if frame is not None:
                    all_frames.append(frame)

        if not all_frames:
            raise RuntimeError("All downloads failed. Check internet connection or ticker symbols.")

        combined = pd.concat(all_frames, axis=1)
        combined.sort_index(inplace=True)
        return combined

    def _download_batch(
        self,
        tickers: list,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Download a batch of tickers using yfinance's multi-ticker interface.
        Returns a DataFrame with tickers as columns.
        """
        raw = yf.download(
            tickers        = tickers,
            start          = start,
            end            = end,
            interval       = CFG.YFINANCE_INTERVAL,
            auto_adjust    = CFG.YFINANCE_AUTO_ADJUST,
            progress       = CFG.YFINANCE_PROGRESS,
            group_by       = "ticker",
            threads        = False,   # sequential — no concurrent burst to Yahoo
        )

        # yfinance returns MultiIndex columns when multiple tickers are fetched.
        # Extract only the "Close" price for each ticker.
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw.xs("Close", level=1, axis=1)
        else:
            # Single ticker returns flat columns
            close = raw[["Close"]].rename(columns={"Close": tickers[0]})

        close.index = pd.to_datetime(close.index)
        close.index.name = "Date"
        return close

    def _download_single_with_retry(
        self,
        ticker: str,
    ) -> Optional[pd.DataFrame]:
        """
        Download a single ticker with retry logic for transient failures.
        Returns None if all retries are exhausted.
        """
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                # Growing, jittered back-off — longer and less predictable
                # with each attempt.
                base = RETRY_DELAY_SEC * attempt
                time.sleep(base + random.uniform(0, base))
                raw = yf.download(
                    tickers     = ticker,
                    start       = CFG.START_DATE,
                    end         = CFG.END_DATE,
                    interval    = CFG.YFINANCE_INTERVAL,
                    auto_adjust = CFG.YFINANCE_AUTO_ADJUST,
                    progress    = False,
                    threads     = False,
                )
                if raw.empty or len(raw) < MIN_ROWS_REQUIRED:
                    logger.warning(
                        f"{ticker}: attempt {attempt} returned {len(raw)} rows (min {MIN_ROWS_REQUIRED})."
                    )
                    continue

                close = raw[["Close"]].rename(columns={"Close": ticker})
                close.index = pd.to_datetime(close.index)
                close.index.name = "Date"
                logger.info(f"{ticker}: downloaded {len(close)} rows.")
                return close

            except Exception as exc:
                logger.warning(f"{ticker}: attempt {attempt} failed — {exc}")

        logger.error(f"{ticker}: all {RETRY_ATTEMPTS} attempts failed. Skipping.")
        return None

    # ─────────────────────────────────────────
    # VALIDATION
    # ─────────────────────────────────────────

    def _validate(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Post-download validation:
          1. Drop tickers with missing data below the threshold.
          2. Drop rows where ALL tickers are NaN (non-trading days).
          3. Ensure chronological order.
          4. Log a data quality report.
        """
        n_tickers_before = prices.shape[1]
        n_days           = prices.shape[0]

        # 1. Drop tickers with too many missing values
        min_valid = int(CFG.MIN_VALID_DAYS_FRACTION * n_days)
        valid_counts = prices.notna().sum()
        dropped = valid_counts[valid_counts < min_valid].index.tolist()

        if dropped:
            logger.warning(
                f"Dropping {len(dropped)} tickers with < {CFG.MIN_VALID_DAYS_FRACTION:.0%} "
                f"valid days: {dropped}"
            )
            prices = prices.drop(columns=dropped)

        # 2. Drop rows where all tickers are NaN (weekends, holidays already
        #    excluded by yfinance, but belt-and-suspenders check)
        all_nan_rows = prices.index[prices.isna().all(axis=1)]
        if len(all_nan_rows) > 0:
            logger.debug(f"Dropping {len(all_nan_rows)} all-NaN rows.")
            prices = prices.drop(index=all_nan_rows)

        # 3. Sort chronologically
        prices.sort_index(inplace=True)

        # 4. Data quality report
        missing_pct = prices.isna().mean() * 100
        logger.info(
            f"Data quality: {prices.shape[1]}/{n_tickers_before} tickers kept, "
            f"{prices.shape[0]} trading days, "
            f"avg missing: {missing_pct.mean():.2f}%"
        )
        if missing_pct.max() > 5.0:
            high_missing = missing_pct[missing_pct > 5.0]
            logger.warning(
                f"High missing data (>5%) in: "
                f"{high_missing.sort_values(ascending=False).head(10).to_dict()}"
            )

        return prices

    # ─────────────────────────────────────────
    # STORAGE
    # ─────────────────────────────────────────

    def _save(self, prices: pd.DataFrame) -> None:
        """Persist prices DataFrame to Parquet."""
        prices.to_parquet(self.output_path, engine="pyarrow", compression="snappy")
        logger.debug(f"Saved to {self.output_path}")

    def _load(self) -> pd.DataFrame:
        """Load prices DataFrame from Parquet cache."""
        prices = pd.read_parquet(self.output_path, engine="pyarrow")
        prices.index = pd.to_datetime(prices.index)
        prices.index.name = "Date"
        logger.debug(
            f"Loaded {prices.shape[0]} rows × {prices.shape[1]} cols "
            f"from {self.output_path}"
        )
        return prices

    # ─────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────

    def get_summary(self) -> pd.DataFrame:
        """
        Returns a summary DataFrame with per-ticker statistics:
        start date, end date, valid days, missing %, last close price.
        """
        prices = self._load()
        summary_rows = []
        for ticker in prices.columns:
            col = prices[ticker].dropna()
            summary_rows.append({
                "ticker"       : ticker,
                "first_date"   : col.index.min().date() if not col.empty else None,
                "last_date"    : col.index.max().date() if not col.empty else None,
                "valid_days"   : col.shape[0],
                "missing_pct"  : f"{prices[ticker].isna().mean() * 100:.2f}%",
                "last_close"   : round(col.iloc[-1], 2) if not col.empty else None,
            })
        return pd.DataFrame(summary_rows).set_index("ticker")


# ─────────────────────────────────────────────
# CLI  —  python -m data.collector
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level   = CFG.LOG_LEVEL,
        format  = CFG.LOG_FORMAT,
    )
    collector = DataCollector(force_refresh=False)
    prices    = collector.fetch_all()

    print(f"\nDataset shape : {prices.shape}")
    print(f"Date range    : {prices.index.min().date()} → {prices.index.max().date()}")
    print(f"\nFirst 3 rows:\n{prices.head(3)}")
    print(f"\nData summary:\n{collector.get_summary()}")