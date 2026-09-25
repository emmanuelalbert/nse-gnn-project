"""
preprocessing/returns.py
------------------------
Computes logarithmic returns from raw adjusted closing prices.

Paper reference: Section III-A-2b (Eq. 1)
    log_return_t = log(P_t / P_{t-1})

Design decisions:
  - Logarithmic returns are used (not simple/arithmetic returns) because:
      1. Additive over time:  r(t1→t3) = r(t1→t2) + r(t2→t3)
      2. Symmetric:           gains and losses are treated symmetrically
      3. Approximate normality: better statistical properties for modelling
      4. Handle large moves:  less distortion than arithmetic returns
  - The Nifty 50 Index (^NSEI) is included as an additional entity, consistent
    with the paper's treatment of the S&P 500 Index as the 498th entity.
  - Returns are computed BEFORE KNN imputation (imputer.py) so that
    interpolated prices do not artificially create spurious return signals.
  - The first row of returns is always NaN (no P_{t-1} for day 0) and is
    dropped, so returns shape = (trading_days - 1, n_tickers).
  - Zero-price rows (data errors from yfinance) are replaced with NaN
    before the log is taken to avoid -inf values.

Usage:
    from preprocessing.returns import ReturnCalculator
    calc    = ReturnCalculator()
    returns = calc.compute(prices_df)          # raw log returns
    stats   = calc.long_term_stats(returns)    # mu and sigma per ticker
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as CFG

logger = logging.getLogger(__name__)


class ReturnCalculator:
    """
    Computes and validates daily logarithmic returns from closing prices.

    The class is stateless — all methods are pure transformations on
    the input DataFrame. No file I/O here; use data/storage.py for that.
    """

    # ─────────────────────────────────────────
    # CORE COMPUTATION  (paper Eq. 1)
    # ─────────────────────────────────────────

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute daily log returns for all tickers.

        Args:
            prices: DataFrame of shape (T, N) — adjusted closing prices,
                    DatetimeIndex, columns = ticker symbols.

        Returns:
            DataFrame of shape (T-1, N) — daily log returns.
            First row (NaN) is dropped. Zero prices become NaN before log.

        Raises:
            ValueError: if prices is empty or has fewer than 2 rows.
        """
        self._validate_prices(prices)

        # Replace zero prices with NaN — yfinance occasionally returns 0.0
        # for tickers with data issues. log(0) = -inf which corrupts downstream.
        n_zeros = (prices == 0).sum().sum()
        if n_zeros > 0:
            logger.warning(
                f"Replacing {n_zeros} zero-price values with NaN before log computation."
            )
            prices = prices.replace(0.0, np.nan)

        # Core computation: log(P_t / P_{t-1})  [paper Eq. 1]
        # pandas .shift(1) gives P_{t-1}; log of ratio = log(P_t) - log(P_{t-1})
        log_prices = np.log(prices)
        returns    = log_prices - log_prices.shift(1)

        # Drop the first row — it is always NaN because there is no P_{t-1}
        returns = returns.iloc[1:]

        # Sanity: log returns should be finite (or NaN, not ±inf)
        n_inf = np.isinf(returns.values).sum()
        if n_inf > 0:
            logger.warning(
                f"Found {n_inf} ±inf values in log returns — replacing with NaN. "
                "Check for price discontinuities in the raw data."
            )
            returns = returns.replace([np.inf, -np.inf], np.nan)

        logger.info(
            f"Log returns computed: {returns.shape[0]} days × {returns.shape[1]} tickers | "
            f"date range: {returns.index.min().date()} → {returns.index.max().date()}"
        )
        self._log_return_stats(returns)
        return returns

    # ─────────────────────────────────────────
    # STATISTICAL HELPERS
    # ─────────────────────────────────────────

    def long_term_stats(
        self, returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute long-term mean (μ) and standard deviation (σ) of log returns
        for each ticker. These are used by shock_detection/detector.py to
        apply the 2σ threshold (paper Eq. 2).

        Returns:
            DataFrame with columns ["mean", "std"] and ticker index.
        """
        stats = pd.DataFrame({
            "mean" : returns.mean(skipna=True),
            "std"  : returns.std(skipna=True, ddof=1),
        })
        logger.info(
            f"Long-term stats computed over {returns.shape[0]} trading days.\n"
            f"  Index mean: {stats.loc[CFG.SHOCK_REFERENCE_TICKER, 'mean']:.6f}  "
            f"std: {stats.loc[CFG.SHOCK_REFERENCE_TICKER, 'std']:.6f}"
            if CFG.SHOCK_REFERENCE_TICKER in stats.index else
            f"  (reference ticker {CFG.SHOCK_REFERENCE_TICKER} not in dataset)"
        )
        return stats

    def rolling_stats(
        self,
        returns: pd.DataFrame,
        window: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute rolling mean and rolling std of log returns.
        Used for visualisation and as alternative node features.

        Args:
            returns: Log returns DataFrame (T, N).
            window:  Rolling window in trading days.

        Returns:
            Tuple of (rolling_mean, rolling_std) DataFrames, both shape (T, N).
        """
        rolling_mean = returns.rolling(window=window, min_periods=1).mean()
        rolling_std  = returns.rolling(window=window, min_periods=1).std(ddof=1)
        return rolling_mean, rolling_std

    def period_mean_return(
        self,
        returns: pd.DataFrame,
        start: str,
        end: str,
    ) -> pd.Series:
        """
        Compute the mean log return for each ticker over a specific period.
        This is the NODE FEATURE used in the TGAT model (paper Section III-B):
        the mean of returns over the shock window is used as the scalar
        node feature fed into the graph encoder.

        Args:
            returns: Full log returns DataFrame.
            start:   Period start date (ISO string).
            end:     Period end date (ISO string).

        Returns:
            Series of shape (N,) — mean return per ticker for the period.
        """
        mask   = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
        period = returns.loc[mask]

        if period.empty:
            raise ValueError(f"No return data found between {start} and {end}.")

        mean_ret = period.mean(skipna=True)
        logger.debug(
            f"Period mean returns [{start} → {end}]: "
            f"{period.shape[0]} days, "
            f"index mean={mean_ret.get(CFG.SHOCK_REFERENCE_TICKER, float('nan')):.4f}"
        )
        return mean_ret

    def period_volatility(
        self,
        returns: pd.DataFrame,
        start: str,
        end: str,
        annualise: bool = False,
    ) -> pd.Series:
        """
        Compute realised volatility (std of log returns) for each ticker over
        a specific period. Optional annualisation (×√252 for NSE trading days).

        Used as an optional additional node feature (extend NODE_FEATURE_COLS
        in settings.py to include "volatility").
        """
        mask   = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
        period = returns.loc[mask]
        vol    = period.std(skipna=True, ddof=1)
        if annualise:
            vol = vol * np.sqrt(252)
        return vol

    # ─────────────────────────────────────────
    # AMPLITUDE  (paper Eq. 20)
    # ─────────────────────────────────────────

    def amplitude(
        self,
        returns: pd.DataFrame,
        start: str,
        end: str,
    ) -> pd.Series:
        """
        Compute amplitude of returns per ticker over a period.
        Paper Eq. 20:  Amplitude = MaxDailyChange - MinDailyChange

        Used in network_analysis/shock_metrics.py for the comparative
        analysis section (paper Section IV-C-d).

        Returns:
            Series of shape (N,) — amplitude per ticker.
        """
        mask   = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
        period = returns.loc[mask]
        return period.max(skipna=True) - period.min(skipna=True)

    # ─────────────────────────────────────────
    # AVERAGE CHANGE  (paper Eq. 19)
    # ─────────────────────────────────────────

    def average_change(
        self,
        prices: pd.DataFrame,
        start: str,
        end: str,
    ) -> pd.Series:
        """
        Compute average daily price change per ticker over a period.
        Paper Eq. 19:  AverageChange = Σ(P_current - P_previous) / N

        Note: computed on raw PRICES (not log returns) to match paper exactly.

        Returns:
            Series of shape (N,) — average daily price change per ticker.
        """
        mask   = (prices.index >= pd.Timestamp(start)) & (prices.index <= pd.Timestamp(end))
        period = prices.loc[mask]
        daily_change = period.diff().iloc[1:]     # P_t - P_{t-1}, drop first NaN row
        return daily_change.mean(skipna=True)

    # ─────────────────────────────────────────
    # VALIDATION & DIAGNOSTICS
    # ─────────────────────────────────────────

    def _validate_prices(self, prices: pd.DataFrame) -> None:
        if prices.empty:
            raise ValueError("prices DataFrame is empty.")
        if prices.shape[0] < 2:
            raise ValueError(
                f"prices has only {prices.shape[0]} row — need at least 2 to compute returns."
            )
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise TypeError("prices.index must be a DatetimeIndex.")

    def _log_return_stats(self, returns: pd.DataFrame) -> None:
        """Log a brief summary of return distribution to the logger."""
        flat  = returns.values.flatten()
        flat  = flat[~np.isnan(flat)]
        if len(flat) == 0:
            logger.warning("All return values are NaN — check the raw price data.")
            return
        logger.debug(
            f"Return distribution: "
            f"mean={flat.mean():.6f}, std={flat.std():.6f}, "
            f"min={flat.min():.4f}, max={flat.max():.4f}, "
            f"skew={float(pd.Series(flat).skew()):.3f}"
        )

    def distribution_report(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Per-ticker return distribution report.
        Returns DataFrame with columns: mean, std, skew, kurtosis, min, max, missing_pct.
        Useful for EDA notebook (notebooks/01_data_exploration.ipynb).
        """
        report = pd.DataFrame({
            "mean"       : returns.mean(),
            "std"        : returns.std(),
            "skew"       : returns.skew(),
            "kurtosis"   : returns.kurt(),
            "min"        : returns.min(),
            "max"        : returns.max(),
            "missing_pct": returns.isna().mean() * 100,
        }).round(6)
        return report


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level="DEBUG", format=CFG.LOG_FORMAT)

    # Synthetic test: 3 tickers, 10 days
    dates  = pd.date_range("2020-01-01", periods=10, freq="B")
    prices = pd.DataFrame(
        {
            "INFY.NS"  : [100, 102, 101, 105, 103, 108, 107, 110, 109, 112],
            "TCS.NS"   : [200, 198, 202, 205, 203, 207, 210, 208, 215, 213],
            "^NSEI"    : [12000, 12100, 12050, 12200, 12150, 12300, 12250, 12400, 12350, 12500],
        },
        index=dates,
    )
    prices.index.name = "Date"

    calc    = ReturnCalculator()
    returns = calc.compute(prices)
    stats   = calc.long_term_stats(returns)

    print(f"\nReturns shape: {returns.shape}")
    print(returns.round(6))
    print(f"\nLong-term stats:\n{stats}")
    print(f"\nDistribution report:\n{calc.distribution_report(returns)}")
    print(f"\nAmplitude (full period): {calc.amplitude(returns, '2020-01-01', '2020-01-15').round(6).to_dict()}")