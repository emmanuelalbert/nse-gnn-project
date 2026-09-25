"""
preprocessing/shock_detection/detector.py
-----------------------------------------
Identifies shock periods in the Nifty 50 log returns using the 2σ rule
from the reference paper.

Paper reference: Section III-A-3 (Eq. 2)

    A shock period is identified if the absolute value of the average
    daily logarithmic return during a potential shock window (e.g. 5
    consecutive trading days) exceeds 2 standard deviations from the
    long-term mean:

        | (1/N) * Σ log(P_t / P_{t-1}) - μ | > 2σ

    where:
        N  = number of consecutive trading days in the shock window
        μ  = long-term mean of daily log returns (^NSEI index)
        σ  = long-term std of daily log returns  (^NSEI index)

Design decisions vs. the paper:
  - Reference series: ^NSEI index (not the cross-sectional average of stocks).
    The index provides a single, unambiguous signal of market-wide distress.
  - Rolling window: we scan every consecutive 5-trading-day window across
    the full history (2005–present) and flag windows that breach the threshold.
  - Consecutive-day merging: adjacent flagged windows are merged into a single
    shock period to avoid counting one crisis event as multiple shocks.
    The SHOCK_MERGE_GAP_DAYS setting (default 3) controls this.
  - Minimum duration: the paper requires at least 5 consecutive trading days.
    We enforce this as a post-merge filter.
  - The paper found 53 distinct shock periods in the S&P 500 (2000–2023).
    For Nifty 50 (2005–present) we expect a broadly similar count (~40–60)
    given the Indian market's exposure to both domestic and global shocks.

Known India-specific shocks this detector should surface:
    2008-10  Global Financial Crisis         (Lehman, global contagion)
    2009-01  Satyam accounting fraud
    2011-08  US debt downgrade / Euro crisis
    2013-06  Taper Tantrum (INR crash)
    2015-08  China-led global selloff
    2016-11  Demonetization shock
    2018-09  IL&FS / NBFC liquidity crisis
    2020-03  COVID-19 pandemic onset
    2022-02  Russia-Ukraine / FII selloff
    2023-01  Adani Group short-seller report

Usage:
    from shock_detection.detector import ShockDetector
    detector = ShockDetector()
    shocks   = detector.detect(returns_df)   # returns list of ShockPeriod dicts
    report   = detector.summary_report(shocks)
"""

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as CFG

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ShockPeriod:
    """
    Represents a single detected shock period.

    Attributes:
        name         : Human-readable label (auto-generated or user-supplied).
        start        : First trading day of the shock window (ISO string).
        end          : Last trading day of the shock window  (ISO string).
        duration     : Number of trading days in the window.
        avg_return   : Average daily log return of ^NSEI over the window.
        max_drawdown : Maximum single-day log return (most negative value).
        sigma_breach : How many σ the avg_return deviates from the long-term μ.
        type         : Optional label ("global_financial", "domestic", "policy", etc.)
    """
    name          : str
    start         : str
    end           : str
    duration      : int
    avg_return    : float
    max_drawdown  : float
    sigma_breach  : float
    type          : str = "auto_detected"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_settings_dict(self) -> dict:
        """Returns a dict in the same format as settings.TARGET_SHOCK_PERIODS."""
        return {
            "name"  : self.name,
            "start" : self.start,
            "end"   : self.end,
            "type"  : self.type,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class ShockDetector:
    """
    Scans log returns of the Nifty 50 Index to detect shock periods
    using the 2σ rolling-window rule (paper Eq. 2).

    The detector operates in four stages:
        1. Compute long-term μ and σ from the full history of the reference series.
        2. Roll a SHOCK_WINDOW_DAYS window and flag windows that breach |avg| > 2σ.
        3. Merge flagged windows separated by fewer than SHOCK_MERGE_GAP_DAYS.
        4. Filter out merged periods shorter than SHOCK_WINDOW_DAYS.
    """

    def __init__(
        self,
        reference_ticker : str   = CFG.SHOCK_REFERENCE_TICKER,
        window_days      : int   = CFG.SHOCK_WINDOW_DAYS,
        sigma_threshold  : float = CFG.SHOCK_SIGMA_THRESHOLD,
        merge_gap_days   : int   = CFG.SHOCK_MERGE_GAP_DAYS,
        min_duration     : int   = CFG.SHOCK_WINDOW_DAYS,
        sigma_basis      : str   = getattr(CFG, "SHOCK_SIGMA_BASIS", "rolling"),
    ):
        self.reference_ticker = reference_ticker
        self.window_days      = window_days
        self.sigma_threshold  = sigma_threshold
        self.merge_gap_days   = merge_gap_days
        self.min_duration     = min_duration
        if sigma_basis not in ("rolling", "daily"):
            raise ValueError(f"sigma_basis must be 'rolling' or 'daily', got '{sigma_basis}'")
        self.sigma_basis      = sigma_basis

        # Populated during detect()
        self._mu    : Optional[float] = None
        self._sigma : Optional[float] = None

    # ─────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────

    def detect(self, returns: pd.DataFrame) -> List[ShockPeriod]:
        """
        Run the full shock detection pipeline on the returns DataFrame.

        Args:
            returns: Log returns DataFrame (T, N) with DatetimeIndex.
                     Must contain self.reference_ticker as a column.

        Returns:
            List of ShockPeriod dataclasses, sorted chronologically.
        """
        self._validate(returns)
        ref = self._get_reference_series(returns)

        # Stage 1: Long-term statistics (paper: mu and sigma of full history)
        self._mu = float(ref.mean(skipna=True))

        if self.sigma_basis == "daily":
            self._sigma = float(ref.std(skipna=True, ddof=1))
        else:  # "rolling" — std of the rolling window-average series itself
            rolling_avg = ref.rolling(
                window=self.window_days, min_periods=self.window_days
            ).mean()
            self._sigma = float(rolling_avg.std(skipna=True, ddof=1))

        logger.info(
            f"Reference series: {self.reference_ticker} | "
            f"sigma_basis={self.sigma_basis} | "
            f"mu={self._mu:.6f}  sigma={self._sigma:.6f}  "
            f"threshold=|x - mu| > {self.sigma_threshold}sigma = "
            f"{abs(self._mu) + self.sigma_threshold * self._sigma:.6f}"
        )

        # Stage 2: Rolling window flagging  (paper Eq. 2)
        flagged_dates = self._rolling_flag(ref)
        logger.info(
            f"Rolling window ({self.window_days} days) flagged "
            f"{len(flagged_dates)} individual trading days."
        )

        if not flagged_dates:
            logger.warning("No shock windows detected. Check σ threshold or data range.")
            return []

        # Stage 3: Merge adjacent flagged windows
        raw_windows = self._dates_to_windows(ref, flagged_dates)
        merged      = self._merge_windows(raw_windows, ref.index)
        logger.info(
            f"Merged into {len(merged)} shock windows "
            f"(gap threshold: {self.merge_gap_days} days)."
        )

        # Stage 4: Filter by minimum duration
        shocks = self._build_shock_periods(merged, ref)
        shocks = [s for s in shocks if s.duration >= self.min_duration]

        logger.info(
            f"Final shock count: {len(shocks)} periods "
            f"(min duration: {self.min_duration} trading days)."
        )
        self._log_shock_summary(shocks)
        return shocks

    def detect_with_stats(
        self, returns: pd.DataFrame
    ) -> Tuple[List[ShockPeriod], pd.DataFrame]:
        """
        Run detection and also return the full rolling-window statistics
        DataFrame for diagnostic use (e.g. plotting in visualize_shocks.py).

        Returns:
            (shocks, stats_df) where stats_df has columns:
                rolling_avg, threshold_upper, threshold_lower, is_shocked
        """
        self._validate(returns)
        ref   = self._get_reference_series(returns)
        shocks = self.detect(returns)
        stats  = self._build_stats_df(ref)
        return shocks, stats

    # ─────────────────────────────────────────
    # STAGE 2: ROLLING FLAG  (paper Eq. 2)
    # ─────────────────────────────────────────

    def _rolling_flag(self, ref: pd.Series) -> List[pd.Timestamp]:
        """
        Slide a window of SHOCK_WINDOW_DAYS over the reference series.
        Flag the START date of any window whose |avg_return - μ| > σ_threshold * σ.

        Returns list of flagged start dates.
        """
        flagged = []
        dates   = ref.index
        vals    = ref.values

        for i in range(len(dates) - self.window_days + 1):
            window_vals = vals[i : i + self.window_days]
            # Skip windows with too many NaN values
            valid = window_vals[~np.isnan(window_vals)]
            if len(valid) < max(1, self.window_days // 2):
                continue

            window_avg = float(np.mean(valid))

            # Paper Eq. 2: |avg - μ| > σ_threshold * σ
            if abs(window_avg - self._mu) > self.sigma_threshold * self._sigma:
                flagged.append(dates[i])

        return flagged

    # ─────────────────────────────────────────
    # STAGE 3: MERGE WINDOWS
    # ─────────────────────────────────────────

    def _dates_to_windows(
        self,
        ref: pd.Series,
        flagged_starts: List[pd.Timestamp],
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Convert flagged start dates to (start, end) tuples.
        Each start date produces a window of length SHOCK_WINDOW_DAYS.
        """
        dates   = ref.index
        windows = []
        for start in flagged_starts:
            idx   = dates.get_loc(start)
            end   = dates[min(idx + self.window_days - 1, len(dates) - 1)]
            windows.append((start, end))
        return windows

    def _merge_windows(
        self,
        windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
        all_dates: pd.DatetimeIndex,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Merge overlapping or near-adjacent shock windows.
        Two windows are merged if the gap between them is ≤ SHOCK_MERGE_GAP_DAYS
        trading days (not calendar days).
        """
        if not windows:
            return []

        # Sort by start date
        windows = sorted(windows, key=lambda w: w[0])
        merged  = [windows[0]]

        for cur_start, cur_end in windows[1:]:
            prev_start, prev_end = merged[-1]

            # Count trading days in the gap
            gap_mask = (all_dates > prev_end) & (all_dates < cur_start)
            gap_days = int(gap_mask.sum())

            if gap_days <= self.merge_gap_days:
                # Extend the current merged window
                merged[-1] = (prev_start, max(prev_end, cur_end))
            else:
                merged.append((cur_start, cur_end))

        return merged

    # ─────────────────────────────────────────
    # STAGE 4: BUILD ShockPeriod OBJECTS
    # ─────────────────────────────────────────

    def _build_shock_periods(
        self,
        windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
        ref: pd.Series,
    ) -> List[ShockPeriod]:
        """
        Convert (start, end) tuples into ShockPeriod dataclasses,
        computing per-period statistics.
        """
        shocks = []
        for i, (start, end) in enumerate(windows, start=1):
            mask   = (ref.index >= start) & (ref.index <= end)
            period = ref.loc[mask].dropna()

            if period.empty:
                continue

            duration     = int(mask.sum())
            avg_return   = float(period.mean())
            max_drawdown = float(period.min())      # most negative single day
            sigma_breach = float(abs(avg_return - self._mu) / self._sigma)

            # Auto-generate name: "Shock_001 (YYYY-MM-DD)"
            name = f"Shock_{i:03d} ({start.strftime('%Y-%m-%d')})"

            shocks.append(ShockPeriod(
                name         = name,
                start        = start.strftime("%Y-%m-%d"),
                end          = end.strftime("%Y-%m-%d"),
                duration     = duration,
                avg_return   = round(avg_return, 6),
                max_drawdown = round(max_drawdown, 6),
                sigma_breach = round(sigma_breach, 2),
                type         = "auto_detected",
            ))
        return shocks

    # ─────────────────────────────────────────
    # STATS DF (for visualisation)
    # ─────────────────────────────────────────

    def _build_stats_df(self, ref: pd.Series) -> pd.DataFrame:
        """
        Build a DataFrame of rolling window statistics alongside the
        shock thresholds. Used by visualize_shocks.py for the timeline plot.

        Columns:
            daily_return      : Raw daily log return of reference ticker
            rolling_avg       : Rolling mean over SHOCK_WINDOW_DAYS
            threshold_upper   : μ + σ_threshold × σ
            threshold_lower   : μ - σ_threshold × σ
            is_shocked        : bool — rolling_avg outside threshold band
        """
        rolling_avg = ref.rolling(
            window=self.window_days, min_periods=self.window_days
        ).mean()

        threshold_upper = self._mu + self.sigma_threshold * self._sigma
        threshold_lower = self._mu - self.sigma_threshold * self._sigma

        is_shocked = (rolling_avg > threshold_upper) | (rolling_avg < threshold_lower)

        stats = pd.DataFrame({
            "daily_return"    : ref,
            "rolling_avg"     : rolling_avg,
            "threshold_upper" : threshold_upper,
            "threshold_lower" : threshold_lower,
            "is_shocked"      : is_shocked,
        })
        return stats

    # ─────────────────────────────────────────
    # MATCH TARGET PERIODS
    # ─────────────────────────────────────────

    def match_target_periods(
        self,
        shocks: List[ShockPeriod],
        target_periods: Optional[list] = None,
        overlap_days: int = 2,
    ) -> List[ShockPeriod]:
        """
        Tag auto-detected shocks that overlap with the manually defined
        TARGET_SHOCK_PERIODS in settings.py.

        This validates that the detector correctly surfaces the four key
        Indian market events we want to compare (GFC, IL&FS, COVID, Demo).

        Args:
            shocks         : List of auto-detected ShockPeriod objects.
            target_periods : List of target dicts (default: settings.TARGET_SHOCK_PERIODS).
            overlap_days   : Minimum overlapping trading days to count as a match.

        Returns:
            The same shock list with .name and .type updated for matching shocks.
        """
        target_periods = target_periods or CFG.TARGET_SHOCK_PERIODS

        for shock in shocks:
            s_start = pd.Timestamp(shock.start)
            s_end   = pd.Timestamp(shock.end)

            for target in target_periods:
                t_start = pd.Timestamp(target["start"])
                t_end   = pd.Timestamp(target["end"])

                # Compute overlap
                overlap_start = max(s_start, t_start)
                overlap_end   = min(s_end,   t_end)

                if overlap_start <= overlap_end:
                    # Approximate trading days in overlap (÷ 1.4 for weekends)
                    cal_days = (overlap_end - overlap_start).days
                    trading_approx = cal_days * 5 / 7

                    if trading_approx >= overlap_days:
                        shock.name = target["name"]
                        shock.type = target["type"]
                        logger.info(
                            f"Matched: '{shock.name}' "
                            f"[{shock.start} → {shock.end}] "
                            f"↔ target [{target['start']} → {target['end']}]"
                        )
                        break

        return shocks

    # ─────────────────────────────────────────
    # DIAGNOSTICS
    # ─────────────────────────────────────────

    def summary_report(self, shocks: List[ShockPeriod]) -> pd.DataFrame:
        """
        Build a summary DataFrame of all detected shock periods.
        Useful for notebook display and saving to results/metrics/.

        Columns: name, start, end, duration, avg_return, max_drawdown,
                 sigma_breach, type
        """
        if not shocks:
            return pd.DataFrame()

        rows = [s.to_dict() for s in shocks]
        df   = pd.DataFrame(rows)
        df   = df.sort_values("start").reset_index(drop=True)
        df.index += 1   # 1-based index for display
        return df

    def stats_for_period(
        self,
        returns: pd.DataFrame,
        shock: ShockPeriod,
    ) -> pd.DataFrame:
        """
        Per-ticker statistics for a single shock period.
        Returns mean return, std, max drawdown for every ticker in the dataset.
        Used by network_analysis/shock_metrics.py.
        """
        mask   = (returns.index >= pd.Timestamp(shock.start)) & \
                 (returns.index <= pd.Timestamp(shock.end))
        period = returns.loc[mask]

        return pd.DataFrame({
            "mean_return"  : period.mean(skipna=True),
            "volatility"   : period.std(skipna=True, ddof=1),
            "max_drawdown" : period.min(skipna=True),
            "amplitude"    : period.max(skipna=True) - period.min(skipna=True),
        }).round(6)

    # ─────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────

    def _get_reference_series(self, returns: pd.DataFrame) -> pd.Series:
        if self.reference_ticker not in returns.columns:
            raise KeyError(
                f"Reference ticker '{self.reference_ticker}' not found in returns. "
                f"Available: {list(returns.columns[:5])} ..."
            )
        return returns[self.reference_ticker].copy()

    def _validate(self, returns: pd.DataFrame) -> None:
        if returns.empty:
            raise ValueError("returns DataFrame is empty.")
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise TypeError("returns.index must be a DatetimeIndex.")
        if returns.shape[0] < self.window_days * 2:
            raise ValueError(
                f"returns has only {returns.shape[0]} rows — need at least "
                f"{self.window_days * 2} for meaningful shock detection."
            )

    def _log_shock_summary(self, shocks: List[ShockPeriod]) -> None:
        if not shocks:
            return
        durations     = [s.duration for s in shocks]
        breaches      = [s.sigma_breach for s in shocks]
        avg_returns   = [s.avg_return for s in shocks]
        logger.info(
            f"Shock statistics across {len(shocks)} periods:\n"
            f"  Duration  : mean={np.mean(durations):.1f}d, "
            f"min={min(durations)}d, max={max(durations)}d\n"
            f"  σ breach  : mean={np.mean(breaches):.2f}σ, "
            f"max={max(breaches):.2f}σ\n"
            f"  Avg return: mean={np.mean(avg_returns):.4f}, "
            f"worst={min(avg_returns):.4f}"
        )

    # ─────────────────────────────────────────
    # PROPERTIES
    # ─────────────────────────────────────────

    @property
    def long_term_mu(self) -> Optional[float]:
        """Long-term mean of the reference series (set after detect() is called)."""
        return self._mu

    @property
    def long_term_sigma(self) -> Optional[float]:
        """Long-term std of the reference series (set after detect() is called)."""
        return self._sigma

    @property
    def threshold_band(self) -> Optional[Tuple[float, float]]:
        """(lower, upper) return thresholds. Set after detect()."""
        if self._mu is None:
            return None
        delta = self.sigma_threshold * self._sigma
        return (self._mu - delta, self._mu + delta)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level="DEBUG", format=CFG.LOG_FORMAT)
    np.random.seed(42)

    # 3000 trading days (~12 years): normal regime + 5 injected shock windows
    n = 3000
    dates   = pd.date_range("2005-01-03", periods=n, freq="B")
    normal  = np.random.normal(0.0003, 0.010, size=n)
    shock_signal = np.zeros(n)

    # Inject 5 known shock windows
    shock_ranges = [(200, 225), (600, 625), (1200, 1250), (1800, 1815), (2500, 2530)]
    for s, e in shock_ranges:
        shock_signal[s:e] = np.random.normal(-0.025, 0.015, size=e - s)

    index_returns  = normal + shock_signal
    stock_returns  = np.random.normal(0.0003, 0.012, size=(n, 5))

    data = np.column_stack([stock_returns, index_returns])
    cols = ["INFY.NS", "TCS.NS", "WIPRO.NS", "HDFCBANK.NS", "RELIANCE.NS", "^NSEI"]

    returns = pd.DataFrame(data, index=dates, columns=cols)
    returns.index.name = "Date"

    detector = ShockDetector()
    shocks, stats_df = detector.detect_with_stats(returns)

    print(f"\nDetected {len(shocks)} shock periods")
    print(f"Long-term μ={detector.long_term_mu:.6f}, σ={detector.long_term_sigma:.6f}")
    print(f"Threshold band: {detector.threshold_band}")
    print()

    report = detector.summary_report(shocks)
    print(report[["name", "start", "end", "duration", "avg_return", "sigma_breach"]].to_string())

    print(f"\nInjected shock ranges (expected to be detected):")
    for s, e in shock_ranges:
        print(f"  trading days [{s}:{e}] → {dates[s].date()} → {dates[e-1].date()}")