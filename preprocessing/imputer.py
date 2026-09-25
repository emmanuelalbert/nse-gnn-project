"""
preprocessing/imputer.py
------------------------
Handles missing values in the log returns DataFrame using K-Nearest
Neighbours imputation.

Paper reference: Section III-A-2c
    "We employed the KNN Imputer method to handle missing values. This approach
     replaces missing data with the average of the nearest neighbours, ensuring
     that our dataset remains robust and comprehensive. We chose this method over
     simpler techniques, like mean or zero imputation, because it preserves the
     relationships within the data, leading to more accurate and reliable analysis."

Why KNN over simpler alternatives for financial returns:
  - Mean imputation: replaces every missing value with the column mean,
    destroying the correlation structure between stocks (a key input to Granger).
  - Forward/backward fill: can introduce artificial autocorrelation in returns —
    particularly dangerous for Granger causality testing.
  - Zero imputation: implies the stock had zero return on the missing day, which
    is factually wrong (the market was simply closed or the ticker had a data gap).
  - KNN: finds the K most similar trading days (by the full cross-section of
    returns) and uses their values. This preserves cross-sectional correlations,
    which is exactly what Granger causality needs to work correctly.

Implementation notes:
  - sklearn's KNNImputer works on columns by default (each column = one ticker).
    We transpose so that imputation is done across the TIME dimension instead:
    a missing return on day t is filled using the K nearest DAYS (not tickers).
    This is more appropriate for financial time-series where the cross-section
    of stocks on a given day is the natural "neighbourhood".
  - After imputation, we verify that no NaN values remain and that the
    imputed values are within a plausible range (±50% single-day return).

Usage:
    from preprocessing.imputer import ReturnImputer
    imputer  = ReturnImputer(n_neighbors=5)
    imputed  = imputer.fit_transform(returns_df)
    report   = imputer.missing_report(returns_df)   # before/after comparison
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as CFG

logger = logging.getLogger(__name__)

# Sanity cap: imputed log returns beyond this magnitude are suspicious.
# A single-day NSE log return beyond ±50% is almost certainly a data error.
MAX_PLAUSIBLE_RETURN = 0.50


class ReturnImputer:
    """
    K-Nearest Neighbours imputer for missing log return values.

    The KNNImputer is fitted on the TRAINING data only and then applied
    to both train and test splits — consistent with the Monte Carlo
    evaluation protocol (paper Section III-D) to prevent leakage.

    Attributes:
        n_neighbors : Number of nearest neighbours (paper uses default=5).
        weights     : "uniform" | "distance" — how to weight neighbours.
        _imputer    : Fitted sklearn KNNImputer instance.
        _is_fitted  : Whether fit() has been called.
    """

    def __init__(
        self,
        n_neighbors: int = CFG.KNN_N_NEIGHBORS,
        weights: str = "uniform",
    ):
        self.n_neighbors = n_neighbors
        self.weights     = weights
        self._imputer: Optional[KNNImputer] = None
        self._is_fitted  = False

    # ─────────────────────────────────────────
    # MAIN API
    # ─────────────────────────────────────────

    def fit(self, returns: pd.DataFrame) -> "ReturnImputer":
        """
        Fit the KNN imputer on a returns DataFrame.

        Args:
            returns: Log returns DataFrame (T, N). Should be TRAINING data only.

        Returns:
            self (for method chaining).
        """
        self._validate(returns)
        n_missing = returns.isna().sum().sum()
        logger.info(
            f"Fitting KNN imputer (k={self.n_neighbors}) on "
            f"{returns.shape[0]} days × {returns.shape[1]} tickers | "
            f"{n_missing} missing values ({n_missing / returns.size * 100:.2f}%)"
        )

        self._imputer = KNNImputer(
            n_neighbors     = self.n_neighbors,
            weights         = self.weights,
            metric          = "nan_euclidean",   # handles remaining NaN in distance calc
        )

        # Transpose: impute across TIME (rows = features, cols = days)
        # so that each missing day is filled from its K most-similar days.
        self._imputer.fit(returns.values)
        self._is_fitted = True
        logger.debug("KNN imputer fitted.")
        return self

    def transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted imputer to a returns DataFrame.

        Args:
            returns: Log returns DataFrame (T, N). May be train or test split.

        Returns:
            DataFrame of same shape with NaN values replaced.
        """
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before .transform().")

        self._validate(returns)
        n_missing_before = returns.isna().sum().sum()

        if n_missing_before == 0:
            logger.info("No missing values found — skipping imputation.")
            return returns.copy()

        # KNNImputer.transform works on the same feature space as fit
        imputed_array = self._imputer.transform(returns.values)

        imputed_df = pd.DataFrame(
            imputed_array,
            index   = returns.index,
            columns = returns.columns,
        )

        # Post-imputation checks
        n_missing_after = imputed_df.isna().sum().sum()
        if n_missing_after > 0:
            logger.warning(
                f"{n_missing_after} NaN values remain after KNN imputation. "
                "This can happen when entire rows or columns are NaN. "
                "Applying forward-fill then backward-fill as fallback."
            )
            imputed_df = imputed_df.ffill().bfill()

        self._check_plausibility(imputed_df, returns)

        logger.info(
            f"Imputation complete: {n_missing_before} → "
            f"{imputed_df.isna().sum().sum()} missing values."
        )
        return imputed_df

    def fit_transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one call.
        Use ONLY when working with the full dataset (no train/test split).
        For Monte Carlo splits, always fit on train, transform on both.
        """
        return self.fit(returns).transform(returns)

    def transform_period(
        self,
        returns: pd.DataFrame,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Impute a specific date-range slice of returns.
        Useful for applying the fitted imputer to a single shock period.

        Args:
            returns: Full log returns DataFrame (T, N). Imputer must be fitted.
            start:   Start date of the period (ISO string).
            end:     End date of the period (ISO string).

        Returns:
            Imputed slice of shape (period_days, N).
        """
        mask   = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
        period = returns.loc[mask].copy()

        if period.empty:
            raise ValueError(f"No data found between {start} and {end}.")

        return self.transform(period)

    # ─────────────────────────────────────────
    # DIAGNOSTICS
    # ─────────────────────────────────────────

    def missing_report(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Per-ticker missing value report BEFORE imputation.

        Returns DataFrame with columns:
            missing_count, missing_pct, first_valid, last_valid, longest_gap
        Sorted by missing_pct descending.
        """
        report_rows = []
        for ticker in returns.columns:
            col     = returns[ticker]
            n_miss  = col.isna().sum()
            valid   = col.dropna()

            # Longest consecutive NaN streak
            is_null    = col.isna().astype(int)
            streak     = is_null * (is_null.groupby((is_null != is_null.shift()).cumsum()).cumcount() + 1)
            max_streak = int(streak.max())

            report_rows.append({
                "ticker"        : ticker,
                "missing_count" : n_miss,
                "missing_pct"   : round(n_miss / len(col) * 100, 2),
                "first_valid"   : valid.index.min().date() if not valid.empty else None,
                "last_valid"    : valid.index.max().date() if not valid.empty else None,
                "longest_gap"   : max_streak,
            })

        report = (
            pd.DataFrame(report_rows)
            .set_index("ticker")
            .sort_values("missing_pct", ascending=False)
        )
        return report

    def imputation_delta(
        self,
        original: pd.DataFrame,
        imputed: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Report the values that were changed by imputation.
        Returns a DataFrame of (date, ticker, imputed_value) for all
        positions that were NaN in original and filled in imputed.
        Useful for auditing the imputation quality.
        """
        was_missing = original.isna()
        filled      = imputed[was_missing]

        rows = []
        for col in filled.columns:
            col_filled = filled[col].dropna()
            for date, val in col_filled.items():
                rows.append({"date": date, "ticker": col, "imputed_value": round(val, 6)})

        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # ─────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────

    def _validate(self, returns: pd.DataFrame) -> None:
        if returns.empty:
            raise ValueError("returns DataFrame is empty.")
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise TypeError("returns.index must be a DatetimeIndex.")
        if returns.shape[0] < self.n_neighbors + 1:
            raise ValueError(
                f"Need at least n_neighbors + 1 = {self.n_neighbors + 1} rows, "
                f"got {returns.shape[0]}."
            )

    def _check_plausibility(
        self,
        imputed: pd.DataFrame,
        original: pd.DataFrame,
    ) -> None:
        """
        Warn if any imputed value exceeds MAX_PLAUSIBLE_RETURN in magnitude.
        These are not clipped automatically — they're flagged for inspection.
        """
        was_missing   = original.isna()
        imputed_vals  = imputed[was_missing].values.flatten()
        imputed_vals  = imputed_vals[~np.isnan(imputed_vals)]

        if len(imputed_vals) == 0:
            return

        extreme = np.abs(imputed_vals) > MAX_PLAUSIBLE_RETURN
        n_extreme = extreme.sum()
        if n_extreme > 0:
            logger.warning(
                f"{n_extreme} imputed values exceed |{MAX_PLAUSIBLE_RETURN:.0%}| "
                "single-day return. Review raw data for potential yfinance errors. "
                f"Extreme range: [{imputed_vals[extreme].min():.4f}, "
                f"{imputed_vals[extreme].max():.4f}]"
            )


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level="DEBUG", format=CFG.LOG_FORMAT)
    import numpy as np

    np.random.seed(42)
    dates   = pd.date_range("2020-01-02", periods=30, freq="B")
    data    = np.random.normal(0, 0.01, size=(30, 4))
    data[2, 1]  = np.nan   # single gap
    data[5:8, 2] = np.nan  # 3-day streak
    data[15, 0] = np.nan

    returns = pd.DataFrame(
        data,
        index   = dates,
        columns = ["INFY.NS", "TCS.NS", "WIPRO.NS", "^NSEI"],
    )
    returns.index.name = "Date"

    imputer = ReturnImputer(n_neighbors=5)

    print("=== Missing report (before imputation) ===")
    print(imputer.missing_report(returns))

    imputed = imputer.fit_transform(returns)

    print(f"\n=== Imputed values ===")
    print(imputer.imputation_delta(returns, imputed))

    print(f"\nRemaining NaN: {imputed.isna().sum().sum()}")
    print("OK")