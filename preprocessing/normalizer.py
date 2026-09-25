"""
preprocessing/normalizer.py
---------------------------
Optional feature normalisation for the node feature matrix fed into the
TGAT model.

Context in the pipeline:
  - The paper uses the "mean of returns" as the scalar node feature for each
    stock during a shock period (paper Section III-B, Table 1).
  - Log returns are already roughly zero-centred and scale-invariant across
    tickers, so normalisation is less critical than for raw prices.
  - However, when additional node features are added (e.g. volatility,
    volume z-score) alongside mean return, their scales can differ by orders
    of magnitude. Normalisation then becomes important for stable TGAT training.
  - This module provides Z-score (StandardScaler) and Min-Max scaling,
    both fitted on training data only to avoid leakage into test periods.

Normalisation strategy (recommended):
  - Z-score per feature across the training shock periods only.
  - Transform both train and test with the training statistics.
  - Keep the scaler object to inverse-transform for interpretability.

Usage:
    from preprocessing.normalizer import FeatureNormalizer
    norm       = FeatureNormalizer(method="zscore")
    train_norm = norm.fit_transform(train_features)   # shape (n_shocks, n_tickers, n_features)
    test_norm  = norm.transform(test_features)
    original   = norm.inverse_transform(train_norm)
"""

import logging
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as CFG

logger = logging.getLogger(__name__)

NormMethod = Literal["zscore", "minmax", "none"]


class FeatureNormalizer:
    """
    Fits and applies feature normalisation to the TGAT node feature matrix.

    Supports:
      - "zscore"  : (x - mean) / std  (recommended for log-return-based features)
      - "minmax"  : (x - min) / (max - min)  (useful when feature range matters)
      - "none"    : pass-through (use when features are already well-scaled)

    Input shape: 2D DataFrame (n_samples, n_features) where each row is one
    ticker-period observation. The normalizer is fitted per COLUMN (feature).
    """

    def __init__(self, method: NormMethod = "zscore"):
        if method not in ("zscore", "minmax", "none"):
            raise ValueError(f"method must be 'zscore', 'minmax', or 'none'. Got '{method}'.")
        self.method      = method
        self._is_fitted  = False
        # Statistics stored at fit time
        self._mean: Optional[pd.Series] = None
        self._std:  Optional[pd.Series] = None
        self._min:  Optional[pd.Series] = None
        self._max:  Optional[pd.Series] = None

    # ─────────────────────────────────────────
    # MAIN API
    # ─────────────────────────────────────────

    def fit(self, features: pd.DataFrame) -> "FeatureNormalizer":
        """
        Compute normalisation statistics from training features.

        Args:
            features: DataFrame of shape (n_samples, n_features).
                      Should contain TRAINING data only.

        Returns:
            self (for method chaining).
        """
        if self.method == "none":
            self._is_fitted = True
            return self

        self._validate(features)

        if self.method == "zscore":
            self._mean = features.mean(skipna=True)
            self._std  = features.std(skipna=True, ddof=1)
            # Guard: replace zero std with 1.0 to avoid division by zero
            # (e.g. a feature that is constant across all training samples)
            zero_std = self._std == 0
            if zero_std.any():
                logger.warning(
                    f"Zero std in features: {self._std[zero_std].index.tolist()} — "
                    "setting std=1.0 for those features."
                )
                self._std = self._std.where(~zero_std, 1.0)

            logger.info(
                f"Z-score normalizer fitted on {features.shape[0]} samples × "
                f"{features.shape[1]} features. "
                f"Mean range: [{self._mean.min():.4f}, {self._mean.max():.4f}], "
                f"Std range:  [{self._std.min():.4f}, {self._std.max():.4f}]"
            )

        elif self.method == "minmax":
            self._min = features.min(skipna=True)
            self._max = features.max(skipna=True)
            zero_range = self._max == self._min
            if zero_range.any():
                logger.warning(
                    f"Zero range in features: {self._max[zero_range].index.tolist()} — "
                    "min-max will return 0 for those features."
                )
            logger.info(
                f"Min-max normalizer fitted on {features.shape[0]} samples × "
                f"{features.shape[1]} features."
            )

        self._is_fitted = True
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted normalisation to a features DataFrame.

        Args:
            features: DataFrame of shape (n_samples, n_features).
                      Columns must match those used at fit time.

        Returns:
            Normalised DataFrame of the same shape.
        """
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before .transform().")

        if self.method == "none":
            return features.copy()

        self._validate(features)
        self._check_columns(features)

        if self.method == "zscore":
            return (features - self._mean) / self._std

        elif self.method == "minmax":
            denom = self._max - self._min
            # Where range is zero, return 0.0
            denom = denom.where(denom != 0, other=1.0)
            return (features - self._min) / denom

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one call (for training data only)."""
        return self.fit(features).transform(features)

    def inverse_transform(self, features_norm: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse the normalisation to recover original-scale features.
        Useful for interpreting model outputs in their original units.
        """
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before .inverse_transform().")

        if self.method == "none":
            return features_norm.copy()

        self._check_columns(features_norm)

        if self.method == "zscore":
            return (features_norm * self._std) + self._mean

        elif self.method == "minmax":
            denom = self._max - self._min
            denom = denom.where(denom != 0, other=1.0)
            return (features_norm * denom) + self._min

    # ─────────────────────────────────────────
    # PERIOD-LEVEL NODE FEATURE BUILDER
    # ─────────────────────────────────────────

    @staticmethod
    def build_node_features(
        returns: pd.DataFrame,
        shock_periods: list,
        feature_cols: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Build a node feature DataFrame for all shock periods.

        For each shock period, computes the selected features per ticker
        and stacks them into a single DataFrame:
            index: MultiIndex (shock_name, ticker)
            columns: feature names (e.g. ["mean_return", "volatility"])

        This output feeds directly into the TGAT dataset builder
        (graph/dataset.py).

        Args:
            returns      : Full log returns DataFrame (T, N).
            shock_periods: List of shock dicts from settings.TARGET_SHOCK_PERIODS
                           or detector output. Each must have "name", "start", "end".
            feature_cols : List of feature names to compute. Defaults to
                           CFG.NODE_FEATURE_COLS = ["mean_return"].

        Returns:
            DataFrame with MultiIndex (shock_name, ticker) and feature columns.
        """
        feature_cols = feature_cols or CFG.NODE_FEATURE_COLS
        all_rows = []

        for shock in shock_periods:
            name  = shock["name"]
            start = shock["start"]
            end   = shock["end"]

            mask   = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
            period = returns.loc[mask]

            if period.empty:
                logger.warning(f"No return data for shock '{name}' [{start} → {end}]. Skipping.")
                continue

            for ticker in returns.columns:
                row = {"shock_name": name, "ticker": ticker}

                if "mean_return" in feature_cols:
                    row["mean_return"] = period[ticker].mean(skipna=True)

                if "volatility" in feature_cols:
                    row["volatility"] = period[ticker].std(skipna=True, ddof=1)

                if "min_return" in feature_cols:
                    row["min_return"] = period[ticker].min(skipna=True)

                if "max_return" in feature_cols:
                    row["max_return"] = period[ticker].max(skipna=True)

                all_rows.append(row)

        df = pd.DataFrame(all_rows).set_index(["shock_name", "ticker"])
        logger.info(
            f"Node feature matrix built: {df.shape[0]} rows "
            f"({len(shock_periods)} shocks × {returns.shape[1]} tickers) × "
            f"{df.shape[1]} features"
        )
        return df

    # ─────────────────────────────────────────
    # PROPERTIES (for inspection)
    # ─────────────────────────────────────────

    @property
    def stats(self) -> Optional[pd.DataFrame]:
        """Return the fitted statistics as a DataFrame (for logging/audit)."""
        if not self._is_fitted or self.method == "none":
            return None
        if self.method == "zscore":
            return pd.DataFrame({"mean": self._mean, "std": self._std})
        elif self.method == "minmax":
            return pd.DataFrame({"min": self._min, "max": self._max})

    # ─────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────

    def _validate(self, features: pd.DataFrame) -> None:
        if features.empty:
            raise ValueError("features DataFrame is empty.")
        if features.shape[0] < 2:
            raise ValueError(f"Need ≥ 2 rows for normalisation, got {features.shape[0]}.")

    def _check_columns(self, features: pd.DataFrame) -> None:
        if self.method == "none":
            return
        ref_cols = (self._mean if self.method == "zscore" else self._min).index
        missing  = set(ref_cols) - set(features.columns)
        extra    = set(features.columns) - set(ref_cols)
        if missing:
            raise ValueError(f"Features missing columns seen at fit time: {missing}")
        if extra:
            logger.warning(f"Extra columns not seen at fit time (will be ignored): {extra}")


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level="DEBUG", format=CFG.LOG_FORMAT)
    np.random.seed(0)

    features = pd.DataFrame(
        np.random.normal(0, 0.01, size=(20, 3)),
        columns=["mean_return", "volatility", "min_return"],
    )

    # Z-score
    norm  = FeatureNormalizer(method="zscore")
    normed = norm.fit_transform(features)
    print("Z-score stats:\n", norm.stats)
    print("Normed mean (should ≈ 0):", normed.mean().round(4).to_dict())
    print("Normed std  (should ≈ 1):", normed.std().round(4).to_dict())

    recovered = norm.inverse_transform(normed)
    max_err   = (recovered - features).abs().max().max()
    print(f"Max inverse-transform error: {max_err:.2e} (should be ~0)")

    # Min-max
    norm2  = FeatureNormalizer(method="minmax")
    normed2 = norm2.fit_transform(features)
    print("\nMin-max normed range:", normed2.min().round(4).to_dict(), "→", normed2.max().round(4).to_dict())
    print("OK")