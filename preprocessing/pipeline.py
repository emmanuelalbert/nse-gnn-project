"""
preprocessing/pipeline.py
-------------------------
Orchestrator that runs the full preprocessing sequence on raw closing prices
and produces the clean, imputed log returns ready for shock detection and
graph construction.

Sequence:
    1. ReturnCalculator.compute()   — raw log returns  (paper Eq. 1)
    2. ReturnImputer.fit_transform() — KNN imputation  (paper Section III-A-2c)
    3. [Optional] FeatureNormalizer  — z-score / min-max for TGAT node features
    4. Save imputed returns to CFG.LOG_RETURNS_FILE

This module is the single entry point called by run_pipeline.py for the
preprocessing stage. Individual modules (returns.py, imputer.py, normalizer.py)
can also be used standalone for notebooks and testing.

Usage:
    from preprocessing.pipeline import PreprocessingPipeline
    pipeline = PreprocessingPipeline()
    returns  = pipeline.run(prices_df)        # full run + save to disk
    returns  = pipeline.run(prices_df, save=False)   # in-memory only
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as CFG
from data.storage import save_returns, load_returns, quick_stats, save_snapshot
from preprocessing.returns import ReturnCalculator
from preprocessing.imputer import ReturnImputer
from preprocessing.normalizer import FeatureNormalizer

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    End-to-end preprocessing from raw prices → clean log returns.

    Attributes:
        calc    : ReturnCalculator instance.
        imputer : ReturnImputer instance (fitted during .run()).
        norm    : FeatureNormalizer instance (fitted during .run(), optional).
    """

    def __init__(
        self,
        n_neighbors: int   = CFG.KNN_N_NEIGHBORS,
        norm_method: str   = "none",        # "zscore" | "minmax" | "none"
        force_refresh: bool = False,         # re-run even if output cache exists
    ):
        self.n_neighbors    = n_neighbors
        self.norm_method    = norm_method
        self.force_refresh  = force_refresh

        self.calc    = ReturnCalculator()
        self.imputer = ReturnImputer(n_neighbors=n_neighbors)
        self.norm    = FeatureNormalizer(method=norm_method)

    # ─────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────

    def run(
        self,
        prices: pd.DataFrame,
        save: bool = True,
        snapshot: bool = False,
    ) -> pd.DataFrame:
        """
        Execute the full preprocessing pipeline.

        Args:
            prices   : Raw adjusted closing prices (T, N) from DataCollector.
            save     : If True, save imputed returns to CFG.LOG_RETURNS_FILE.
            snapshot : If True, save a timestamped snapshot as well.

        Returns:
            Imputed log returns DataFrame (T-1, N), ready for downstream use.
        """
        # ── Load from cache if available and not forced ──────────────
        if not self.force_refresh and CFG.LOG_RETURNS_FILE.exists():
            logger.info(
                f"Loading cached log returns from {CFG.LOG_RETURNS_FILE}. "
                "Pass force_refresh=True to recompute."
            )
            returns = load_returns()
            quick_stats(returns, label="cached log returns")
            return returns

        logger.info("=" * 60)
        logger.info("PREPROCESSING PIPELINE START")
        logger.info("=" * 60)

        # ── Step 1: Log returns ──────────────────────────────────────
        logger.info("[1/3] Computing log returns (paper Eq. 1)...")
        raw_returns = self.calc.compute(prices)
        quick_stats(raw_returns, label="raw log returns")

        # ── Step 2: KNN Imputation ───────────────────────────────────
        logger.info(f"[2/3] Imputing missing values (KNN, k={self.n_neighbors})...")
        missing_report = self.imputer.missing_report(raw_returns)
        n_missing = raw_returns.isna().sum().sum()

        if n_missing > 0:
            logger.info(
                f"    Missing values before imputation: {n_missing} "
                f"({n_missing / raw_returns.size * 100:.3f}%)"
            )
            # Fit and transform on the full dataset
            # (for Monte Carlo splits, re-run with split data via run_split())
            imputed_returns = self.imputer.fit_transform(raw_returns)
        else:
            logger.info("    No missing values — skipping KNN imputation.")
            imputed_returns = raw_returns.copy()

        quick_stats(imputed_returns, label="imputed log returns")

        # ── Step 3: Optional normalisation ──────────────────────────
        if self.norm_method != "none":
            logger.info(f"[3/3] Normalising features (method={self.norm_method})...")
            imputed_returns = self.norm.fit_transform(imputed_returns)
            logger.info(f"    Normalisation stats:\n{self.norm.stats}")
        else:
            logger.info("[3/3] Normalisation skipped (method='none').")

        # ── Save ─────────────────────────────────────────────────────
        if save:
            save_returns(imputed_returns)
            logger.info(f"Saved imputed returns → {CFG.LOG_RETURNS_FILE}")

        if snapshot:
            save_snapshot(imputed_returns, name="log_returns")

        logger.info("PREPROCESSING PIPELINE COMPLETE")
        logger.info(
            f"Output shape: {imputed_returns.shape[0]} trading days × "
            f"{imputed_returns.shape[1]} tickers"
        )
        return imputed_returns

    # ─────────────────────────────────────────
    # MONTE CARLO SPLIT-AWARE PREPROCESSING
    # ─────────────────────────────────────────

    def run_split(
        self,
        raw_returns: pd.DataFrame,
        train_idx: pd.DatetimeIndex,
        test_idx: pd.DatetimeIndex,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocessing for a single Monte Carlo train/test split.

        The KNN imputer is fitted on training data ONLY, then applied to
        both train and test. This prevents any leakage of future data
        statistics into the imputation of training values.

        This is called by training/monte_carlo.py for each of the 10 splits.

        Args:
            raw_returns : Full unimputed log returns DataFrame.
            train_idx   : DatetimeIndex of training rows.
            test_idx    : DatetimeIndex of test rows.

        Returns:
            (train_imputed, test_imputed) — both DataFrames fully imputed.
        """
        train_raw = raw_returns.loc[train_idx]
        test_raw  = raw_returns.loc[test_idx]

        # Fit imputer on training data only
        split_imputer = ReturnImputer(n_neighbors=self.n_neighbors)
        train_imputed = split_imputer.fit(train_raw).transform(train_raw)
        test_imputed  = split_imputer.transform(test_raw)

        logger.debug(
            f"Split preprocessing: "
            f"train {train_imputed.shape[0]} rows, "
            f"test  {test_imputed.shape[0]} rows | "
            f"remaining NaN train={train_imputed.isna().sum().sum()}, "
            f"test={test_imputed.isna().sum().sum()}"
        )
        return train_imputed, test_imputed

    # ─────────────────────────────────────────
    # DIAGNOSTICS
    # ─────────────────────────────────────────

    def full_report(self, prices: pd.DataFrame) -> None:
        """
        Print a comprehensive preprocessing diagnostics report.
        Useful for the EDA notebook before committing to a pipeline run.
        """
        print("=" * 60)
        print("PREPROCESSING DIAGNOSTICS REPORT")
        print("=" * 60)

        print(f"\n[Prices] shape={prices.shape}")
        print(f"  Date range : {prices.index.min().date()} → {prices.index.max().date()}")
        print(f"  Missing    : {prices.isna().sum().sum()} values "
              f"({prices.isna().mean().mean() * 100:.3f}%)")

        raw_returns = self.calc.compute(prices)
        print(f"\n[Log Returns] shape={raw_returns.shape}")

        missing_report = self.imputer.missing_report(raw_returns)
        print(f"\n[Missing Values — Top 10 tickers]")
        print(missing_report.head(10).to_string())

        print(f"\n[Return Distribution — sample 5 tickers]")
        print(self.calc.distribution_report(raw_returns).head(5).to_string())

        print(f"\n[Long-term Stats for shock reference ({CFG.SHOCK_REFERENCE_TICKER})]")
        stats = self.calc.long_term_stats(raw_returns)
        if CFG.SHOCK_REFERENCE_TICKER in stats.index:
            ref = stats.loc[CFG.SHOCK_REFERENCE_TICKER]
            print(f"  μ  = {ref['mean']:.6f}")
            print(f"  σ  = {ref['std']:.6f}")
            print(f"  2σ = {2 * ref['std']:.6f}  (shock threshold)")
        print("=" * 60)


# ─────────────────────────────────────────────
# CLI  —  python -m preprocessing.pipeline
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=CFG.LOG_LEVEL, format=CFG.LOG_FORMAT)

    from data.storage import load_prices

    try:
        prices = load_prices()
    except FileNotFoundError:
        logger.error(
            "Raw prices not found. Run `python -m data.collector` first."
        )
        sys.exit(1)

    pipeline = PreprocessingPipeline(force_refresh=True)
    pipeline.full_report(prices)

    returns = pipeline.run(prices, save=True, snapshot=True)
    print(f"\nFinal returns shape: {returns.shape}")
    print(returns.describe().round(6))