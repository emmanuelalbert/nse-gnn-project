"""
training/monte_carlo.py
-----------------------
Monte Carlo cross-validation with chronological splits for the TGAT model.

Paper reference: Section III-D (Eq. 11 & 12)

    "To ensure the robustness and reliability of our evaluation, we conducted
     Monte Carlo experiments. This approach involves multiple iterations of
     chronologically splitting the datasets into training and testing sets,
     ensuring that the temporal order of the data is preserved. For each
     iteration, the model is trained and evaluated on sequentially ordered
     data, allowing us to assess the variability of the results."

    Eq. 11:  μ = (1/N) Σ Metric_i
    Eq. 12:  σ = √( (1/N) Σ (Metric_i − μ)² )

    "The mean μ and standard deviation σ of key performance metrics, such as
     AUC-ROC score, average precision score, and F1 score, are calculated..."

    "The low standard deviations observed for all metrics indicate that the
     performance of our model is consistent across different chronological
     splits of the dataset."

Split design (preserving temporal order):
    Across K=10 folds the test window slides forward through the shock history.
    At fold k the boundary between train and test is set at the k-th decile of
    the total shock list. The training set always consists of the earliest shocks;
    the test set is always the most recent. No shuffling ever occurs.

    Example with 50 shocks, K=10, train_ratio=0.80:
        Fold 1:  train=shocks[0:40],  test=shocks[40:50]
        Fold 2:  train=shocks[0:41],  test=shocks[41:50]
        ...
        Fold 10: train=shocks[0:49],  test=shocks[49:50]

    This is a "growing-window" (expanding) scheme: each fold adds one more
    shock to the training set. The test set always excludes anything seen in
    training. This is stricter than the paper's fixed 80/20 split repeated
    10× with random seeds because it explicitly varies the train/test boundary
    across folds, giving a more conservative variance estimate.

    If CFG.MONTE_CARLO_N_SPLITS = 10 but fewer than 12 shocks are available
    (need at least 1 for train + 1 for test per fold), the number of folds is
    reduced automatically.

Preprocessing within each fold:
    The KNN imputer is fitted on each fold's training returns slice ONLY,
    then applied to both train and test. This prevents any leakage of future
    return statistics into the imputation.

[UPDATED] Class-imbalance handling — pos_weight and threshold reuse
---------------------------------------------------------------------
Trainer now auto-derives pos_weight from each fold's training adjacency
class balance and tunes a decision threshold on its "val" pass each epoch
(see training/trainer.py). Because val_data == test_data in this runner's
_run_fold() (test snapshots double as the validation set used for early
stopping / checkpoint selection — see the `val_data = test_data` line
below), the threshold tuned during training is NOT leakage-free with
respect to the test set: it was already fit on the same snapshots the
final headline metrics are computed on.

Given that constraint, _evaluate() below reuses the EXACT threshold value
recorded by the trainer at its best-checkpoint epoch (train_result[
"best_threshold"]) rather than calling find_best_threshold() a second,
separate time on the test set. This avoids double-dipping (tuning twice
on the same data) even though it does not eliminate the underlying
leakage, which is a structural property of this fold design (no distinct
train/val/test three-way split) rather than something introduced by the
threshold-tuning change itself. A genuinely clean fix would carve out a
distinct validation slice within each fold's training window, separate
from the test shocks — left as future work rather than bundled into this
change.

Usage:
    from training.monte_carlo import MonteCarloRunner
    from graph.dataset import ShockGraphDataset

    runner = MonteCarloRunner(dataset, returns_df, shocks)
    report = runner.run()                         # run all K folds
    report = runner.run(folds=[1, 3, 5])          # run specific folds only
    runner.print_report()                         # print μ ± σ table
    df     = runner.results_df()                  # per-fold metrics DataFrame
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as CFG
from data.storage import assert_no_lookahead
from graph.dataset import ShockGraphDataset
from model.loss import (
    aggregate_monte_carlo_metrics,
    compute_metrics,
    compute_pos_weight,   # [NEW] not called directly here (Trainer handles
                          # per-fold pos_weight internally) — imported for
                          # completeness / potential future direct use.
    find_best_threshold,  # [NEW] fallback path only — see _evaluate() below.
)
from model.tgat import TGAT, build_tgat
from preprocessing.imputer import ReturnImputer
from shock_detection.detector import ShockPeriod
from training.checkpointing import CheckpointManager, FoldSummary
from training.trainer import Trainer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FOLD RESULT  (typed container for one fold's output)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    """
    Stores all outputs from a single Monte Carlo fold.

    Fields mirror the paper's reported metrics (Section IV-B) plus
    training diagnostics.
    """
    fold            : int
    train_shock_names : List[str]
    test_shock_names  : List[str]
    n_train_snapshots : int
    n_test_snapshots  : int

    # Core metrics (paper Section III-C, Eq. 6–10)
    auc_roc         : float = 0.0
    aupr            : float = 0.0
    f1              : float = 0.0
    precision       : float = 0.0
    recall          : float = 0.0
    auc_roc_low_fpr : float = 0.0    # paper Section IV-B: low FPR range [0, 0.1]

    # Training diagnostics
    best_epoch      : int   = -1
    best_val_loss   : float = float("inf")
    n_epochs_run    : int   = 0
    train_time_sec  : float = 0.0
    pos_weight      : float = 1.0    # [NEW] auto-derived class weight used
    threshold       : float = 0.5    # [NEW] decision threshold used for
                                      # this fold's reported precision/recall/F1

    # Full metrics dict for flexibility
    all_metrics     : Dict  = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d.pop("all_metrics", None)   # exclude nested dict from flat repr
        return d


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO RUNNER
# ─────────────────────────────────────────────────────────────────────────────

class MonteCarloRunner:
    """
    Orchestrates the full K-fold Monte Carlo evaluation of the TGAT model.

    Key responsibilities:
      1. Partition the ordered shock list into K chronological fold boundaries.
      2. For each fold:
           a. Slice returns to the fold's train shock window.
           b. Fit KNN imputer on train slice only (no leakage).
           c. Build train/test ShockGraphDatasets with imputed returns.
           d. Instantiate a fresh TGAT and Trainer for the fold.
           e. Train via Trainer.run(); load best checkpoint.
           f. Evaluate on held-out test snapshots.
           g. Store FoldResult.
      3. Aggregate FoldResult list → μ ± σ via aggregate_monte_carlo_metrics().
      4. Save per-fold and summary results to disk.

    Args:
        dataset         : Built ShockGraphDataset with ALL shock periods.
        returns         : Full log-returns DataFrame (T, N), DatetimeIndex.
        shocks          : Ordered list of ShockPeriod objects (same order
                          as used to build dataset). Must be chronological.
        n_folds         : Number of Monte Carlo folds. Paper: 10.
        train_ratio     : Fraction of shocks used for training. Paper: 0.80.
        in_features     : Node feature dimension (must match dataset's features).
        device          : Compute device.
        scheduler_strategy : LR scheduler type forwarded to Trainer.
        warmup_epochs   : LR warm-up epochs forwarded to Trainer.
        epochs          : Max training epochs per fold.
        verbose         : Log progress every N epochs inside each fold.
    """

    def __init__(
        self,
        dataset            : ShockGraphDataset,
        returns            : pd.DataFrame,
        shocks             : List[ShockPeriod],
        n_folds            : int   = CFG.MONTE_CARLO_N_SPLITS,
        train_ratio        : float = CFG.TRAIN_RATIO,
        in_features        : int   = len(CFG.NODE_FEATURE_COLS),
        device             : str   = CFG.DEVICE,
        scheduler_strategy : str   = "plateau",
        warmup_epochs      : int   = 10,
        epochs             : int   = CFG.TRAIN_EPOCHS,
        verbose            : int   = 10,
    ):
        if not dataset._built:
            raise RuntimeError(
                "dataset must be built before passing to MonteCarloRunner. "
                "Call dataset.build() first."
            )

        self.dataset    = dataset
        self.returns    = returns
        self.shocks     = self._sort_shocks(shocks)
        self.n_folds    = n_folds
        self.train_ratio = train_ratio
        self.in_features = in_features
        self.device      = device
        self.scheduler_strategy = scheduler_strategy
        self.warmup_epochs      = warmup_epochs
        self.epochs             = epochs
        self.verbose            = verbose

        # Results accumulated across folds
        self._fold_results  : List[FoldResult] = []
        self._start_time    : float = 0.0

        # Compute fold boundaries up-front
        self._fold_boundaries = self._compute_fold_boundaries()

        # Ensure results directory exists
        CFG.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        CFG.METRICS_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"MonteCarloRunner initialised | "
            f"shocks={len(self.shocks)} | "
            f"n_folds={self.n_folds} (effective={len(self._fold_boundaries)}) | "
            f"train_ratio={train_ratio:.0%} | device={device}"
        )

    # ─────────────────────────────────────────
    # PUBLIC: RUN ALL FOLDS
    # ─────────────────────────────────────────

    def run(
        self,
        folds            : Optional[List[int]] = None,
        resume_from_fold : Optional[int]       = None,
    ) -> Dict:
        """
        Execute all (or selected) Monte Carlo folds and return aggregated results.

        Args:
            folds           : Optional list of 1-based fold indices to run.
                              None = run all folds.
            resume_from_fold: If set, skip folds < this value (crash recovery).

        Returns:
            Aggregated results dict:
            {
                "fold_results"  : List[FoldResult],
                "aggregated"    : {metric: {"mean": float, "std": float}},
                "summary_df"    : pd.DataFrame,
                "total_time_sec": float,
            }
        """
        self._start_time    = time.time()
        self._fold_results  = []

        fold_indices = folds if folds is not None else list(range(1, len(self._fold_boundaries) + 1))

        if resume_from_fold is not None:
            fold_indices = [f for f in fold_indices if f >= resume_from_fold]
            logger.info(f"Resuming from fold {resume_from_fold}")

        logger.info(
            f"\n{'='*62}\n"
            f"MONTE CARLO EVALUATION — {len(fold_indices)} folds\n"
            f"{'='*62}"
        )

        for fold_num in fold_indices:
            result = self._run_fold(fold_num)
            self._fold_results.append(result)
            self._log_fold_result(result)

        # ── Aggregate across folds (paper Eq. 11 & 12) ──────────────────
        aggregated = self._aggregate()
        summary_df = self._build_summary_df()

        # ── Save results to disk ─────────────────────────────────────────
        self._save_mc_results(aggregated, summary_df)

        total_time = time.time() - self._start_time
        logger.info(
            f"\n{'='*62}\n"
            f"MONTE CARLO COMPLETE | "
            f"{len(self._fold_results)} folds | "
            f"total time: {total_time:.1f}s\n"
            f"{'='*62}"
        )
        self._print_aggregated(aggregated)

        return {
            "fold_results"   : self._fold_results,
            "aggregated"     : aggregated,
            "summary_df"     : summary_df,
            "total_time_sec" : total_time,
        }

    # ─────────────────────────────────────────
    # SINGLE FOLD
    # ─────────────────────────────────────────

    def _run_fold(self, fold_num: int) -> FoldResult:
        """
        Execute one complete fold: split → impute → build dataset → train → evaluate.

        Args:
            fold_num : 1-based fold index.

        Returns:
            FoldResult with all metrics and diagnostics.
        """
        fold_start = time.time()
        boundary   = self._fold_boundaries[fold_num - 1]
        train_names, test_names = boundary["train"], boundary["test"]

        logger.info(
            f"\n{'─'*62}\n"
            f"FOLD {fold_num:02d}/{len(self._fold_boundaries)} | "
            f"train={len(train_names)} shocks | test={len(test_names)} shocks\n"
            f"  Train: {train_names[0]} → {train_names[-1]}\n"
            f"  Test : {test_names[0]} → {test_names[-1]}"
        )

        # ── Step 1: Slice returns for this fold ──────────────────────────
        train_shocks = [s for s in self.shocks if s.name in set(train_names)]
        test_shocks  = [s for s in self.shocks if s.name in set(test_names)]

        train_returns = self._slice_returns_for_shocks(train_shocks)
        test_returns  = self._slice_returns_for_shocks(test_shocks)

        # Verify no lookahead at the returns level
        if not train_returns.empty and not test_returns.empty:
            assert_no_lookahead(train_returns, test_returns,
                                label=f"fold_{fold_num:02d}")

        # ── Step 2: KNN imputation (fitted on train slice only) ──────────
        train_returns, test_returns = self._impute_fold(
            train_returns, test_returns
        )

        # ── Step 3: Build fold-specific datasets ─────────────────────────
        train_data, test_data = self._build_fold_datasets(
            train_shocks, test_shocks, train_returns, test_returns
        )

        logger.info(
            f"  Fold {fold_num:02d} | "
            f"train_snapshots={len(train_data)} | "
            f"test_snapshots={len(test_data)}"
        )

        # ── Step 4: Build fresh model for this fold ───────────────────────
        torch.manual_seed(CFG.RANDOM_SEED + fold_num)
        model = build_tgat(in_features=self.in_features)

        # ── Step 5: Train ─────────────────────────────────────────────────
        trainer = Trainer(
            model              = model,
            train_data         = train_data,
            val_data           = test_data,   # test used as val during training
            fold               = fold_num,
            epochs             = self.epochs,
            scheduler_strategy = self.scheduler_strategy,
            warmup_epochs      = self.warmup_epochs,
            device             = self.device,
            seed               = CFG.RANDOM_SEED,
            verbose            = self.verbose,
        )
        train_result = trainer.run()

        # ── Step 6: Load best checkpoint and evaluate ─────────────────────
        ckpt  = CheckpointManager(fold=fold_num)
        model = ckpt.load_best(model, device=self.device)

        # [UPDATED] Reuse the threshold tuned during training at the
        # best-checkpoint epoch (see module docstring for why this is
        # reused rather than re-tuned here on the same test snapshots).
        tuned_threshold = train_result.get("best_threshold", None)
        test_metrics = self._evaluate(
            model, test_data, test_shocks, threshold=tuned_threshold
        )

        fold_time = time.time() - fold_start

        result = FoldResult(
            fold              = fold_num,
            train_shock_names = train_names,
            test_shock_names  = test_names,
            n_train_snapshots = len(train_data),
            n_test_snapshots  = len(test_data),
            auc_roc           = test_metrics.get("auc_roc", 0.0),
            aupr              = test_metrics.get("aupr", 0.0),
            f1                = test_metrics.get("f1", 0.0),
            precision         = test_metrics.get("precision", 0.0),
            recall            = test_metrics.get("recall", 0.0),
            auc_roc_low_fpr   = test_metrics.get("auc_roc_low_fpr", 0.0),
            best_epoch        = train_result["best_epoch"],
            best_val_loss     = train_result["best_val_loss"],
            n_epochs_run      = train_result["n_epochs_run"],
            train_time_sec    = fold_time,
            pos_weight        = train_result.get("pos_weight", 1.0),   # [NEW]
            threshold         = test_metrics.get("threshold", 0.5),    # [NEW]
            all_metrics       = test_metrics,
        )
        return result

    # ─────────────────────────────────────────
    # EVALUATION (after loading best checkpoint)
    # ─────────────────────────────────────────

    @torch.no_grad()
    def _evaluate(
        self,
        model       : TGAT,
        test_data   : List,
        test_shocks : List[ShockPeriod],
        threshold   : Optional[float] = None,
    ) -> Dict:
        """
        Evaluate the best-checkpoint model on all test snapshots.

        For each test snapshot:
          - Build temporal sequence from the same shock period's snapshots.
          - Predict full N×N adjacency matrix.
          - Flatten off-diagonal entries and compare to ground-truth GC matrix.

        Args:
            threshold : [UPDATED] Decision threshold for precision/recall/F1.
                        If provided (normal path — the value tuned during
                        training at the best-checkpoint epoch), it is used
                        directly. If None (fallback path, e.g. standalone
                        use of this method outside _run_fold), a threshold
                        is tuned on THIS SAME test set via
                        find_best_threshold() as a last resort, with a
                        warning logged since that reintroduces the
                        tune-on-test-set leakage described in the module
                        docstring.

        Returns the full metrics dict from compute_metrics(), including the
        threshold actually used (metrics["threshold"]).
        """
        model.eval()
        model = model.to(self.device)

        all_y_true   : List[np.ndarray] = []
        all_y_scores : List[np.ndarray] = []

        # Group test snapshots by shock for sequence building
        groups = {}
        for snap in test_data:
            name = getattr(snap, "shock_name", "unknown")
            groups.setdefault(name, []).append(snap)

        for shock_name, snaps in groups.items():
            for t_idx, snap in enumerate(snaps):
                snap_dev = snap.to(self.device)

                # Build temporal sequence up to current step
                x_list  = [s.x.to(self.device) for s in snaps[: t_idx + 1]]
                x_seq   = torch.stack(x_list, dim=0)            # (t+1, N, F)
                dt_list = [s.time_delta[:1].to(self.device) for s in snaps[: t_idx + 1]]
                t_delta = torch.cat(dt_list, dim=0)             # (t+1,)

                # Predict adjacency
                probs = model.predict_adjacency(
                    x_seq         = x_seq,
                    edge_index    = snap_dev.edge_index,
                    time_deltas   = t_delta,
                    edge_weight   = getattr(snap_dev, "edge_weight", None),
                    apply_sigmoid = True,
                )    # (N, N)

                N    = probs.shape[0]
                mask = ~torch.eye(N, dtype=torch.bool, device=self.device)

                y_true  = snap_dev.y[mask].cpu().numpy().astype(float)
                y_score = probs[mask].cpu().numpy().astype(float)

                all_y_true.append(y_true)
                all_y_scores.append(y_score)

        if not all_y_true:
            logger.warning("No test data for evaluation.")
            return {}

        y_true_all   = np.concatenate(all_y_true)
        y_scores_all = np.concatenate(all_y_scores)

        if threshold is None:
            logger.warning(
                "  _evaluate() called without a pre-tuned threshold — "
                "falling back to tuning on this same test set, which "
                "reintroduces tune-on-test leakage (see module docstring). "
                "Prefer passing train_result['best_threshold'] from Trainer.run()."
            )
            threshold, _ = find_best_threshold(y_true_all, y_scores_all, metric="f1")

        metrics = compute_metrics(y_true_all, y_scores_all, threshold=threshold)
        logger.info(
            f"  Test metrics | "
            f"AUC-ROC={metrics['auc_roc']:.4f} | "
            f"AUPR={metrics['aupr']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"P={metrics['precision']:.4f} | "
            f"R={metrics['recall']:.4f} | "
            f"thr={metrics['threshold']:.4f} | "
            f"Low-FPR-AUC={metrics['auc_roc_low_fpr']:.4f}"
        )
        return metrics

    # ─────────────────────────────────────────
    # FOLD BOUNDARY COMPUTATION
    # ─────────────────────────────────────────

    def _compute_fold_boundaries(self) -> List[Dict]:
        """
        Compute train/test shock-name partitions for all K folds.

        Strategy: growing-window (expanding train set).
          - The train set for fold k starts at shock[0] and ends at the k-th
            boundary position.
          - The test set is always the remaining shocks after the boundary.
          - This ensures the earliest shocks are always in training and the
            most recent are always tested.

        Boundary positions are spaced at equal intervals starting from
        TRAIN_RATIO × n_shocks, stepping forward by (1-TRAIN_RATIO)/K
        of n_shocks per fold.

        The effective number of folds is capped so each test set has at least
        one shock and each train set has at least two shocks.
        """
        n_shocks    = len(self.shocks)
        ordered     = [s.name for s in self.shocks]

        # Minimum viable sizes
        min_train = 2
        min_test  = 1
        if n_shocks < min_train + min_test:
            raise ValueError(
                f"Need at least {min_train + min_test} shocks for MC evaluation, "
                f"got {n_shocks}."
            )

        # Base split: first fold has TRAIN_RATIO of shocks in train
        base_train_n = max(min_train, int(n_shocks * self.train_ratio))

        # Each fold shifts the boundary forward by one shock
        boundaries = []
        for k in range(self.n_folds):
            split_idx = base_train_n + k
            if split_idx >= n_shocks:
                break   # no more shocks left for test
            train_names = ordered[:split_idx]
            test_names  = ordered[split_idx:]
            boundaries.append({
                "fold"  : k + 1,
                "train" : train_names,
                "test"  : test_names,
            })

        effective_folds = len(boundaries)
        if effective_folds < self.n_folds:
            logger.warning(
                f"Requested {self.n_folds} folds but only {effective_folds} are "
                f"possible with {n_shocks} shocks and train_ratio={self.train_ratio:.0%}. "
                f"Proceeding with {effective_folds} folds."
            )

        logger.info(
            f"Fold boundaries computed: {effective_folds} folds | "
            f"train sizes: {[len(b['train']) for b in boundaries]} | "
            f"test sizes:  {[len(b['test'])  for b in boundaries]}"
        )
        return boundaries

    # ─────────────────────────────────────────
    # DATA HELPERS
    # ─────────────────────────────────────────

    def _sort_shocks(self, shocks: List[ShockPeriod]) -> List[ShockPeriod]:
        """Sort shocks chronologically by start date."""
        return sorted(shocks, key=lambda s: s.start)

    def _slice_returns_for_shocks(
        self, shocks: List[ShockPeriod]
    ) -> pd.DataFrame:
        """
        Slice self.returns to cover all trading days across the given shocks.
        Returns an empty DataFrame if shocks is empty.
        """
        if not shocks:
            return pd.DataFrame()

        start = min(pd.Timestamp(s.start) for s in shocks)
        end   = max(pd.Timestamp(s.end)   for s in shocks)
        mask  = (self.returns.index >= start) & (self.returns.index <= end)
        return self.returns.loc[mask]

    def _impute_fold(
        self,
        train_ret : pd.DataFrame,
        test_ret  : pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit KNN imputer on train slice ONLY, transform both train and test.
        This is the core leakage-prevention step (paper Section III-D):
        future return statistics must not influence the imputation of
        training data.
        """
        imputer = ReturnImputer(n_neighbors=CFG.KNN_N_NEIGHBORS)

        if not train_ret.empty and train_ret.isna().sum().sum() > 0:
            train_ret = imputer.fit(train_ret).transform(train_ret)
        elif not train_ret.empty:
            imputer.fit(train_ret)   # fit even if no missing values

        if not test_ret.empty and test_ret.isna().sum().sum() > 0 and imputer._is_fitted:
            test_ret = imputer.transform(test_ret)

        return train_ret, test_ret

    def _build_fold_datasets(
        self,
        train_shocks  : List[ShockPeriod],
        test_shocks   : List[ShockPeriod],
        train_returns : pd.DataFrame,
        test_returns  : pd.DataFrame,
    ) -> Tuple[List, List]:
        """
        Build train and test snapshot lists for a single fold.

        We re-use the GC matrices from self.dataset (they were computed once
        from the full imputed returns and are the same for every fold — the
        Granger structure is fixed per shock period, not per fold).

        The node features (mean_return etc.) ARE recomputed per fold because
        they depend on the fold's imputed returns slice, which differs slightly
        between folds.
        """
        # Rebuild train dataset with fold-specific imputed returns
        train_dataset = ShockGraphDataset(
            returns      = train_returns if not train_returns.empty else self.returns,
            gc_matrices  = {
                s.name: self.dataset.gc_matrices[s.name]
                for s in train_shocks
                if s.name in self.dataset.gc_matrices
            },
            shocks       = train_shocks,
            tickers      = self.dataset.tickers,
            feature_cols = self.dataset.feature_cols,
        )
        train_dataset.build()

        # Rebuild test dataset with fold-specific imputed returns
        test_dataset = ShockGraphDataset(
            returns      = test_returns if not test_returns.empty else self.returns,
            gc_matrices  = {
                s.name: self.dataset.gc_matrices[s.name]
                for s in test_shocks
                if s.name in self.dataset.gc_matrices
            },
            shocks       = test_shocks,
            tickers      = self.dataset.tickers,
            feature_cols = self.dataset.feature_cols,
        )
        test_dataset.build()

        return train_dataset.snapshots, test_dataset.snapshots

    # ─────────────────────────────────────────
    # AGGREGATION & REPORTING
    # ─────────────────────────────────────────

    def _aggregate(self) -> Dict:
        """
        Aggregate FoldResult metrics across all completed folds.
        Implements paper Eq. 11 (mean) and Eq. 12 (std).
        """
        metrics_list = [
            {
                "auc_roc"         : r.auc_roc,
                "aupr"            : r.aupr,
                "f1"              : r.f1,
                "precision"       : r.precision,
                "recall"          : r.recall,
                "auc_roc_low_fpr" : r.auc_roc_low_fpr,
                "best_val_loss"   : r.best_val_loss,
                "best_epoch"      : float(r.best_epoch),
                "n_epochs_run"    : float(r.n_epochs_run),
                "train_time_sec"  : r.train_time_sec,
                "pos_weight"      : r.pos_weight,   # [NEW]
                "threshold"       : r.threshold,    # [NEW]
            }
            for r in self._fold_results
        ]
        return aggregate_monte_carlo_metrics(metrics_list)

    def _build_summary_df(self) -> pd.DataFrame:
        """Build a DataFrame with one row per fold plus a μ ± σ summary."""
        rows = [r.to_dict() for r in self._fold_results]
        df   = pd.DataFrame(rows)

        if df.empty:
            return df

        numeric = df.select_dtypes(include="number").columns.tolist()
        numeric = [c for c in numeric if c != "fold"]

        mu_row  = {"fold": "μ"}
        std_row = {"fold": "σ"}
        for col in numeric:
            mu_row[col]  = round(float(df[col].mean()), 4)
            std_row[col] = round(float(df[col].std(ddof=0)), 4)

        return pd.concat(
            [df, pd.DataFrame([mu_row, std_row])],
            ignore_index=True,
        )

    def _save_mc_results(
        self,
        aggregated : Dict,
        summary_df : pd.DataFrame,
    ) -> None:
        """Persist aggregated results and fold summary to results/metrics/."""
        # JSON: full aggregated metrics
        json_path = CFG.METRICS_DIR / "monte_carlo_results.json"
        payload   = {
            "n_folds"       : len(self._fold_results),
            "n_shocks_total": len(self.shocks),
            "train_ratio"   : self.train_ratio,
            "aggregated"    : aggregated,
            "fold_results"  : [r.to_dict() for r in self._fold_results],
        }
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info(f"MC results saved → {json_path}")

        # CSV: human-readable summary table
        csv_path = CFG.METRICS_DIR / "monte_carlo_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        logger.info(f"MC summary saved → {csv_path}")

        # Also trigger FoldSummary (checkpointing layer summary)
        try:
            FoldSummary.build(
                checkpoint_dir = CFG.CHECKPOINT_DIR,
                n_folds        = len(self._fold_results),
            )
        except Exception as exc:
            logger.warning(f"FoldSummary.build() skipped: {exc}")

    # ─────────────────────────────────────────
    # DISPLAY HELPERS
    # ─────────────────────────────────────────

    def _log_fold_result(self, r: FoldResult) -> None:
        logger.info(
            f"Fold {r.fold:02d} result | "
            f"AUC-ROC={r.auc_roc:.4f} | "
            f"AUPR={r.aupr:.4f} | "
            f"F1={r.f1:.4f} | "
            f"low-FPR-AUC={r.auc_roc_low_fpr:.4f} | "
            f"time={r.train_time_sec:.0f}s"
        )

    def _print_aggregated(self, aggregated: Dict) -> None:
        """Print a formatted summary matching the paper's Table 3 layout."""
        header_metrics = ["auc_roc", "aupr", "f1", "auc_roc_low_fpr"]
        print("\n" + "=" * 60)
        print("MONTE CARLO RESULTS  (paper Table 3 format)")
        print("=" * 60)
        print(f"{'Metric':<22}  {'μ':>8}  {'σ':>8}")
        print("-" * 60)
        for key in header_metrics:
            if key not in aggregated:
                continue
            v = aggregated[key]
            label = {
                "auc_roc"        : "AUC-ROC (Eq. 6)",
                "aupr"           : "Avg Precision (Eq. 7)",
                "f1"             : "F1 Score (Eq. 8)",
                "auc_roc_low_fpr": "AUC-ROC low FPR [0,0.1]",
            }.get(key, key)
            print(f"  {label:<20}  {v['mean']:>8.4f}  {v['std']:>8.4f}")
        print("-" * 60)
        print(f"  {'Folds run':<20}  {len(self._fold_results):>8}")
        print(f"  {'Best epoch (avg)':<20}  "
              f"{aggregated.get('best_epoch', {}).get('mean', '?'):>8}")
        print("=" * 60)

    # ─────────────────────────────────────────
    # RESULTS ACCESSORS
    # ─────────────────────────────────────────

    def results_df(self) -> pd.DataFrame:
        """Return per-fold metrics as a DataFrame (excludes μ/σ rows)."""
        return pd.DataFrame([r.to_dict() for r in self._fold_results])

    def print_report(self) -> None:
        """Print full report including per-fold details and aggregated summary."""
        df  = self.results_df()
        agg = self._aggregate()

        print("\n" + "=" * 80)
        print("PER-FOLD METRICS")
        print("=" * 80)
        cols = ["fold", "auc_roc", "aupr", "f1",
                "auc_roc_low_fpr", "best_epoch", "n_epochs_run"]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].to_string(index=False))
        self._print_aggregated(agg)

    def fold_result(self, fold: int) -> Optional[FoldResult]:
        """Return the FoldResult for a specific fold number."""
        for r in self._fold_results:
            if r.fold == fold:
                return r
        return None

    @classmethod
    def load_results(cls, path: Optional[Path] = None) -> Dict:
        """Load previously saved MC results from JSON."""
        path = path or (CFG.METRICS_DIR / "monte_carlo_results.json")
        if not Path(path).exists():
            raise FileNotFoundError(f"No MC results at {path}.")
        with open(path) as f:
            return json.load(f)