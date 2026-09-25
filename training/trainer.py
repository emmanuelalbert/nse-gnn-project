"""
training/trainer.py
--------------------
Single-fold training loop for the TGAT shock-propagation model.

Paper reference: Section III-B (Table 1), Section III-D

    "To ensure our model was not overfitting or underfitting, we employed
     multiple robust techniques during the training process:
       - Dropout (rate 0.3)
       - LayerNorm
       - L2 regularisation (weight decay 1e-5)
       - Residual connections after each pair of TGCN layers
       - Early stopping monitored validation loss
       - Negative sampling balanced positive and negative link instances"

Training protocol:
    1. For each epoch:
         a. Iterate over training snapshots (one shock period × T time steps).
         b. For each snapshot, sample balanced positive + negative edges.
         c. Run TGAT forward pass → logits → BCE loss → backward.
         d. Clip gradients (max_norm=1.0) to prevent exploding gradients.
         e. Scheduler step.
    2. After each epoch, run validation:
         a. Evaluate full N×N adjacency prediction on all validation snapshots.
         b. Compute AUC-ROC, AUPR, F1 using compute_metrics(), with the
            decision threshold re-tuned each epoch via find_best_threshold()
            rather than a fixed 0.5 cutoff. [UPDATED — see class docstring.]
    3. Early stopping: if val_loss hasn't improved by TRAIN_EARLY_STOP_DELTA
       for TRAIN_EARLY_STOP_PATIENCE consecutive epochs, halt training.
    4. Save best checkpoint (by CFG.CHECKPOINT_MONITOR, default AUC-ROC —
       see settings.py) via CheckpointManager.

[UPDATED] Class-imbalance handling
-----------------------------------
Two changes vs. the original version, both aimed at the precision=0.164/
recall=1.0 issue observed in the real fold-1 run (history.json):

  1. pos_weight is no longer hardcoded to 1.0. It is computed once at
     Trainer construction from the TRAINING split's actual target-adjacency
     class balance (compute_pos_weight), so the loss penalises missed
     positive links roughly in proportion to how rare they actually are.
     This can be overridden explicitly via the `pos_weight` constructor arg
     if a fixed project-wide value is preferred instead.

  2. Validation-time metrics no longer use compute_metrics()'s default
     threshold=0.5. Each epoch, find_best_threshold() sweeps a fine grid on
     the validation predictions themselves to find the F1-maximising cutoff,
     and metrics are reported at that threshold. This does NOT touch the
     held-out TEST set threshold — MonteCarloRunner is responsible for
     re-deriving its own threshold from validation-like data before scoring
     the final test set (see monte_carlo.py).

The Trainer is designed to be called by MonteCarloRunner (monte_carlo.py)
once per fold, but can also be used standalone for a single full-dataset run.

Usage:
    from training.trainer import Trainer
    from model import build_tgat

    model   = build_tgat(in_features=1)
    trainer = Trainer(model, train_data, val_data, fold=1)
    result  = trainer.run()
    # result: {best_val_loss, best_epoch, metrics, history_df, best_threshold}
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as CFG
from graph.dataset import ShockGraphDataset
from model.loss import TGATLoss, compute_metrics, compute_pos_weight, find_best_threshold
from model.tgat import TGAT
from training.checkpointing import CheckpointManager
from training.scheduler import WarmUpScheduler, build_scheduler

logger = logging.getLogger(__name__)

# Gradient clipping max norm (prevents exploding gradients in early epochs)
GRAD_CLIP_NORM = 1.0

# Default decision threshold used ONLY as a fallback before the first
# validation pass has produced enough data to tune one (or if tuning
# degenerates, e.g. a validation batch with a single class present).
FALLBACK_THRESHOLD = 0.5


class EarlyStopping:
    """
    Monitors validation loss and signals when training should stop.

    Paper Table 1:
        "Early stopping monitored validation loss to prevent overfitting."

    Improvement is measured as: current_loss < best_loss - delta.
    Training stops when no improvement has been seen for `patience` epochs.

    Args:
        patience : Number of epochs to wait after last improvement.
        delta    : Minimum change to qualify as an improvement.
        verbose  : Log messages when patience counter increases.
    """

    def __init__(
        self,
        patience : int   = CFG.TRAIN_EARLY_STOP_PATIENCE,
        delta    : float = CFG.TRAIN_EARLY_STOP_DELTA,
        verbose  : bool  = True,
    ):
        self.patience = patience
        self.delta    = delta
        self.verbose  = verbose

        self._best_loss   : float = float("inf")
        self._counter     : int   = 0
        self._should_stop : bool  = False

    def __call__(self, val_loss: float) -> bool:
        """
        Update state and return True if training should stop.

        Args:
            val_loss : Current epoch's validation loss.

        Returns:
            True if early stopping criterion is met.
        """
        if val_loss < self._best_loss - self.delta:
            self._best_loss = val_loss
            self._counter   = 0
        else:
            self._counter += 1
            if self.verbose:
                logger.debug(
                    f"  EarlyStopping: no improvement for {self._counter}/{self.patience} epochs "
                    f"(best={self._best_loss:.6f})"
                )
            if self._counter >= self.patience:
                self._should_stop = True

        return self._should_stop

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    @property
    def best_loss(self) -> float:
        return self._best_loss

    @property
    def counter(self) -> int:
        return self._counter

    def reset(self) -> None:
        self._best_loss   = float("inf")
        self._counter     = 0
        self._should_stop = False


class Trainer:
    """
    Single-fold TGAT training loop.

    Args:
        model        : Initialised TGAT model.
        train_data   : List of PyG Data objects (training snapshots).
        val_data     : List of PyG Data objects (validation snapshots).
        fold         : Current fold index (1-based, for checkpoint naming).
        epochs       : Maximum training epochs.
        lr           : Initial learning rate.
        scheduler_strategy : LR scheduler type ("plateau" | "cosine" | "constant").
        warmup_epochs: Warm-up epochs before main LR schedule.
        neg_ratio    : Negative sampling ratio (negatives per positive edge).
        device       : Compute device.
        seed         : Random seed for negative sampling reproducibility.
        verbose      : Log progress every `verbose` epochs (0 = silent).
        pos_weight   : [UPDATED] Positive-class BCE weight. If None (default),
                       computed automatically from train_data's target
                       adjacency class balance via compute_pos_weight().
                       Pass an explicit float to override.
        threshold_metric : [UPDATED] Metric to maximise when tuning the
                       validation decision threshold each epoch. Default "f1".
    """

    def __init__(
        self,
        model               : TGAT,
        train_data          : List,
        val_data            : List,
        fold                : int   = 1,
        epochs              : int   = CFG.TRAIN_EPOCHS,
        lr                  : float = CFG.TRAIN_LR,
        weight_decay        : float = CFG.TRAIN_L2_WEIGHT_DECAY,
        scheduler_strategy  : str   = "plateau",
        warmup_epochs       : int   = 10,
        neg_ratio           : float = CFG.NEGATIVE_SAMPLING_RATIO,
        device              : str   = CFG.DEVICE,
        seed                : int   = CFG.RANDOM_SEED,
        verbose             : int   = 10,
        pos_weight          : Optional[float] = None,
        threshold_metric    : str   = "f1",
    ):
        self.model     = model.to(device)
        self.train_data = train_data
        self.val_data   = val_data
        self.fold       = fold
        self.epochs     = epochs
        self.neg_ratio  = neg_ratio
        self.device     = device
        self.seed       = seed
        self.verbose    = verbose
        self.threshold_metric = threshold_metric

        # Optimizer: Adam with L2 weight decay (paper: 1e-5)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr           = lr,
            weight_decay = weight_decay,
        )

        # Scheduler
        self.scheduler = build_scheduler(
            self.optimizer,
            strategy      = scheduler_strategy,
            warmup_epochs = warmup_epochs,
        )

        # ── [UPDATED] Class-weighted loss ────────────────────────────────
        # pos_weight is derived from the TRAINING split's true target-
        # adjacency class balance (not the balanced 1:1 negative-sampled
        # training pairs), because from_adjacency() -- used in _val_epoch --
        # evaluates the full, imbalanced adjacency matrix. If the caller
        # passed an explicit pos_weight, that takes precedence.
        if pos_weight is None:
            pos_weight = self._compute_train_pos_weight(train_data)
            logger.info(
                f"Fold {fold:02d} | auto pos_weight = {pos_weight:.4f} "
                f"(derived from training adjacency class balance)"
            )
        self.pos_weight = pos_weight

        self.criterion = TGATLoss(
            pos_weight   = self.pos_weight,
            label_smooth = 0.05,
        ).to(device)

        # Early stopping
        self.early_stopper = EarlyStopping(
            patience = CFG.TRAIN_EARLY_STOP_PATIENCE,
            delta    = CFG.TRAIN_EARLY_STOP_DELTA,
        )

        # Checkpoint manager for this fold
        self.ckpt = CheckpointManager(
            fold    = fold,
            monitor = CFG.CHECKPOINT_MONITOR,
            mode    = CFG.CHECKPOINT_MODE,
        )

        # Internal state
        self._train_losses : List[float] = []
        self._val_losses   : List[float] = []
        self._start_time   : float = 0.0

        # [UPDATED] Tracks the threshold tuned at the best-checkpoint epoch,
        # so downstream evaluation (MonteCarloRunner) can reuse it on the
        # held-out test set instead of re-deriving from scratch or falling
        # back to 0.5.
        self._best_threshold : float = FALLBACK_THRESHOLD

    # ─────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────

    def run(self) -> Dict:
        """
        Execute the full training loop for this fold.

        Returns:
            result dict with keys:
                fold, best_epoch, best_val_loss, metrics, history_df,
                train_time_sec, n_epochs_run, best_threshold, pos_weight
        """
        self._start_time = time.time()
        torch.manual_seed(self.seed + self.fold)
        np.random.seed(self.seed + self.fold)

        logger.info(
            f"{'='*60}\n"
            f"FOLD {self.fold:02d} | "
            f"train={len(self.train_data)} snapshots | "
            f"val={len(self.val_data)} snapshots | "
            f"max_epochs={self.epochs} | pos_weight={self.pos_weight:.4f}"
        )

        for epoch in range(1, self.epochs + 1):
            # ── Training step ────────────────────────────────────────────
            train_loss = self._train_epoch(epoch)

            # ── Validation step ──────────────────────────────────────────
            val_loss, val_metrics, val_threshold = self._val_epoch()

            # ── Scheduler step ───────────────────────────────────────────
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step(metric=val_loss)

            # ── Logging ──────────────────────────────────────────────────
            self.ckpt.log_epoch(epoch, train_loss, val_loss, val_metrics, lr=current_lr)
            self._train_losses.append(train_loss)
            self._val_losses.append(val_loss)

            if self.verbose > 0 and epoch % self.verbose == 0:
                self._log_progress(epoch, train_loss, val_loss, val_metrics, current_lr)

            # ── Checkpoint ───────────────────────────────────────────────
            is_new_best = self.ckpt.save_best(
                self.model, self.optimizer, self.scheduler,
                epoch, val_loss, val_metrics
            )
            self.ckpt.save_last(
                self.model, self.optimizer, self.scheduler,
                epoch, val_loss
            )

            # [UPDATED] Remember the threshold that was tuned at whichever
            # epoch the checkpoint manager considers "best" (per
            # CFG.CHECKPOINT_MONITOR — default AUC-ROC), not just the last
            # epoch run, so early-stopping/patience epochs after the true
            # best don't silently overwrite it with a worse threshold.
            if is_new_best:
                self._best_threshold = val_threshold

            # ── Early stopping ───────────────────────────────────────────
            if self.early_stopper(val_loss):
                logger.info(
                    f"Fold {self.fold:02d} | Early stopping at epoch {epoch} "
                    f"(patience={CFG.TRAIN_EARLY_STOP_PATIENCE})"
                )
                break

        # ── Finalize ─────────────────────────────────────────────────────
        hist_path    = self.ckpt.finalize()
        elapsed      = time.time() - self._start_time

        logger.info(
            f"Fold {self.fold:02d} DONE | "
            f"best_epoch={self.ckpt.best_epoch()} | "
            f"best_val_loss={self.ckpt.best_val_loss():.6f} | "
            f"best_threshold={self._best_threshold:.4f} | "
            f"time={elapsed:.1f}s"
        )

        return {
            "fold"           : self.fold,
            "best_epoch"     : self.ckpt.best_epoch(),
            "best_val_loss"  : self.ckpt.best_val_loss(),
            "metrics"        : self.ckpt.best_metrics(),
            "history_df"     : self.ckpt.history_df(),
            "train_time_sec" : elapsed,
            "n_epochs_run"   : len(self._train_losses),
            "best_threshold" : self._best_threshold,   # [NEW]
            "pos_weight"     : self.pos_weight,          # [NEW]
        }

    # ─────────────────────────────────────────
    # [NEW] CLASS-WEIGHT COMPUTATION
    # ─────────────────────────────────────────

    def _compute_train_pos_weight(self, train_data: List) -> float:
        """
        Derive pos_weight from the training split's off-diagonal target
        adjacency class balance, pooled across all training snapshots.

        Falls back to 1.0 (no reweighting) if train_data is empty or
        target adjacency is unavailable, so this never hard-fails a run.
        """
        y_all = []
        for snap in train_data:
            y = getattr(snap, "y", None)
            if y is None:
                continue
            y_np = y.cpu().numpy() if hasattr(y, "cpu") else np.asarray(y)
            N    = y_np.shape[0]
            mask = ~np.eye(N, dtype=bool)
            y_all.append(y_np[mask].flatten())

        if not y_all:
            logger.warning(
                f"Fold {self.fold:02d} | could not derive pos_weight from "
                f"train_data (no target adjacency found) — defaulting to 1.0"
            )
            return 1.0

        y_concat = np.concatenate(y_all)
        return compute_pos_weight(y_concat)

    # ─────────────────────────────────────────
    # TRAINING EPOCH
    # ─────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        """
        One full pass over all training snapshots.

        For each snapshot:
          1. Build temporal sequence: all snapshots from the same shock period
             up to and including this one  → (t, N, F) tensor.
          2. Sample balanced positive + negative edges.
          3. Forward pass → logits → BCE loss.
          4. Backward + gradient clip + optimizer step.

        Returns:
            Mean training loss across all snapshots.
        """
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        # Group snapshots by shock name for temporal sequence building
        shock_groups = self._group_by_shock(self.train_data)

        for shock_name, snaps in shock_groups.items():
            for t_idx, snap in enumerate(snaps):
                snap = self._to_device(snap)

                # Build temporal sequence [snap_0, ..., snap_t]
                x_seq, time_deltas = self._build_sequence(snaps[: t_idx + 1])

                # Sample positive and negative edges
                pos_edges, neg_edges = self._sample_edges(snap, epoch + t_idx)

                if pos_edges.shape[1] == 0 or neg_edges.shape[1] == 0:
                    continue   # edge case: skip degenerate samples

                # Forward
                self.optimizer.zero_grad()
                pos_logits, neg_logits, _ = self.model(
                    x_seq       = x_seq,
                    edge_index  = snap.edge_index,
                    time_deltas = time_deltas,
                    edge_weight = getattr(snap, "edge_weight", None),
                    pos_edges   = pos_edges,
                    neg_edges   = neg_edges,
                )

                loss = self.criterion(pos_logits, neg_logits)

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
                self.optimizer.step()

                total_loss += loss.item()
                n_batches  += 1

        return total_loss / max(1, n_batches)

    # ─────────────────────────────────────────
    # VALIDATION EPOCH
    # ─────────────────────────────────────────

    @torch.no_grad()
    def _val_epoch(self) -> Tuple[float, Dict, float]:
        """
        Full validation pass.

        For each validation snapshot:
          - Predict full N×N adjacency matrix.
          - Flatten (excluding diagonal) and compare to ground-truth GC matrix.

        [UPDATED] After collecting predictions across all validation
        snapshots, the decision threshold is re-tuned via
        find_best_threshold() (maximising self.threshold_metric, default
        F1) instead of using compute_metrics()'s 0.5 default. This directly
        targets the precision=0.164/recall=1.0 symptom, which was caused by
        the decoder's outputs being compressed into a narrow band where a
        fixed 0.5 cutoff classifies almost everything as positive.

        Returns:
            (val_loss, metrics_dict, tuned_threshold)
            val_loss is computed as BCE on the full adjacency prediction
            (using the fold's pos_weight — see class docstring).
        """
        self.model.eval()
        all_y_true   : List[np.ndarray] = []
        all_y_scores : List[np.ndarray] = []
        total_loss   = 0.0
        n_batches    = 0

        shock_groups = self._group_by_shock(self.val_data)

        for shock_name, snaps in shock_groups.items():
            for t_idx, snap in enumerate(snaps):
                snap = self._to_device(snap)
                x_seq, time_deltas = self._build_sequence(snaps[: t_idx + 1])

                # Predict full adjacency
                logits = self.model.predict_adjacency(
                    x_seq, snap.edge_index, time_deltas,
                    getattr(snap, "edge_weight", None),
                    apply_sigmoid=False,   # raw logits for BCE loss
                )   # (N, N)

                # Ground truth adjacency
                y = snap.y.to(self.device)    # (N, N) binary GC matrix

                # Validation loss on full adjacency (no negative sampling)
                loss = self.criterion.from_adjacency(logits, y, mask_diagonal=True)
                total_loss += loss.item()
                n_batches  += 1

                # Collect predictions for metric computation (exclude diagonal)
                N    = logits.shape[0]
                mask = ~torch.eye(N, dtype=torch.bool, device=self.device)
                y_true_flat   = y[mask].cpu().numpy().flatten()
                y_score_flat  = torch.sigmoid(logits)[mask].cpu().numpy().flatten()

                all_y_true.append(y_true_flat)
                all_y_scores.append(y_score_flat)

        val_loss = total_loss / max(1, n_batches)

        # Aggregate across all val snapshots, tune threshold, compute metrics
        if all_y_true:
            y_true_all   = np.concatenate(all_y_true)
            y_scores_all = np.concatenate(all_y_scores)

            # [NEW] Tune threshold on this epoch's validation predictions.
            tuned_threshold, _ = find_best_threshold(
                y_true_all, y_scores_all, metric=self.threshold_metric
            )
            metrics = compute_metrics(
                y_true_all, y_scores_all, threshold=tuned_threshold
            )
        else:
            metrics = {}
            tuned_threshold = FALLBACK_THRESHOLD

        return val_loss, metrics, tuned_threshold

    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────

    def _group_by_shock(self, data: List) -> Dict[str, List]:
        """
        Group a flat list of snapshots by their shock_name attribute.
        Returns an ordered dict: {shock_name: [snap_t0, snap_t1, ...]}.
        Preserves chronological ordering within each group.
        """
        groups: Dict[str, List] = {}
        for snap in data:
            name = getattr(snap, "shock_name", "unknown")
            groups.setdefault(name, []).append(snap)
        return groups

    def _build_sequence(self, snaps: List) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build the temporal input sequence from a list of snapshots.

        Stacks node feature matrices into (T, N, F) and extracts time deltas.
        The last element of the sequence is the "current" step being predicted.

        Args:
            snaps : Chronologically ordered list of snapshots for one shock period,
                    up to and including the current time step t.

        Returns:
            x_seq       : FloatTensor (T, N, F).
            time_deltas : FloatTensor (T,).
        """
        x_list  = [s.x.to(self.device) for s in snaps]          # list of (N, F)
        x_seq   = torch.stack(x_list, dim=0)                     # (T, N, F)

        # Time delta for each step: take the mean of the per-edge time deltas
        # (all edges in a snapshot have the same normalised time position)
        dt_list = [s.time_delta[:1].to(self.device) for s in snaps]
        time_deltas = torch.cat(dt_list, dim=0)                  # (T,)

        return x_seq, time_deltas

    def _sample_edges(
        self,
        snap : object,
        seed : int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample balanced positive and negative edges from a snapshot.
        Returns (pos_edges, neg_edges), each shape (2, k).
        """
        torch.manual_seed(seed)
        N = snap.num_nodes
        y = snap.y.cpu().numpy()

        # Positive edges: non-diagonal entries where GC = 1
        eye_mask          = ~np.eye(N, dtype=bool)
        pos_rows, pos_cols = np.where((y > 0) & eye_mask)
        neg_rows, neg_cols = np.where((y == 0) & eye_mask)

        n_pos = len(pos_rows)
        n_neg = min(len(neg_rows), int(n_pos * self.neg_ratio))

        if n_pos == 0:
            # Degenerate case: no edges — return empty tensors
            empty = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            return empty, empty

        # Random subsample of negatives
        perm     = torch.randperm(len(neg_rows))[:n_neg]
        neg_rows = neg_rows[perm.numpy()]
        neg_cols = neg_cols[perm.numpy()]

        pos_edges = torch.tensor(
            np.array([pos_rows, pos_cols]), dtype=torch.long, device=self.device
        )
        neg_edges = torch.tensor(
            np.array([neg_rows, neg_cols]), dtype=torch.long, device=self.device
        )
        return pos_edges, neg_edges

    def _to_device(self, snap: object) -> object:
        """Move all tensor attributes of a PyG Data snapshot to self.device."""
        from torch_geometric.data import Data
        if isinstance(snap, Data):
            return snap.to(self.device)
        return snap

    def _log_progress(
        self,
        epoch      : int,
        train_loss : float,
        val_loss   : float,
        metrics    : Dict,
        lr         : float,
    ) -> None:
        m_str = ""
        if metrics:
            m_str = (
                f" | AUC={metrics.get('auc_roc','?'):.4f}"
                f" AUPR={metrics.get('aupr','?'):.4f}"
                f" F1={metrics.get('f1','?'):.4f}"
                f" P={metrics.get('precision','?'):.4f}"
                f" R={metrics.get('recall','?'):.4f}"
                f" thr={metrics.get('threshold','?'):.4f}"
            )
        logger.info(
            f"Fold {self.fold:02d} | Ep {epoch:4d}/{self.epochs} | "
            f"train={train_loss:.4f} val={val_loss:.4f} | "
            f"lr={lr:.2e}{m_str} | "
            f"ES={self.early_stopper.counter}/{CFG.TRAIN_EARLY_STOP_PATIENCE}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level="INFO", format=CFG.LOG_FORMAT)
    import pandas as pd
    import tempfile, shutil

    torch.manual_seed(42)
    np.random.seed(42)

    # ── Mini synthetic dataset ──
    N, F = 20, 1
    dates   = pd.date_range("2008-10-01", periods=40, freq="B")
    tickers = [f"T{i}" for i in range(N - 1)] + ["^NSEI"]
    ret     = np.random.normal(0, 0.012, (40, N))
    returns = pd.DataFrame(ret, index=dates, columns=tickers)

    from shock_detection.detector import ShockPeriod
    from graph.dataset import ShockGraphDataset

    shocks = [
        ShockPeriod("Shock_A", "2008-10-01", "2008-10-15", 11, -0.03, -0.08, 4.0),
        ShockPeriod("Shock_B", "2008-10-16", "2008-10-31", 12, -0.02, -0.06, 3.2),
        ShockPeriod("Shock_C", "2008-11-01", "2008-11-14", 10, -0.02, -0.05, 2.8),
    ]

    gc_mats = {}
    for s in shocks:
        # Make an imbalanced (sparse) GC matrix so pos_weight auto-compute
        # has something non-trivial to react to in this quick test.
        m = (np.random.rand(N, N) > 0.82).astype(np.float32)
        np.fill_diagonal(m, 1.0)
        gc_mats[s.name] = m

    dataset = ShockGraphDataset(returns, gc_mats, shocks, tickers, ["mean_return"])
    dataset.build()

    # 2 train shocks, 1 val shock
    train_data, val_data = dataset.split(train_ratio=0.67)
    print(f"Train snapshots: {len(train_data)}, Val snapshots: {len(val_data)}")

    # ── Train for a few epochs ──
    from model import build_tgat
    model = build_tgat(in_features=F)

    tmp = Path(tempfile.mkdtemp())
    import graph.adjacency as adj_mod
    # Temporarily redirect checkpoint dir to tmp
    orig_ckpt = CFG.CHECKPOINT_DIR
    CFG.CHECKPOINT_DIR = tmp

    try:
        trainer = Trainer(
            model             = model,
            train_data        = train_data,
            val_data          = val_data,
            fold              = 1,
            epochs            = 5,
            scheduler_strategy = "constant",
            warmup_epochs      = 2,
            verbose            = 1,
        )
        print(f"Auto-computed pos_weight: {trainer.pos_weight:.4f}")
        result = trainer.run()

        print(f"\nTraining result:")
        print(f"  best_epoch     : {result['best_epoch']}")
        print(f"  best_val_loss  : {result['best_val_loss']:.4f}")
        print(f"  n_epochs_run   : {result['n_epochs_run']}")
        print(f"  train_time     : {result['train_time_sec']:.1f}s")
        print(f"  pos_weight     : {result['pos_weight']:.4f}")
        print(f"  best_threshold : {result['best_threshold']:.4f}")
        if result["metrics"]:
            print(f"  metrics        : {result['metrics']}")

        df = result["history_df"]
        print(f"\nHistory ({len(df)} epochs):\n{df[['epoch','train_loss','val_loss']].to_string()}")
        print("\nAll trainer tests PASSED.")
    finally:
        CFG.CHECKPOINT_DIR = orig_ckpt
        shutil.rmtree(tmp)