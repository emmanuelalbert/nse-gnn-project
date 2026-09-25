"""
training/checkpointing.py
--------------------------
Versioned checkpoint management for TGAT model state across Monte Carlo folds.

Paper reference: Section III-D
    "We conducted Monte Carlo experiments with 10 chronological splits of
     our dataset into training and testing sets."

Checkpoint responsibilities:
  1. Save the best model state for each MC fold, per CFG.CHECKPOINT_MONITOR
     / CFG.CHECKPOINT_MODE (default: maximise AUC-ROC — see settings.py).
  2. Save a "last" checkpoint at every epoch for crash recovery.
  3. Load the best checkpoint for a fold at inference / evaluation time.
  4. Track and log training history (loss curves, metric curves) per fold.
  5. Produce a final summary CSV of best metrics across all folds.

[PATCHED] Selection criterion generalised (was: val_loss only)
------------------------------------------------------------------
Previously, save_best() had a single hardcoded rule:

    if val_loss >= self._best_val_loss:
        return False

This ignored CFG.CHECKPOINT_MONITOR / CFG.CHECKPOINT_MODE entirely, even
though:
  (a) settings.py sets CHECKPOINT_MONITOR="auc_roc", CHECKPOINT_MODE="max"
      by default, with a code comment explaining exactly why (val_loss and
      AUC-ROC diverge on this task — the lowest-val_loss epoch is often
      the LEAST useful one, essentially pre-training).
  (b) trainer.py already calls CheckpointManager(fold=..., monitor=...,
      mode=...) — this constructor didn't even accept those kwargs, which
      would raise a TypeError as soon as Trainer.run() instantiated it.
  (c) A real training run's history.json showed best_epoch=3 selected by
      max AUC-ROC (0.7493 at epoch 3, vs. 0.3928 at epoch 1), while the
      top-level "best_val_loss" field in that same JSON was stuck at
      epoch 1's value (0.581479) — a stale artifact of exactly this
      val_loss-only logic never being generalised, even though some other
      selection mechanism (now understood to be what THIS patch restores)
      was clearly driving best_epoch elsewhere in the pipeline.

This version:
  - Accepts `monitor` and `mode` in __init__ (default from CFG, matching
    what trainer.py already passes).
  - Generalises save_best()'s comparison to work off ANY metric named in
    `monitor` (found in `metrics` dict, or "val_loss" itself as a special
    case since it isn't part of the metrics dict).
  - Tracks self._best_val_loss as "the val_loss AT the epoch that was
    actually selected as best" (not the global minimum val_loss ever
    seen across all epochs) — this directly fixes the staleness bug in
    (c) above. If you want the old global-minimum-val_loss behaviour for
    comparison, it's no longer tracked separately; recompute it from
    history_df()["val_loss"].min() if needed.

Directory layout:
    results/checkpoints/
        fold_01/
            best.pt          ← best checkpoint per CFG.CHECKPOINT_MONITOR
            last.pt          ← latest epoch checkpoint (crash recovery)
            history.json     ← epoch-level loss and metric logs
        fold_02/
            ...
        summary.csv          ← one row per fold, best metrics

Checkpoint .pt format:
    {
        "epoch"         : int,
        "fold"          : int,
        "val_loss"      : float,
        "monitor"       : str,        ← [NEW] which metric decided "best"
        "monitor_value" : float,      ← [NEW] its value at this checkpoint
        "model_state"   : OrderedDict,
        "optim_state"   : dict,
        "scheduler_state": dict,
        "config"        : dict,       ← model hyperparameters
        "metrics"       : dict,       ← val metrics at this checkpoint
    }

Usage:
    from training.checkpointing import CheckpointManager
    ckpt = CheckpointManager(fold=1, monitor="auc_roc", mode="max")
    ckpt.save_best(model, optimizer, scheduler, epoch, val_loss, metrics)
    ckpt.save_last(model, optimizer, scheduler, epoch, val_loss)
    model = ckpt.load_best(model)
    ckpt.log_epoch(epoch, train_loss, val_loss, metrics)
    ckpt.finalize()            # write history.json
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings as CFG

logger = logging.getLogger(__name__)

# Sentinel values for "no best seen yet", direction-aware.
_WORST_FOR_MODE = {
    "min": float("inf"),
    "max": float("-inf"),
}


class CheckpointManager:
    """
    Manages model checkpoints for a single Monte Carlo fold.

    One CheckpointManager instance is created per fold in monte_carlo.py.

    Args:
        fold       : 1-based fold index (1–10).
        base_dir   : Root checkpoint directory. Defaults to CFG.CHECKPOINT_DIR.
        keep_last  : If True, maintain a rolling "last.pt" for crash recovery.
        monitor    : [NEW] Name of the metric that decides "best". Either
                     "val_loss" (special-cased — always available) or any
                     key present in the `metrics` dict passed to save_best()
                     (e.g. "auc_roc", "f1", "aupr"). Defaults to
                     CFG.CHECKPOINT_MONITOR.
        mode       : [NEW] "min" or "max" — whether lower or higher values
                     of `monitor` are better. Defaults to CFG.CHECKPOINT_MODE.
    """

    def __init__(
        self,
        fold      : int,
        base_dir  : Path = CFG.CHECKPOINT_DIR,
        keep_last : bool = True,
        monitor   : str  = CFG.CHECKPOINT_MONITOR,
        mode      : str  = CFG.CHECKPOINT_MODE,
    ):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

        self.fold       = fold
        self.keep_last  = keep_last
        self.monitor    = monitor
        self.mode       = mode
        self.fold_dir   = Path(base_dir) / f"fold_{fold:02d}"
        self.fold_dir.mkdir(parents=True, exist_ok=True)

        self.best_path  = self.fold_dir / "best.pt"
        self.last_path  = self.fold_dir / "last.pt"
        self.hist_path  = self.fold_dir / "history.json"

        # [PATCHED] best_val_loss now means "val_loss recorded AT the
        # epoch that was selected as best by `monitor`/`mode`" — not the
        # global minimum val_loss seen across all epochs. This is the
        # semantic fix for the staleness bug described in the module
        # docstring.
        self._best_val_loss    : float = _WORST_FOR_MODE["min"]  # updated once a best is found
        self._best_monitor_val : float = _WORST_FOR_MODE[mode]
        self._best_epoch       : int   = -1
        self._best_metrics     : Dict  = {}
        self._history          : List[Dict] = []

        logger.info(
            f"CheckpointManager | fold={fold} | dir={self.fold_dir} | "
            f"monitor={monitor} ({mode})"
        )

    # ─────────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────────

    def _extract_monitor_value(
        self,
        val_loss : float,
        metrics  : Optional[Dict],
    ) -> Optional[float]:
        """
        Pull the value of self.monitor out of (val_loss, metrics).

        "val_loss" is handled specially since it's a positional arg, not
        part of the metrics dict. Anything else is looked up in `metrics`.
        Returns None if the monitored key isn't available this epoch
        (e.g. metrics={} on a degenerate validation pass) — callers should
        treat None as "cannot evaluate this epoch as a candidate best".
        """
        if self.monitor == "val_loss":
            return val_loss
        if metrics and self.monitor in metrics:
            return metrics[self.monitor]
        return None

    def _is_improvement(self, candidate: float, current_best: float) -> bool:
        """Direction-aware comparison per self.mode."""
        if self.mode == "min":
            return candidate < current_best
        return candidate > current_best

    def save_best(
        self,
        model     : torch.nn.Module,
        optimizer : torch.optim.Optimizer,
        scheduler : Any,
        epoch     : int,
        val_loss  : float,
        metrics   : Optional[Dict] = None,
    ) -> bool:
        """
        Save checkpoint if self.monitor improved (per self.mode).
        Returns True if saved.

        [PATCHED] Previously this only ever compared val_loss and ignored
        self.monitor/self.mode entirely. Now it compares whichever metric
        monitor names — by default "auc_roc"/"max" per settings.py — and
        falls back to skipping the epoch (returns False, logs a warning)
        if that metric isn't present yet (e.g. metrics={} very early on).

        Args:
            model     : The TGAT model.
            optimizer : Current optimizer.
            scheduler : Current scheduler (WarmUpScheduler or base scheduler).
            epoch     : Current epoch number (1-based).
            val_loss  : Validation loss for this epoch (always recorded,
                        regardless of whether it's the monitored metric).
            metrics   : Optional dict of evaluation metrics (AUC, F1, etc.).

        Returns:
            True if this is a new best and the checkpoint was saved.
        """
        candidate = self._extract_monitor_value(val_loss, metrics)

        if candidate is None:
            logger.warning(
                f"  Fold {self.fold:02d} | Epoch {epoch:4d} | "
                f"monitor='{self.monitor}' not found in metrics "
                f"({list((metrics or {}).keys())}) — skipping as best-checkpoint candidate."
            )
            return False

        if not self._is_improvement(candidate, self._best_monitor_val):
            return False

        self._best_monitor_val = candidate
        self._best_val_loss    = val_loss   # val_loss AT this (selected) epoch
        self._best_epoch       = epoch
        self._best_metrics     = metrics or {}

        payload = self._build_payload(
            model, optimizer, scheduler, epoch, val_loss, metrics, candidate
        )
        torch.save(payload, self.best_path)

        logger.info(
            f"  ✓ Fold {self.fold:02d} | Epoch {epoch:4d} | "
            f"{self.monitor}={candidate:.6f} (val_loss={val_loss:.6f}) → best.pt saved"
        )
        return True

    def save_last(
        self,
        model     : torch.nn.Module,
        optimizer : torch.optim.Optimizer,
        scheduler : Any,
        epoch     : int,
        val_loss  : float,
        metrics   : Optional[Dict] = None,
    ) -> None:
        """
        Overwrite last.pt with the current epoch's state.
        Used for crash recovery — always reflects the most recent epoch.
        """
        if not self.keep_last:
            return
        payload = self._build_payload(model, optimizer, scheduler, epoch, val_loss, metrics)
        torch.save(payload, self.last_path)

    def _build_payload(
        self,
        model         : torch.nn.Module,
        optimizer     : torch.optim.Optimizer,
        scheduler     : Any,
        epoch         : int,
        val_loss      : float,
        metrics       : Optional[Dict],
        monitor_value : Optional[float] = None,
    ) -> Dict:
        """Build the checkpoint payload dict."""
        sched_state = {}
        if hasattr(scheduler, "state_dict"):
            try:
                sched_state = scheduler.state_dict()
            except Exception:
                pass

        # Extract model config if available
        model_config = {}
        if hasattr(model, "in_features"):
            model_config = {
                "in_features"    : model.in_features,
                "hidden_dim"     : model.hidden_dim,
                "num_layers"     : model.num_layers,
                "num_heads"      : model.num_heads,
                "time_dim"       : model.time_dim,
                "decoder_hidden" : model.decoder.mlp[0].out_features,
                "dropout"        : model.dropout_rate,
                "use_layer_norm" : model.use_layer_norm,
                "residual"       : model.residual,
            }

        return {
            "epoch"           : epoch,
            "fold"            : self.fold,
            "val_loss"        : val_loss,
            "monitor"         : self.monitor,          # [NEW]
            "monitor_value"   : monitor_value,          # [NEW]
            "model_state"     : model.state_dict(),
            "optim_state"     : optimizer.state_dict(),
            "scheduler_state" : sched_state,
            "config"          : model_config,
            "metrics"         : metrics or {},
            "saved_at"        : datetime.now().isoformat(timespec="seconds"),
        }

    # ─────────────────────────────────────────
    # LOAD
    # ─────────────────────────────────────────

    def load_best(
        self,
        model  : torch.nn.Module,
        device : str = CFG.DEVICE,
    ) -> torch.nn.Module:
        """
        Load the best-checkpoint weights into model.

        Args:
            model  : An already-instantiated TGAT model.
            device : Device to load onto ("cpu" | "cuda" | "mps").

        Returns:
            model with best weights loaded.
        """
        if not self.best_path.exists():
            raise FileNotFoundError(
                f"No best checkpoint for fold {self.fold} at {self.best_path}. "
                "Run training first."
            )
        ckpt = torch.load(self.best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        monitor_str = (
            f" | {ckpt.get('monitor', 'val_loss')}={ckpt.get('monitor_value', ckpt['val_loss']):.6f}"
            if ckpt.get("monitor_value") is not None else ""
        )
        logger.info(
            f"Loaded best checkpoint | fold={self.fold} | "
            f"epoch={ckpt['epoch']} | val_loss={ckpt['val_loss']:.6f}{monitor_str}"
        )
        return model

    def load_last(
        self,
        model     : torch.nn.Module,
        optimizer : Optional[torch.optim.Optimizer] = None,
        device    : str = CFG.DEVICE,
    ) -> Dict:
        """
        Load the last checkpoint for crash recovery.
        Restores model weights + optimizer state.

        Returns:
            Payload dict (contains epoch, val_loss, metrics).
        """
        if not self.last_path.exists():
            raise FileNotFoundError(
                f"No last checkpoint for fold {self.fold} at {self.last_path}."
            )
        ckpt = torch.load(self.last_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        if optimizer is not None:
            optimizer.load_state_dict(ckpt["optim_state"])
        logger.info(
            f"Loaded last checkpoint | fold={self.fold} | "
            f"epoch={ckpt['epoch']} | val_loss={ckpt['val_loss']:.6f}"
        )
        return ckpt

    def has_best(self) -> bool:
        return self.best_path.exists()

    def best_val_loss(self) -> float:
        """
        val_loss recorded AT the selected best epoch (per self.monitor).
        [PATCHED] No longer the global minimum val_loss across all epochs —
        see module docstring for why that was the staleness bug.
        """
        return self._best_val_loss

    def best_monitor_value(self) -> float:
        """[NEW] The monitored metric's value at the selected best epoch."""
        return self._best_monitor_val

    def best_epoch(self) -> int:
        return self._best_epoch

    def best_metrics(self) -> Dict:
        return self._best_metrics

    # ─────────────────────────────────────────
    # HISTORY LOGGING
    # ─────────────────────────────────────────

    def log_epoch(
        self,
        epoch      : int,
        train_loss : float,
        val_loss   : float,
        metrics    : Optional[Dict] = None,
        lr         : Optional[float] = None,
    ) -> None:
        """
        Record per-epoch training statistics.
        Called once per epoch in trainer.py.
        """
        entry = {
            "epoch"      : epoch,
            "train_loss" : round(train_loss, 6),
            "val_loss"   : round(val_loss,   6),
            "lr"         : lr,
            "is_best"    : epoch == self._best_epoch,
        }
        if metrics:
            entry.update({f"val_{k}": v for k, v in metrics.items()})

        self._history.append(entry)

    def finalize(self) -> Path:
        """
        Write history.json to disk.
        Called at the end of fold training.
        """
        with open(self.hist_path, "w") as f:
            json.dump({
                "fold"           : self.fold,
                "best_epoch"     : self._best_epoch,
                "best_val_loss"  : self._best_val_loss,     # now epoch-consistent, see docstring
                "monitor"        : self.monitor,             # [NEW]
                "best_monitor_value" : self._best_monitor_val,  # [NEW]
                "best_metrics"   : self._best_metrics,
                "history"        : self._history,
            }, f, indent=2, default=str)

        logger.info(
            f"Fold {self.fold:02d} finalized | "
            f"best_epoch={self._best_epoch} | "
            f"{self.monitor}={self._best_monitor_val:.6f} | "
            f"best_val_loss={self._best_val_loss:.6f} | "
            f"history → {self.hist_path}"
        )
        return self.hist_path

    def history_df(self) -> pd.DataFrame:
        """Return training history as a DataFrame (for plotting)."""
        return pd.DataFrame(self._history)

    def load_history(self) -> Dict:
        """Load history.json from disk."""
        if not self.hist_path.exists():
            return {}
        with open(self.hist_path) as f:
            return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SUMMARY  (across all folds)
# ─────────────────────────────────────────────────────────────────────────────

class FoldSummary:
    """
    Aggregates best metrics from all folds into a summary CSV.
    Used by monte_carlo.py after all 10 folds complete.

    Output: results/checkpoints/summary.csv
    """

    SUMMARY_PATH = CFG.CHECKPOINT_DIR / "summary.csv"

    @classmethod
    def build(
        cls,
        checkpoint_dir : Path = CFG.CHECKPOINT_DIR,
        n_folds        : int  = CFG.MONTE_CARLO_N_SPLITS,
    ) -> pd.DataFrame:
        """
        Scan all fold_XX/history.json files and produce a summary.

        Returns:
            DataFrame with one row per fold + a final μ ± σ row.
        """
        rows = []
        for fold in range(1, n_folds + 1):
            hist_path = checkpoint_dir / f"fold_{fold:02d}" / "history.json"
            if not hist_path.exists():
                logger.warning(f"Missing history for fold {fold}: {hist_path}")
                continue
            with open(hist_path) as f:
                hist = json.load(f)
            row = {
                "fold"          : fold,
                "best_epoch"    : hist.get("best_epoch", -1),
                "best_val_loss" : hist.get("best_val_loss", float("nan")),
                "monitor"       : hist.get("monitor", "val_loss"),          # [NEW]
                "monitor_value" : hist.get("best_monitor_value", float("nan")),  # [NEW]
            }
            row.update(hist.get("best_metrics", {}))
            rows.append(row)

        if not rows:
            logger.warning("No fold histories found — summary is empty.")
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Append μ ± σ summary rows
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "fold"]

        mu_row  = {"fold": "μ"}
        sig_row = {"fold": "σ"}
        for col in numeric_cols:
            mu_row[col]  = round(float(df[col].mean(skipna=True)), 4)
            sig_row[col] = round(float(df[col].std(skipna=True, ddof=0)), 4)

        summary = pd.concat(
            [df, pd.DataFrame([mu_row, sig_row])],
            ignore_index=True,
        )

        # Save
        cls.SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(cls.SUMMARY_PATH, index=False)
        logger.info(f"Fold summary saved → {cls.SUMMARY_PATH}")
        return summary

    @classmethod
    def load(cls) -> pd.DataFrame:
        if not cls.SUMMARY_PATH.exists():
            raise FileNotFoundError(f"No summary found at {cls.SUMMARY_PATH}.")
        return pd.read_csv(cls.SUMMARY_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, shutil
    logging.basicConfig(level="INFO", format=CFG.LOG_FORMAT)

    tmp = Path(tempfile.mkdtemp())
    try:
        from model import build_tgat
        import torch.optim as optim

        model     = build_tgat(in_features=1)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # [UPDATED TEST] monitor="auc_roc"/"max" — mirrors settings.py default
        # and the real history.json pattern: val_loss goes UP after epoch 1
        # (0.58 -> higher) while AUC-ROC keeps improving through epoch 5.
        # A correct patch should select epoch 5 as best (highest AUC-ROC),
        # NOT epoch 1 (lowest val_loss) — the exact divergence documented
        # in settings.py's CHECKPOINT_MONITOR comment.
        ckpt = CheckpointManager(fold=1, base_dir=tmp, monitor="auc_roc", mode="max")

        val_losses = [0.58, 0.74, 0.72, 0.71, 0.73]     # lowest at epoch 1
        auc_rocs   = [0.39, 0.63, 0.70, 0.72, 0.75]     # highest at epoch 5
        for epoch, (vloss, auc) in enumerate(zip(val_losses, auc_rocs), 1):
            metrics = {"auc_roc": auc, "f1": 0.2 + epoch * 0.02}
            saved   = ckpt.save_best(model, optimizer, None, epoch, vloss, metrics)
            ckpt.save_last(model, optimizer, None, epoch, vloss)
            ckpt.log_epoch(epoch, vloss + 0.05, vloss, metrics, lr=1e-3)
            print(f"  Epoch {epoch}: val_loss={vloss:.2f} auc_roc={auc:.2f} saved_best={saved}")

        # Core assertion: best epoch should be 5 (max AUC-ROC), NOT 1 (min val_loss)
        assert ckpt.best_epoch() == 5, f"Expected best_epoch=5 (max AUC-ROC), got {ckpt.best_epoch()}"
        assert ckpt.best_monitor_value() == 0.75
        # best_val_loss should now be epoch 5's val_loss (0.73), not the
        # global minimum (0.58 from epoch 1) — this is the staleness fix.
        assert ckpt.best_val_loss() == 0.73, (
            f"Expected best_val_loss=0.73 (epoch-5-consistent), got {ckpt.best_val_loss()} "
            f"— if this is 0.58, the staleness bug is back."
        )
        print(f"\n  PASS: best_epoch={ckpt.best_epoch()} (max AUC-ROC, not min val_loss)")
        print(f"  PASS: best_val_loss={ckpt.best_val_loss()} (epoch-consistent, not stale global min)")

        # Load best
        model2 = build_tgat(in_features=1)
        model2 = ckpt.load_best(model2)

        # History
        hist_path = ckpt.finalize()
        df        = ckpt.history_df()
        assert len(df) == 5, f"Expected 5 history rows, got {len(df)}"
        print(f"\nHistory (5 epochs):\n{df[['epoch','train_loss','val_loss','is_best']]}")

        # Sanity-check the raw history.json content directly
        with open(hist_path) as f:
            raw = json.load(f)
        assert raw["best_epoch"] == 5
        assert raw["monitor"] == "auc_roc"
        assert raw["best_monitor_value"] == 0.75
        assert raw["best_val_loss"] == 0.73
        print(f"\n  PASS: history.json fields internally consistent "
              f"(best_epoch={raw['best_epoch']}, monitor={raw['monitor']}, "
              f"best_monitor_value={raw['best_monitor_value']}, "
              f"best_val_loss={raw['best_val_loss']})")

        # ── Backward-compat check: monitor="val_loss"/"min" still works ──
        ckpt2 = CheckpointManager(fold=2, base_dir=tmp, monitor="val_loss", mode="min")
        for epoch, vloss in enumerate([0.9, 0.8, 0.78, 0.82, 0.76], 1):
            ckpt2.save_best(model, optimizer, None, epoch, vloss, {"auc_roc": 0.5})
        assert ckpt2.best_epoch() == 5   # lowest val_loss (0.76) is at epoch 5 here
        assert ckpt2.best_val_loss() == 0.76
        print(f"\n  PASS: legacy val_loss/min mode still works "
              f"(best_epoch={ckpt2.best_epoch()}, best_val_loss={ckpt2.best_val_loss()})")

        # ── Constructor kwarg compatibility check ──
        # This exact call signature is what trainer.py actually uses —
        # confirms the original TypeError-on-instantiation bug is fixed.
        ckpt3 = CheckpointManager(fold=3, monitor=CFG.CHECKPOINT_MONITOR, mode=CFG.CHECKPOINT_MODE)
        print(f"\n  PASS: CheckpointManager(fold=3, monitor=CFG.CHECKPOINT_MONITOR, "
              f"mode=CFG.CHECKPOINT_MODE) — no TypeError "
              f"(monitor={ckpt3.monitor}, mode={ckpt3.mode})")

        # FoldSummary
        # Write a second fold history manually
        fold2_dir = tmp / "fold_02_manual"
        fold2_dir.mkdir()

        summary = FoldSummary.build(checkpoint_dir=tmp, n_folds=2)
        FoldSummary.SUMMARY_PATH = tmp / "summary.csv"
        print(f"\nFold summary:\n{summary}")

        print("\nAll checkpointing tests PASSED.")
    finally:
        shutil.rmtree(tmp)