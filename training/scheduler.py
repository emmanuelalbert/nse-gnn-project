"""
training/scheduler.py
---------------------
Learning-rate scheduler configuration for TGAT training.

Paper reference: Table 1
    "We used ... early stopping monitored validation loss to prevent
     overfitting."

The paper does not specify a scheduler explicitly — the assumption is a
fixed LR with early stopping acting as the convergence criterion. We expose
a flexible scheduler factory that covers four strategies:

    "constant"    — Fixed LR (paper default). Early stopping is the only
                    convergence mechanism. Simple and effective for small
                    graphs like the Nifty 50 (50 nodes).

    "cosine"      — CosineAnnealingLR. LR decays smoothly from lr to
                    eta_min over T_max epochs, then restarts. Good for
                    longer training runs where the loss plateau is reached
                    slowly.

    "plateau"     — ReduceLROnPlateau. Halves LR when validation loss
                    hasn't improved for `patience` epochs. Adaptive and
                    safe — the paper's early stopping complements this well
                    (stop if LR gets too small and val loss still stalls).

    "onecycle"    — OneCycleLR. Implements the 1-cycle policy (Smith &
                    Toler, 2019): LR rises to max_lr then decays with
                    cosine annealing, combined with momentum cycling.
                    Best for fast convergence when total_steps is known.

Linear warm-up:
    All strategies support an optional linear warm-up phase (default:
    10 epochs). During warm-up, LR scales linearly from lr/100 → lr.
    This prevents the first few batches from causing large weight updates
    before the model has seen enough data, which is especially important
    for the TGAT where the temporal attention weights are randomly initialised.

Usage:
    from training.scheduler import build_scheduler, WarmUpScheduler

    scheduler = build_scheduler(optimizer, strategy="plateau")
    scheduler = build_scheduler(optimizer, strategy="cosine", T_max=200)

    # Manual step:
    scheduler.step(val_loss)         # for "plateau"
    scheduler.step()                 # for all others
"""

import logging
import math
from typing import Any, Dict, Optional

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings as CFG

logger = logging.getLogger(__name__)

# Strategies that require scheduler.step(metric) instead of scheduler.step()
METRIC_SCHEDULERS = {"plateau"}


# ─────────────────────────────────────────────────────────────────────────────
# LINEAR WARM-UP WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class WarmUpScheduler:
    """
    Wraps any base scheduler with a linear warm-up phase.

    During the first `warmup_epochs` epochs, the LR scales linearly from
    `start_factor * base_lr` to `base_lr`. After warm-up, control is handed
    to the base scheduler.

    This is implemented as a thin wrapper rather than a chained scheduler
    so that the `step(metric)` interface of ReduceLROnPlateau is preserved
    cleanly.

    Args:
        optimizer     : The model optimizer.
        base_scheduler: Post-warm-up scheduler (from build_scheduler).
        warmup_epochs : Number of warm-up epochs.
        start_factor  : Initial LR multiplier (LR starts at base_lr × start_factor).
    """

    def __init__(
        self,
        optimizer      : optim.Optimizer,
        base_scheduler : Any,                  # any torch LR scheduler
        warmup_epochs  : int   = 10,
        start_factor   : float = 0.01,
        is_metric_based: bool  = False,
    ):
        self.optimizer       = optimizer
        self.base_scheduler  = base_scheduler
        self.warmup_epochs   = warmup_epochs
        self.start_factor    = start_factor
        self.is_metric_based = is_metric_based
        self._epoch          = 0

        # Record base LRs before warm-up modifies them
        self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._set_warmup_lr(epoch=0)

    def step(self, metric: Optional[float] = None) -> None:
        """
        Advance one epoch. Applies warm-up scaling or delegates to the
        base scheduler depending on the current epoch count.
        """
        self._epoch += 1

        if self._epoch <= self.warmup_epochs:
            self._set_warmup_lr(self._epoch)
        else:
            if self.is_metric_based and metric is not None:
                self.base_scheduler.step(metric)
            else:
                self.base_scheduler.step()

    def _set_warmup_lr(self, epoch: int) -> None:
        """Linearly interpolate LR from start_factor → 1.0 over warmup_epochs."""
        if self.warmup_epochs == 0:
            return
        progress = epoch / max(1, self.warmup_epochs)
        factor   = self.start_factor + (1.0 - self.start_factor) * progress
        for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            pg["lr"] = base_lr * factor

    def get_last_lr(self) -> list:
        """Return current LRs for all parameter groups."""
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def state_dict(self) -> Dict:
        return {
            "epoch"          : self._epoch,
            "base_lrs"       : self._base_lrs,
            "base_scheduler" : self.base_scheduler.state_dict(),
        }

    def load_state_dict(self, state: Dict) -> None:
        self._epoch    = state["epoch"]
        self._base_lrs = state["base_lrs"]
        self.base_scheduler.load_state_dict(state["base_scheduler"])


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(
    optimizer       : optim.Optimizer,
    strategy        : str   = "plateau",
    warmup_epochs   : int   = 10,
    # "cosine" options
    T_max           : int   = CFG.TRAIN_EPOCHS,
    eta_min         : float = 1e-6,
    # "plateau" options
    patience        : int   = CFG.TRAIN_EARLY_STOP_PATIENCE // 2,
    plateau_factor  : float = 0.5,
    plateau_min_lr  : float = 1e-6,
    plateau_threshold: float = CFG.TRAIN_EARLY_STOP_DELTA,
    # "onecycle" options
    max_lr          : float = CFG.TRAIN_LR * 10,
    total_steps     : Optional[int] = None,
    # warm-up options
    warmup_start_factor: float = 0.01,
) -> WarmUpScheduler:
    """
    Build a learning-rate scheduler with optional linear warm-up.

    Args:
        optimizer    : The model optimizer (e.g. Adam from trainer.py).
        strategy     : "constant" | "cosine" | "plateau" | "onecycle".
        warmup_epochs: Linear warm-up epochs before the main schedule.
        T_max        : Epochs for cosine annealing period.
        eta_min      : Minimum LR for cosine annealing.
        patience     : Epochs without improvement before LR reduction (plateau).
        plateau_factor: LR multiplication factor on plateau (e.g. 0.5 = halve).
        plateau_min_lr: Floor LR for plateau scheduler.
        plateau_threshold: Minimum val-loss improvement to reset patience.
        max_lr       : Peak LR for OneCycleLR.
        total_steps  : Total training steps for OneCycleLR (required).
        warmup_start_factor: Initial LR fraction at epoch 0 during warm-up.

    Returns:
        WarmUpScheduler wrapping the base scheduler.
    """
    strategy       = strategy.lower()
    is_metric      = strategy in METRIC_SCHEDULERS

    if strategy == "constant":
        base = sched.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        logger.info(
            f"Scheduler: constant LR (warm-up={warmup_epochs} epochs)"
        )

    elif strategy == "cosine":
        base = sched.CosineAnnealingLR(
            optimizer, T_max=T_max - warmup_epochs, eta_min=eta_min
        )
        logger.info(
            f"Scheduler: CosineAnnealing T_max={T_max} eta_min={eta_min} "
            f"(warm-up={warmup_epochs})"
        )

    elif strategy == "plateau":
        base = sched.ReduceLROnPlateau(
            optimizer,
            mode      = "min",          # monitor validation loss
            factor    = plateau_factor,
            patience  = patience,
            min_lr    = plateau_min_lr,
            threshold = plateau_threshold,
            verbose   = False,
        )
        logger.info(
            f"Scheduler: ReduceLROnPlateau factor={plateau_factor} "
            f"patience={patience} min_lr={plateau_min_lr} "
            f"(warm-up={warmup_epochs})"
        )

    elif strategy == "onecycle":
        if total_steps is None:
            raise ValueError(
                "total_steps is required for OneCycleLR. "
                "Pass total_steps = n_epochs × steps_per_epoch."
            )
        base = sched.OneCycleLR(
            optimizer,
            max_lr        = max_lr,
            total_steps   = total_steps,
            pct_start     = warmup_epochs / max(1, total_steps),
            anneal_strategy = "cos",
            div_factor    = 1.0 / warmup_start_factor,
            final_div_factor = max_lr / max(1e-8, eta_min),
        )
        # OneCycleLR handles warm-up internally — disable external warm-up
        warmup_epochs = 0
        is_metric     = False
        logger.info(
            f"Scheduler: OneCycleLR max_lr={max_lr} total_steps={total_steps}"
        )

    else:
        raise ValueError(
            f"Unknown scheduler strategy '{strategy}'. "
            "Choose from: constant, cosine, plateau, onecycle."
        )

    return WarmUpScheduler(
        optimizer        = optimizer,
        base_scheduler   = base,
        warmup_epochs    = warmup_epochs,
        start_factor     = warmup_start_factor,
        is_metric_based  = is_metric,
    )


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level="DEBUG", format=CFG.LOG_FORMAT)
    import torch.nn as nn

    model    = nn.Linear(10, 1)
    base_lr  = 1e-3

    for strategy in ["constant", "cosine", "plateau"]:
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
        scheduler = build_scheduler(optimizer, strategy=strategy, warmup_epochs=5)

        lrs = []
        for epoch in range(1, 31):
            val_loss = 1.0 / epoch   # simulated decreasing loss
            scheduler.step(metric=val_loss)
            lrs.append(scheduler.get_last_lr()[0])

        warmup_done = lrs[4]    # epoch 5 should be near base_lr
        print(f"\n{strategy:12s} | epoch1={lrs[0]:.2e}  epoch5={warmup_done:.2e}  "
              f"epoch10={lrs[9]:.2e}  epoch30={lrs[29]:.2e}")
        # Warm-up check: lr at epoch 5 should be close to base_lr
        assert abs(warmup_done - base_lr) < 1e-5, \
            f"Warm-up didn't reach base_lr: {warmup_done} vs {base_lr}"

    # OneCycleLR
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    scheduler = build_scheduler(optimizer, strategy="onecycle",
                                total_steps=100, max_lr=base_lr * 5)
    for _ in range(50):
        scheduler.step()
    print(f"\nonecycle    | mid-step lr={scheduler.get_last_lr()[0]:.2e}")

    # State dict round-trip
    sd = scheduler.state_dict()
    scheduler.load_state_dict(sd)
    print("state_dict  | save/load OK")

    print("\nAll scheduler tests PASSED.")