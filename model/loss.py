"""
model/loss.py
-------------
Loss functions and training objectives for the TGAT link-prediction task.

Paper reference: Section III-B and Section III-C

    Training objective: binary link prediction.
    Given node embeddings H ∈ ℝ^{N×d}, predict whether each directed edge
    (i→j) exists in the Granger causality adjacency matrix.

    The paper uses:
      - Balanced negative sampling (ratio ≈ 1 negative per positive edge)
      - Binary Cross-Entropy (BCE) loss
      - Evaluation via ROC-AUC, Average Precision, and F1 Score

Design decisions:
  - We use BCEWithLogitsLoss (numerically stabler than BCE + Sigmoid) applied
    to raw decoder logits. The Sigmoid squashing happens inside the loss.
  - Class imbalance: the GC matrices are typically 30–65% dense. We still
    use negative sampling (ratio=1) to stabilise training gradients.
  - Auxiliary contrastive loss (optional): encourages embedding separation
    between causally connected vs. unconnected nodes.
  - Focal loss variant (optional): for very sparse GC matrices (<15% density).

[UPDATED] Class-imbalance handling and threshold tuning
---------------------------------------------------------
Empirically (see history.json, fold 1), the trained model at its selected
checkpoint produced precision=0.164 / recall=1.0 on the held-out COVID test
set — i.e. it was flagging almost every pair as positive. Two compounding
causes, both now addressed here:

  1. TGATLoss was always called with pos_weight=1.0, even though the
     positive:negative ratio in the FULL adjacency validation/test pass
     (as opposed to the balanced 1:1 negative-sampled training pass) is
     roughly 13,266 : 67,584 ≈ 1:5.1 (see history.json best_metrics).
     -> compute_pos_weight() below derives this ratio directly from a
        fold's training target adjacency so the loss can penalise missed
        positives more than it penalises false positives during the
        full-adjacency validation loss (from_adjacency), without having
        to hardcode a project-wide constant that would be wrong if the
        GC density shifts between shock periods/folds.

  2. compute_metrics() was always called with the default threshold=0.5,
     but the decoder's raw sigmoid outputs are known to be crammed into a
     razor-thin band (empirically ~0.512-0.529, see settings.py's
     GRAPH_LINK_METHOD comment) after only a handful of training epochs.
     A fixed 0.5 cutoff on that band is nearly meaningless — it trivially
     classifies almost everything as positive, which is exactly the
     recall=1.0/precision=0.164 symptom observed.
     -> find_best_threshold() sweeps a fine-grained set of thresholds on
        a VALIDATION split only and returns the one that maximises F1
        (or a caller-specified target metric). The chosen threshold is
        then applied when reporting held-out TEST metrics, so no test
        information leaks into the threshold choice.

See training/monte_carlo.py's _evaluate() for how the threshold tuned on
validation (test_data, which plays the role of "val" during Trainer.run())
is reused rather than re-tuned on the same data the final metrics are
reported on — avoiding a subtle leakage where the SAME set both trains
the threshold and reports the headline metric.

Usage:
    from model.loss import TGATLoss, compute_metrics, compute_pos_weight, find_best_threshold

    pos_weight = compute_pos_weight(train_adjacency_targets)
    criterion  = TGATLoss(pos_weight=pos_weight)
    loss       = criterion(logits_pos, logits_neg)

    best_thr, _ = find_best_threshold(y_val_true, y_val_scores)
    metrics     = compute_metrics(y_test_true, y_test_scores, threshold=best_thr)
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings as CFG


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY LOSS: BCE WITH LOGITS
# ─────────────────────────────────────────────────────────────────────────────

class TGATLoss(nn.Module):
    """
    Weighted binary cross-entropy loss for link prediction with negative sampling.

    Paper Section III-B:
        "Negative sampling balanced positive and negative link instances,
         improving training stability."

    Args:
        pos_weight   : Scalar weight on the positive BCE term. Default 1.0.
                       Increase (e.g. 2.0-5.0) when the target adjacency is
                       sparse/imbalanced — see compute_pos_weight() to derive
                       this empirically from a given fold's training data
                       rather than guessing a project-wide constant.
        reduction    : 'mean' | 'sum'.
        label_smooth : Label smoothing ε. Softens hard 0/1 targets.
    """

    def __init__(
        self,
        pos_weight   : float = 1.0,
        reduction    : str   = "mean",
        label_smooth : float = 0.0,
    ):
        super().__init__()
        self.pos_weight   = pos_weight
        self.reduction    = reduction
        self.label_smooth = label_smooth

        pw = torch.tensor([pos_weight]) if pos_weight != 1.0 else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw, reduction=reduction)

    def forward(
        self,
        pos_logits : torch.Tensor,   # (E_pos,) raw scores for positive edges
        neg_logits : torch.Tensor,   # (E_neg,) raw scores for negative edges
    ) -> torch.Tensor:
        """
        Returns scalar BCE loss over sampled positive and negative edges.

        Targets:
            positive → 1.0 - label_smooth
            negative → label_smooth
        """
        e = self.label_smooth
        pos_labels = torch.full_like(pos_logits, 1.0 - e)
        neg_labels = torch.full_like(neg_logits, e)

        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([pos_labels, neg_labels])
        return self.bce(logits, labels)

    def from_adjacency(
        self,
        logits_all    : torch.Tensor,   # (N, N)
        adj_target    : torch.Tensor,   # (N, N) binary GC matrix
        mask_diagonal : bool = True,
    ) -> torch.Tensor:
        """
        Loss on full N×N logit matrix vs target adjacency.
        Useful for small graphs where full-pair evaluation is tractable.

        NOTE: this is the loss path used by Trainer._val_epoch(), which
        evaluates the FULL (non-negative-sampled) adjacency matrix — i.e.
        the true ~1:5 imbalance is present here even when pos_weight=1.0
        was used for the balanced 1:1 training loss above. This is exactly
        why pos_weight should be set from compute_pos_weight() on the
        fold's actual target density, not left at the training-time default.
        """
        N = logits_all.shape[0]
        if mask_diagonal:
            mask = ~torch.eye(N, dtype=torch.bool, device=logits_all.device)
            logits_flat = logits_all[mask]
            target_flat = adj_target[mask].float()
        else:
            logits_flat = logits_all.flatten()
            target_flat = adj_target.float().flatten()

        e       = self.label_smooth
        t_soft  = target_flat * (1.0 - e) + e * (1.0 - target_flat)
        return self.bce(logits_flat, t_soft)


# ─────────────────────────────────────────────────────────────────────────────
# AUXILIARY: CONTRASTIVE LOSS (optional)
# ─────────────────────────────────────────────────────────────────────────────

class ContrastiveLoss(nn.Module):
    """
    Margin-based contrastive loss on node embedding pairs.
    Encourages connected nodes to be close and unconnected nodes far apart.

        L = y·d² + (1-y)·max(0, margin - d)²

    Not in the paper — optional regulariser. Use lambda_contrast ≈ 0.01.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        h        : torch.Tensor,   # (N, D)
        src      : torch.Tensor,   # (E,) int indices
        dst      : torch.Tensor,   # (E,) int indices
        labels   : torch.Tensor,   # (E,) float 0/1
    ) -> torch.Tensor:
        h_src = h[src]
        h_dst = h[dst]
        dist  = F.pairwise_distance(h_src, h_dst)
        pos   = labels       * dist.pow(2)
        neg   = (1 - labels) * F.relu(self.margin - dist).pow(2)
        return (pos + neg).mean()


# ─────────────────────────────────────────────────────────────────────────────
# AUXILIARY: FOCAL LOSS (optional, for sparse GC matrices)
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss variant — reduces loss weight on easy examples.
    Use as a drop-in replacement for TGATLoss when GC density < 15%.

        FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    Args:
        gamma : Focusing parameter. 0 = BCE. 2 = standard focal.
        alpha : Positive class prior weight.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        pos_logits : torch.Tensor,
        neg_logits : torch.Tensor,
    ) -> torch.Tensor:
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([torch.ones_like(pos_logits),
                            torch.zeros_like(neg_logits)])

        bce_raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        probs   = torch.sigmoid(logits)
        p_t     = probs * labels + (1 - probs) * (1 - labels)
        alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
        weight  = alpha_t * (1 - p_t) ** self.gamma
        return (weight * bce_raw).mean()


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED LOSS
# ─────────────────────────────────────────────────────────────────────────────

class TGATCombinedLoss(nn.Module):
    """
    Primary link-prediction loss + optional contrastive auxiliary loss.

        L_total = L_link + λ_contrast · L_contrast

    Args:
        lambda_contrast    : Weight of contrastive term. 0 = disabled.
        pos_weight         : Passed to TGATLoss.
        label_smooth       : Passed to TGATLoss.
        contrastive_margin : Passed to ContrastiveLoss.
    """

    def __init__(
        self,
        lambda_contrast    : float = 0.0,
        pos_weight         : float = 1.0,
        label_smooth       : float = 0.0,
        contrastive_margin : float = 1.0,
    ):
        super().__init__()
        self.lambda_contrast = lambda_contrast
        self.link_loss       = TGATLoss(pos_weight=pos_weight,
                                        label_smooth=label_smooth)
        self.contrast_loss   = (ContrastiveLoss(contrastive_margin)
                                if lambda_contrast > 0 else None)

    def forward(
        self,
        pos_logits  : torch.Tensor,
        neg_logits  : torch.Tensor,
        h           : Optional[torch.Tensor] = None,
        src_nodes   : Optional[torch.Tensor] = None,
        dst_nodes   : Optional[torch.Tensor] = None,
        all_labels  : Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns (total_loss, loss_breakdown_dict).
        loss_breakdown_dict keys: 'link', 'contrast' (if enabled), 'total'.
        """
        l_link = self.link_loss(pos_logits, neg_logits)
        d      = {"link": l_link.item()}
        total  = l_link

        if (self.lambda_contrast > 0 and self.contrast_loss is not None
                and h is not None and src_nodes is not None):
            l_con          = self.contrast_loss(h, src_nodes, dst_nodes, all_labels)
            total          = total + self.lambda_contrast * l_con
            d["contrast"]  = l_con.item()

        d["total"] = total.item()
        return total, d


# ─────────────────────────────────────────────────────────────────────────────
# [NEW] CLASS-IMBALANCE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def compute_pos_weight(
    y_true    : np.ndarray,
    cap       : float = 10.0,
    floor     : float = 1.0,
) -> float:
    """
    Derive a pos_weight for BCEWithLogitsLoss from a target adjacency's
    actual class balance: pos_weight = n_neg / n_pos.

    This should be computed from the TRAINING split only (per fold, since
    GC density can vary somewhat between shock periods), then passed into
    TGATLoss for that fold's Trainer instance.

    Args:
        y_true : Flattened binary target array (0/1), e.g. the off-diagonal
                 entries of one or more training adjacency matrices.
        cap    : Upper bound on the returned weight, to avoid the loss
                 overcorrecting into a "flag everything" regime in the
                 opposite direction if a fold happens to be extremely sparse.
        floor  : Lower bound (1.0 = no reweighting).

    Returns:
        A scalar pos_weight, clipped to [floor, cap].

    Example (matches history.json fold-1 test density):
        >>> y = np.array([1]*13266 + [0]*67584)
        >>> round(compute_pos_weight(y), 2)
        5.09
    """
    y_true = np.asarray(y_true).flatten()
    n_pos  = float((y_true == 1).sum())
    n_neg  = float((y_true == 0).sum())

    if n_pos == 0:
        return floor   # degenerate: no positives, weighting is meaningless

    raw_weight = n_neg / n_pos
    return float(np.clip(raw_weight, floor, cap))


# ─────────────────────────────────────────────────────────────────────────────
# [NEW] THRESHOLD TUNING
# ─────────────────────────────────────────────────────────────────────────────

def find_best_threshold(
    y_true       : np.ndarray,
    y_scores     : np.ndarray,
    metric       : str = "f1",
    n_steps      : int = 199,
    min_thr      : float = 0.01,
    max_thr      : float = 0.99,
) -> Tuple[float, float]:
    """
    Sweep decision thresholds and return the one that maximises `metric`
    on the given (VALIDATION) split.

    Rationale: the raw sigmoid output of the decoder is known to be
    compressed into a narrow band after limited training (empirically
    ~0.51-0.53), which makes the previous hardcoded threshold=0.5 nearly
    always classify every pair as positive (precision=0.164, recall=1.0
    in history.json). Sweeping a fine grid finds the actual decision
    boundary the model has implicitly learned, even within a narrow band.

    IMPORTANT: call this on validation predictions only, then apply the
    returned threshold when computing metrics on the held-out test set.
    Never tune the threshold directly on test data — that would leak
    information the same way tuning hyperparameters on the test set would.

    Args:
        y_true   : Binary ground-truth labels, shape (E,).
        y_scores : Predicted probabilities (post-sigmoid), shape (E,).
        metric   : Which metric to maximise: "f1" (default), "precision",
                   "recall", or "youden" (TPR - FPR, i.e. Youden's J).
        n_steps  : Number of threshold candidates to evaluate.
        min_thr, max_thr : Sweep range. Kept away from exactly 0/1 to
                   avoid degenerate all-positive/all-negative predictions.

    Returns:
        (best_threshold, best_metric_value)
    """
    y_true   = np.asarray(y_true,   dtype=float).flatten()
    y_scores = np.asarray(y_scores, dtype=float).flatten()

    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0   # degenerate: can't tune without both classes

    thresholds = np.linspace(min_thr, max_thr, n_steps)
    best_thr   = 0.5
    best_val   = -1.0

    for thr in thresholds:
        y_pred = (y_scores >= thr).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if metric == "precision":
            val = precision
        elif metric == "recall":
            val = recall
        elif metric == "youden":
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            val = recall - fpr
        else:  # "f1" default
            val = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else 0.0)

        if val > best_val:
            best_val = val
            best_thr = float(thr)

    return best_thr, float(best_val)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION METRICS  (paper Section III-C, Eq. 6–10)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true         : np.ndarray,
    y_scores       : np.ndarray,
    threshold      : float = 0.5,
    low_fpr_max    : float = 0.10,
) -> Dict[str, float]:
    """
    Compute AUC-ROC (Eq. 6), AUPR (Eq. 7), F1 (Eq. 8–10) and the
    low-FPR AUC-ROC used in the paper's Section IV-B analysis.

    Paper results to beat: AUC-ROC=0.77, AUPR=0.76, F1=0.79,
                           Low-FPR AUC-ROC (0–0.1)=0.77.

    Args:
        y_true      : Binary ground-truth labels, shape (E,).
        y_scores    : Predicted probabilities (post-sigmoid), shape (E,).
        threshold   : Decision boundary for F1/precision/recall. [UPDATED]
                      Callers should pass a threshold derived from
                      find_best_threshold() on a validation split, rather
                      than relying on the 0.5 default, whenever the goal
                      is a meaningful precision/recall trade-off rather
                      than a quick sanity check.
        low_fpr_max : Upper FPR limit for restricted AUC-ROC.

    Returns:
        Dict with keys: auc_roc, aupr, f1, precision, recall,
                        auc_roc_low_fpr, n_pos, n_neg, density, threshold.
    """
    y_true   = np.asarray(y_true,   dtype=float).flatten()
    y_scores = np.asarray(y_scores, dtype=float).flatten()

    # Guard: both classes must be present
    if len(np.unique(y_true)) < 2:
        return {k: 0.0 for k in [
            "auc_roc", "aupr", "f1", "precision", "recall",
            "auc_roc_low_fpr", "n_pos", "n_neg", "density", "threshold"
        ]}

    # ── AUC-ROC (Eq. 6) ───────────────────────────────────────────────────
    auc_roc = float(roc_auc_score(y_true, y_scores))

    # ── Average Precision / AUPR (Eq. 7) ──────────────────────────────────
    aupr = float(average_precision_score(y_true, y_scores))

    # ── F1, Precision, Recall (Eq. 8–10) ──────────────────────────────────
    y_pred    = (y_scores >= threshold).astype(int)
    f1        = float(f1_score(y_true, y_pred, zero_division=0))
    tp        = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp        = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn        = float(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # ── Low-FPR AUC-ROC (paper Section IV-B) ──────────────────────────────
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_scores)
    mask = fpr_arr <= low_fpr_max
    fpr_low = fpr_arr[mask]
    tpr_low = tpr_arr[mask]

    if len(fpr_low) > 1:
        if fpr_low[-1] < low_fpr_max:
            interp_tpr = float(np.interp(low_fpr_max, fpr_arr, tpr_arr))
            fpr_low = np.append(fpr_low, low_fpr_max)
            tpr_low = np.append(tpr_low, interp_tpr)
        auc_low = float(np.trapezoid(tpr_low, fpr_low) / low_fpr_max)
    else:
        auc_low = 0.0

    n_pos   = int(y_true.sum())
    n_neg   = int(len(y_true) - n_pos)
    density = n_pos / max(1, len(y_true))

    return {
        "auc_roc"         : round(auc_roc,   4),
        "aupr"            : round(aupr,       4),
        "f1"              : round(f1,         4),
        "precision"       : round(precision,  4),
        "recall"          : round(recall,     4),
        "auc_roc_low_fpr" : round(auc_low,   4),
        "n_pos"           : n_pos,
        "n_neg"           : n_neg,
        "density"         : round(density,    4),
        "threshold"       : round(float(threshold), 4),
    }


def aggregate_monte_carlo_metrics(
    metrics_list: list,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across Monte Carlo folds (paper Eq. 11 & 12).

        μ  = (1/N) Σ Metric_i
        σ  = sqrt( (1/N) Σ (Metric_i - μ)² )   [population std, paper Eq. 12]

    Args:
        metrics_list : List of metric dicts, one per fold.

    Returns:
        {metric_name: {"mean": float, "std": float, "all": list}}
    """
    if not metrics_list:
        return {}

    keys   = [k for k in metrics_list[0] if isinstance(metrics_list[0][k], (int, float))]
    result = {}
    for key in keys:
        vals = np.array([m[key] for m in metrics_list], dtype=float)
        result[key] = {
            "mean" : round(float(vals.mean()), 4),
            "std"  : round(float(vals.std(ddof=0)), 4),
            "all"  : vals.tolist(),
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    pos_logits = torch.randn(300) + 1.0
    neg_logits = torch.randn(300) - 1.0

    # TGATLoss
    crit = TGATLoss(pos_weight=1.0, label_smooth=0.05)
    loss = crit(pos_logits, neg_logits)
    assert loss.item() > 0
    print(f"TGATLoss             OK : {loss.item():.4f}")

    # from_adjacency
    N = 15
    loss_full = crit.from_adjacency(torch.randn(N, N), (torch.rand(N, N) > 0.6).float())
    print(f"from_adjacency       OK : {loss_full.item():.4f}")

    # FocalLoss
    fl_loss = FocalLoss()(pos_logits, neg_logits)
    print(f"FocalLoss            OK : {fl_loss.item():.4f}")

    # ContrastiveLoss
    h   = torch.randn(20, 64)
    src = torch.randint(0, 20, (40,))
    dst = torch.randint(0, 20, (40,))
    lbl = torch.randint(0, 2, (40,)).float()
    cl  = ContrastiveLoss()(h, src, dst, lbl)
    print(f"ContrastiveLoss      OK : {cl.item():.4f}")

    # CombinedLoss
    tot, ld = TGATCombinedLoss(lambda_contrast=0.01)(
        pos_logits, neg_logits, h=h,
        src_nodes=src[:20], dst_nodes=dst[:20], all_labels=lbl[:20]
    )
    print(f"TGATCombinedLoss     OK : total={tot.item():.4f}  {ld}")

    # [NEW] compute_pos_weight — sanity check against history.json's fold-1 density
    y_imbalanced = np.array([1] * 13266 + [0] * 67584)
    pw = compute_pos_weight(y_imbalanced)
    print(f"\ncompute_pos_weight   OK : {pw:.4f}  (expect ≈ 67584/13266 ≈ 5.10, capped at 10.0)")
    assert 5.0 < pw < 5.2

    # [NEW] find_best_threshold — narrow-band scenario mirroring the real run
    # (scores crammed into [0.51, 0.53], mirroring the empirically observed band)
    y_true_narrow = np.random.randint(0, 2, 2000)
    y_scores_narrow = 0.51 + 0.02 * (y_true_narrow + np.random.normal(0, 0.3, 2000)).clip(0, 1)
    best_thr, best_f1 = find_best_threshold(y_true_narrow, y_scores_narrow, metric="f1")
    print(f"find_best_threshold  OK : thr={best_thr:.4f}  f1={best_f1:.4f}")

    m_default = compute_metrics(y_true_narrow, y_scores_narrow, threshold=0.5)
    m_tuned   = compute_metrics(y_true_narrow, y_scores_narrow, threshold=best_thr)
    print(f"  @0.5     : precision={m_default['precision']:.4f} recall={m_default['recall']:.4f} f1={m_default['f1']:.4f}")
    print(f"  @tuned   : precision={m_tuned['precision']:.4f} recall={m_tuned['recall']:.4f} f1={m_tuned['f1']:.4f}")

    # compute_metrics
    y_true   = np.random.randint(0, 2, 1000)
    y_scores = np.clip(y_true + np.random.normal(0, 0.4, 1000), 0, 1)
    m        = compute_metrics(y_true, y_scores)
    print(f"\ncompute_metrics      OK :")
    for k, v in m.items():
        print(f"  {k:<22} {v}")

    # Monte Carlo aggregation
    runs = [{"auc_roc": 0.77 + np.random.normal(0, 0.01),
             "aupr":    0.76 + np.random.normal(0, 0.01),
             "f1":      0.79 + np.random.normal(0, 0.01)} for _ in range(10)]
    agg  = aggregate_monte_carlo_metrics(runs)
    print(f"\naggregate_mc         OK :")
    for k, v in agg.items():
        print(f"  {k:<10} μ={v['mean']:.4f}  σ={v['std']:.4f}")

    print("\nAll loss.py tests PASSED.")