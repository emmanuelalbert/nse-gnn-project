"""
evaluation/pr_analysis.py
---------------------------
Precision-Recall curve analysis with special focus on the low-recall
regime, as reported in the paper Section IV-B.

Paper reference: Section IV-B, "Precision-Recall Curve Analysis"

    "The Precision-Recall curve offers insights into the trade-off between
     precision and recall. The overall AUPR was calculated to be 0.76,
     underscoring the model's effectiveness in predicting the positive
     class. Focusing on the low recall range, the curve remains relatively
     stable, suggesting that the model performs well in detecting true
     positives without introducing a high number of false positives."

Why the low-recall region matters for shock propagation:
    Precision at low recall measures how trustworthy the model's most
    confident predictions are — the few causal links it is most sure about.
    In a financial risk-management context, these are exactly the edges an
    analyst would act on first (e.g. flagging the handful of stocks most
    likely to transmit a shock). High precision in this regime means the
    model's top-ranked predictions are reliable even before considering
    its full recall range, which matters more operationally than the
    full-range AUPR that averages over recall levels no one would act on
    at once.

Outputs:
    1. Full Precision-Recall curve (Recall vs Precision) — standard reference.
    2. Zoomed PR curve restricted to Recall ∈ [0, low_recall_max] (paper's
       "low recall range" discussion, mirroring roc_analysis.py's low-FPR zoom).
    3. Low-recall AUPR computed via restricted trapezoidal integration.
    4. Per-shock-period PR comparison (overlay multiple curves).
    5. Optimal threshold selection via F1-maximising point on the curve.

Usage:
    from evaluation.pr_analysis import PRAnalyzer
    analyzer = PRAnalyzer(y_true, y_scores)
    result   = analyzer.analyze()
    analyzer.plot(save=True)
    analyzer.plot_zoomed(save=True)

    # Multi-period comparison
    from evaluation.pr_analysis import compare_pr_curves
    compare_pr_curves({"GFC": (y_true_gfc, y_scores_gfc), ...}, save=True)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as CFG

logger = logging.getLogger(__name__)

# Paper's low-recall analysis window (mirrors roc_analysis.py's LOW_FPR_MAX
# convention — no explicit numeric bound is given in the paper text for the
# "low recall range", so we use the same 0.10 scale as the ROC low-FPR
# analysis for a consistent, comparably-sized focus region).
LOW_RECALL_MAX  = 0.10
PAPER_AUPR_FULL = 0.76


# ─────────────────────────────────────────────────────────────────────────────
# PR ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

class PRAnalyzer:
    """
    Computes and visualises Precision-Recall curve statistics for a single
    set of predictions, with emphasis on the low-recall regime (paper
    Section IV-B).

    Args:
        y_true        : Binary ground-truth labels, shape (E,).
        y_scores      : Predicted probabilities ∈ [0, 1], shape (E,).
        label         : Display name for this curve (e.g. shock period name).
        low_recall_max: Upper bound of the "low recall" region.
    """

    def __init__(
        self,
        y_true         : np.ndarray,
        y_scores       : np.ndarray,
        label          : str   = "Model",
        low_recall_max : float = LOW_RECALL_MAX,
    ):
        self.y_true         = np.asarray(y_true, dtype=float).flatten()
        self.y_scores       = np.asarray(y_scores, dtype=float).flatten()
        self.label           = label
        self.low_recall_max  = low_recall_max

        if len(np.unique(self.y_true)) < 2:
            raise ValueError(
                f"y_true for '{label}' has only one class — PR analysis "
                "requires both positive and negative examples."
            )

        # sklearn returns precision/recall with one extra point (1, 0) and
        # thresholds one element shorter than precision/recall.
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_scores)
        self.precision = precision
        self.recall    = recall
        self.thresholds = thresholds

    # ─────────────────────────────────────────
    # CORE ANALYSIS
    # ─────────────────────────────────────────

    def analyze(self) -> Dict:
        """
        Run the full PR analysis and return a structured results dict.

        Returns:
            {
                "aupr_full"        : float — standard full-range AUPR (Eq. 7),
                "aupr_low_recall"  : float — AUPR restricted to [0, low_recall_max],
                "optimal_threshold": float — threshold maximising F1,
                "optimal_precision": float,
                "optimal_recall"   : float,
                "optimal_f1"       : float,
                "n_pos"            : int,
                "n_neg"            : int,
                "vs_paper_full"    : float — difference from paper's 0.76,
            }
        """
        aupr_full       = float(average_precision_score(self.y_true, self.y_scores))
        aupr_low_recall = self._low_recall_aupr()

        opt_thresh, opt_p, opt_r, opt_f1 = self._best_f1_point()

        n_pos = int(self.y_true.sum())
        n_neg = int(len(self.y_true) - n_pos)

        result = {
            "label"             : self.label,
            "aupr_full"         : round(aupr_full, 4),
            "aupr_low_recall"   : round(aupr_low_recall, 4),
            "optimal_threshold" : round(opt_thresh, 4),
            "optimal_precision" : round(opt_p, 4),
            "optimal_recall"    : round(opt_r, 4),
            "optimal_f1"        : round(opt_f1, 4),
            "n_pos"             : n_pos,
            "n_neg"             : n_neg,
            "vs_paper_full"     : round(aupr_full - PAPER_AUPR_FULL, 4),
        }

        logger.info(
            f"PR[{self.label}] | "
            f"AUPR_full={aupr_full:.4f} | "
            f"AUPR_low_recall(0-{self.low_recall_max})={aupr_low_recall:.4f} | "
            f"optimal_thresh={opt_thresh:.4f} (F1={opt_f1:.4f})"
        )
        return result

    def _low_recall_aupr(self) -> float:
        """
        Compute AUPR restricted to Recall ∈ [0, low_recall_max], normalised
        so the result is comparable to a standard 0-1 AUPR scale.

        Mirrors ROCAnalyzer._low_fpr_auc()'s restricted trapezoidal
        integration, but along the recall axis instead of FPR.
        """
        # precision_recall_curve returns recall in DEcreasing order (starts
        # near 1, ends at 0) — sort ascending by recall for integration.
        order   = np.argsort(self.recall)
        recall_sorted    = self.recall[order]
        precision_sorted = self.precision[order]

        mask = recall_sorted <= self.low_recall_max
        recall_low    = recall_sorted[mask]
        precision_low = precision_sorted[mask]

        if len(recall_low) < 2:
            return 0.0

        if recall_low[-1] < self.low_recall_max:
            interp_prec = float(np.interp(self.low_recall_max, recall_sorted, precision_sorted))
            recall_low    = np.append(recall_low, self.low_recall_max)
            precision_low = np.append(precision_low, interp_prec)

        area = float(np.trapezoid(precision_low, recall_low))
        return area / self.low_recall_max   # normalise to [0, 1] scale

    def _best_f1_point(self) -> Tuple[float, float, float, float]:
        """
        Find the threshold that maximises F1 along the PR curve.

        Returns:
            (threshold, precision, recall, f1) at the best point.
            Falls back to (0.5, 0, 0, 0) if no valid threshold exists
            (e.g. degenerate curve with a single point).
        """
        if len(self.thresholds) == 0:
            return 0.5, 0.0, 0.0, 0.0

        # precision/recall arrays are one element longer than thresholds
        p = self.precision[:-1]
        r = self.recall[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)

        best_idx = int(np.argmax(f1))
        return (
            float(self.thresholds[best_idx]),
            float(p[best_idx]),
            float(r[best_idx]),
            float(f1[best_idx]),
        )

    # ─────────────────────────────────────────
    # CURVE DATA  (for custom plotting / export)
    # ─────────────────────────────────────────

    def curve_df(self) -> pd.DataFrame:
        """
        Return the full PR curve as a DataFrame (recall, precision, threshold).
        The last (precision, recall) point has no corresponding threshold
        (sklearn convention: precision=1, recall=0 sentinel) and is dropped
        here so all three columns align.
        """
        return pd.DataFrame({
            "recall"    : self.recall[:-1],
            "precision" : self.precision[:-1],
            "threshold" : self.thresholds,
        })

    def low_recall_curve_df(self) -> pd.DataFrame:
        """Return only the points within the low-recall region."""
        df = self.curve_df()
        return df[df["recall"] <= self.low_recall_max].reset_index(drop=True)

    # ─────────────────────────────────────────
    # PLOTTING
    # ─────────────────────────────────────────

    def plot(
        self,
        figsize  : Tuple = (7, 6),
        save     : bool  = True,
        filename : str   = "pr_curve_full.png",
    ):
        """
        Plot the full Precision-Recall curve (0 to 1 recall range).
        Shades the low-recall region analysed separately by plot_zoomed().
        """
        import matplotlib.pyplot as plt
        from evaluation._plot_style import apply_style, save_figure

        apply_style()
        fig, ax = plt.subplots(figsize=figsize)

        aupr = average_precision_score(self.y_true, self.y_scores)
        ax.plot(self.recall, self.precision, color="#58a6ff", linewidth=1.6,
                label=f"{self.label} (AUPR={aupr:.3f})")

        pos_rate = float(self.y_true.mean())
        ax.axhline(pos_rate, color="#8b949e", linewidth=0.8, linestyle="--",
                   label=f"Random (AUPR={pos_rate:.2f})")

        # Shade the low-recall region examined in plot_zoomed()
        ax.axvspan(0, self.low_recall_max, alpha=0.10, color="#f0883e",
                   label=f"Low-recall region (0–{self.low_recall_max})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve — {self.label}", fontsize=11)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=0.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        fig.tight_layout()

        if save:
            save_figure(fig, filename)
        return fig, ax

    def plot_zoomed(
        self,
        figsize  : Tuple = (7, 6),
        save     : bool  = True,
        filename : str   = "pr_curve_low_recall.png",
    ):
        """
        Plot the PR curve zoomed into the low-recall region [0, low_recall_max].
        Mirrors roc_analysis.py's plot_zoomed() (paper Figure 4 equivalent,
        Section IV-B) but along the precision-recall axes.
        """
        import matplotlib.pyplot as plt
        from evaluation._plot_style import apply_style, save_figure

        apply_style()
        fig, ax = plt.subplots(figsize=figsize)

        df = self.low_recall_curve_df()
        aupr_low = self._low_recall_aupr()

        ax.fill_between(df["recall"], 0, df["precision"], alpha=0.15, color="#58a6ff")
        ax.plot(df["recall"], df["precision"], color="#58a6ff", linewidth=1.8,
                label=f"{self.label} (AUPR={aupr_low:.3f})")

        pos_rate = float(self.y_true.mean())
        ax.axhline(pos_rate, color="#8b949e", linewidth=0.8, linestyle="--",
                   label="Random")

        # Paper reference line
        ax.axhline(PAPER_AUPR_FULL, color="#f0883e", linewidth=0.8,
                   linestyle=":", label=f"Paper reference ({PAPER_AUPR_FULL})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(
            f"Precision-Recall Curve — Low Recall Range [0, {self.low_recall_max}] — {self.label}",
            fontsize=11,
        )
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=0.4)
        ax.set_xlim(0, self.low_recall_max)
        ax.set_ylim(0, 1.02)
        fig.tight_layout()

        if save:
            save_figure(fig, filename)
        return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-PERIOD COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def compare_pr_curves(
    data_dict      : Dict[str, Tuple[np.ndarray, np.ndarray]],
    low_recall_max : float = LOW_RECALL_MAX,
    figsize        : Tuple = (8, 7),
    save           : bool  = True,
    filename       : str   = "pr_comparison.png",
):
    """
    Overlay Precision-Recall curves from multiple shock periods (or folds)
    on one plot.

    Mirrors roc_analysis.py's compare_roc_curves() and the paper's
    comparative analysis across the GFC, IL&FS crisis, and COVID-19 onset
    periods.

    Args:
        data_dict      : {period_name: (y_true, y_scores)}.
        low_recall_max : Low-recall cutoff for the inset zoom.
        save           : Save the figure to CFG.FIGURES_DIR.

    Returns:
        (fig, (ax_main, ax_inset), summary_df)
    """
    import matplotlib.pyplot as plt
    from evaluation._plot_style import apply_style, save_figure, COLOR_CYCLE

    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax_inset = fig.add_axes([0.50, 0.55, 0.36, 0.36])   # inset for low-recall zoom

    rows = []
    for i, (name, (y_true, y_scores)) in enumerate(data_dict.items()):
        try:
            analyzer = PRAnalyzer(y_true, y_scores, label=name, low_recall_max=low_recall_max)
        except ValueError as exc:
            logger.warning(f"Skipping '{name}': {exc}")
            continue

        result = analyzer.analyze()
        rows.append(result)

        colour = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        ax.plot(analyzer.recall, analyzer.precision, color=colour, linewidth=1.4,
                label=f"{name} (AUPR={result['aupr_full']:.3f})")

        low_df = analyzer.low_recall_curve_df()
        ax_inset.plot(low_df["recall"], low_df["precision"], color=colour, linewidth=1.2)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve Comparison Across Shock Periods", fontsize=11)
    ax.legend(loc="lower left", fontsize=7.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ax_inset.set_xlim(0, low_recall_max)
    ax_inset.set_ylim(0, 1)
    ax_inset.set_title(f"Zoom: Recall ∈ [0, {low_recall_max}]", fontsize=7)
    ax_inset.tick_params(labelsize=6)
    ax_inset.grid(True, alpha=0.3)

    fig.tight_layout()

    if save:
        save_figure(fig, filename)

    summary_df = pd.DataFrame(rows)
    return fig, (ax, ax_inset), summary_df


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, shutil
    logging.basicConfig(level="INFO", format=CFG.LOG_FORMAT)
    np.random.seed(42)

    # Synthetic predictions roughly matching paper performance
    n = 2000
    y_true   = np.random.binomial(1, 0.4, n)
    y_scores = np.clip(
        y_true * np.random.beta(6, 2, n) + (1 - y_true) * np.random.beta(2, 5, n),
        0, 1,
    )

    analyzer = PRAnalyzer(y_true, y_scores, label="Test Shock")
    result   = analyzer.analyze()

    print("PR Analysis Result:")
    for k, v in result.items():
        print(f"  {k:<20} {v}")

    assert 0 <= result["aupr_full"] <= 1
    assert 0 <= result["aupr_low_recall"] <= 1
    assert 0 <= result["optimal_precision"] <= 1
    assert 0 <= result["optimal_recall"] <= 1

    df = analyzer.curve_df()
    assert {"recall", "precision", "threshold"}.issubset(df.columns)
    print(f"\nCurve points: {len(df)}")

    low_df = analyzer.low_recall_curve_df()
    assert (low_df["recall"] <= LOW_RECALL_MAX).all()
    print(f"Low-recall curve points: {len(low_df)}")

    # ── Plotting test ──
    tmp = Path(tempfile.mkdtemp())
    orig_figs = CFG.FIGURES_DIR
    import config.settings as cfg_mod
    cfg_mod.FIGURES_DIR = tmp

    try:
        fig1, ax1 = analyzer.plot(save=True)
        fig2, ax2 = analyzer.plot_zoomed(save=True)

        # Multi-period comparison
        data = {
            "GFC"   : (y_true, y_scores),
            "COVID" : (y_true, np.clip(y_scores + np.random.normal(0, 0.05, n), 0, 1)),
        }
        fig3, axes3, summary = compare_pr_curves(data, save=True)
        assert len(summary) == 2

        saved = list(tmp.glob("*.png"))
        print(f"\n{len(saved)} figures saved: {[p.name for p in saved]}")
        assert len(saved) >= 2
    finally:
        cfg_mod.FIGURES_DIR = orig_figs
        shutil.rmtree(tmp)

    print("\nAll pr_analysis.py tests PASSED.")