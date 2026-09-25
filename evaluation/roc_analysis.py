"""
evaluation/roc_analysis.py
---------------------------
ROC curve analysis with special focus on the low false-positive-rate (FPR)
regime, as reported in the paper Section IV-B.

Paper reference: Section IV-B, "ROC Curve Analysis (Low FPR Range)"

    "The ROC curve, focusing on the low FPR range, provides a detailed
     view of the model's performance in scenarios where low FPR is
     critical. The AUC-ROC for the low FPR range (0 to 0.1) was found to
     be 0.77, indicating that the model maintains good discrimination
     ability even when the tolerance for false positives is very low."

    "These focused evaluations are essential for high-precision applications,
     such as financial risk management and anomaly detection in stock
     market analysis."

Why the low-FPR region matters for shock propagation:
    In a financial risk-management context, a false positive (predicting a
    causal link that doesn't exist) can trigger unnecessary hedging or
    portfolio rebalancing — costly false alarms. The model's reliability in
    the low-FPR regime (where it is very conservative about claiming edges)
    is therefore more practically relevant than the full-range AUC-ROC,
    which gives equal weight to all FPR thresholds including those that are
    operationally unrealistic (no one acts on a model with 50% FPR).

Outputs:
    1. Full ROC curve (FPR vs TPR) — standard reference.
    2. Zoomed ROC curve restricted to FPR ∈ [0, 0.1] (paper Figure 4).
    3. Low-FPR AUC-ROC computed via restricted trapezoidal integration.
    4. Per-shock-period ROC comparison (overlay multiple curves).
    5. Optimal threshold selection via Youden's J statistic.

Usage:
    from evaluation.roc_analysis import ROCAnalyzer
    analyzer = ROCAnalyzer(y_true, y_scores)
    result   = analyzer.analyze()
    analyzer.plot(save=True)
    analyzer.plot_zoomed(save=True)

    # Multi-period comparison
    from evaluation.roc_analysis import compare_roc_curves
    compare_roc_curves({"GFC": (y_true_gfc, y_scores_gfc), ...}, save=True)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as CFG

logger = logging.getLogger(__name__)

# Paper's low-FPR analysis window
LOW_FPR_MAX   = 0.10
PAPER_AUC_FULL     = 0.77
PAPER_AUC_LOW_FPR  = 0.77


# ─────────────────────────────────────────────────────────────────────────────
# ROC ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

class ROCAnalyzer:
    """
    Computes and visualises ROC curve statistics for a single set of
    predictions, with emphasis on the low-FPR regime (paper Section IV-B).

    Args:
        y_true     : Binary ground-truth labels, shape (E,).
        y_scores   : Predicted probabilities ∈ [0, 1], shape (E,).
        label      : Display name for this curve (e.g. shock period name).
        low_fpr_max: Upper bound of the "low FPR" region. Paper: 0.10.
    """

    def __init__(
        self,
        y_true      : np.ndarray,
        y_scores    : np.ndarray,
        label       : str   = "Model",
        low_fpr_max : float = LOW_FPR_MAX,
    ):
        self.y_true      = np.asarray(y_true, dtype=float).flatten()
        self.y_scores    = np.asarray(y_scores, dtype=float).flatten()
        self.label       = label
        self.low_fpr_max = low_fpr_max

        if len(np.unique(self.y_true)) < 2:
            raise ValueError(
                f"y_true for '{label}' has only one class — ROC analysis "
                "requires both positive and negative examples."
            )

        self.fpr, self.tpr, self.thresholds = roc_curve(self.y_true, self.y_scores)

    # ─────────────────────────────────────────
    # CORE ANALYSIS
    # ─────────────────────────────────────────

    def analyze(self) -> Dict:
        """
        Run the full ROC analysis and return a structured results dict.

        Returns:
            {
                "auc_full"        : float — standard full-range AUC-ROC,
                "auc_low_fpr"     : float — AUC restricted to [0, low_fpr_max],
                "optimal_threshold": float — threshold maximising Youden's J,
                "optimal_tpr"     : float,
                "optimal_fpr"     : float,
                "youden_j"        : float — max(TPR - FPR),
                "n_pos"           : int,
                "n_neg"           : int,
                "vs_paper_full"   : float — difference from paper's 0.77,
                "vs_paper_low_fpr": float — difference from paper's 0.77 (low FPR),
            }
        """
        auc_full    = float(roc_auc_score(self.y_true, self.y_scores))
        auc_low_fpr = self._low_fpr_auc()

        opt_idx     = np.argmax(self.tpr - self.fpr)   # Youden's J statistic
        opt_thresh  = float(self.thresholds[opt_idx])
        opt_tpr     = float(self.tpr[opt_idx])
        opt_fpr     = float(self.fpr[opt_idx])
        youden_j    = float(opt_tpr - opt_fpr)

        n_pos = int(self.y_true.sum())
        n_neg = int(len(self.y_true) - n_pos)

        result = {
            "label"             : self.label,
            "auc_full"          : round(auc_full, 4),
            "auc_low_fpr"       : round(auc_low_fpr, 4),
            "optimal_threshold" : round(opt_thresh, 4),
            "optimal_tpr"       : round(opt_tpr, 4),
            "optimal_fpr"       : round(opt_fpr, 4),
            "youden_j"          : round(youden_j, 4),
            "n_pos"             : n_pos,
            "n_neg"             : n_neg,
            "vs_paper_full"     : round(auc_full - PAPER_AUC_FULL, 4),
            "vs_paper_low_fpr"  : round(auc_low_fpr - PAPER_AUC_LOW_FPR, 4),
        }

        logger.info(
            f"ROC[{self.label}] | "
            f"AUC_full={auc_full:.4f} | "
            f"AUC_low_fpr(0-{self.low_fpr_max})={auc_low_fpr:.4f} | "
            f"optimal_thresh={opt_thresh:.4f} (J={youden_j:.4f})"
        )
        return result

    def _low_fpr_auc(self) -> float:
        """
        Compute AUC-ROC restricted to FPR ∈ [0, low_fpr_max], normalised
        so the result is comparable to a standard 0-1 AUC scale.

        Paper Section IV-B: "The AUC-ROC for the low FPR range (0 to 0.1)
        was found to be 0.77."
        """
        mask    = self.fpr <= self.low_fpr_max
        fpr_low = self.fpr[mask]
        tpr_low = self.tpr[mask]

        if len(fpr_low) < 2:
            return 0.0

        # Close the integration region at exactly low_fpr_max via interpolation
        if fpr_low[-1] < self.low_fpr_max:
            interp_tpr = float(np.interp(self.low_fpr_max, self.fpr, self.tpr))
            fpr_low = np.append(fpr_low, self.low_fpr_max)
            tpr_low = np.append(tpr_low, interp_tpr)

        area = float(np.trapezoid(tpr_low, fpr_low))
        return area / self.low_fpr_max   # normalise to [0, 1] scale

    # ─────────────────────────────────────────
    # CURVE DATA  (for custom plotting / export)
    # ─────────────────────────────────────────

    def curve_df(self) -> pd.DataFrame:
        """Return the full ROC curve as a DataFrame (fpr, tpr, threshold)."""
        return pd.DataFrame({
            "fpr"       : self.fpr,
            "tpr"       : self.tpr,
            "threshold" : self.thresholds,
        })

    def low_fpr_curve_df(self) -> pd.DataFrame:
        """Return only the points within the low-FPR region."""
        df = self.curve_df()
        return df[df["fpr"] <= self.low_fpr_max].reset_index(drop=True)

    # ─────────────────────────────────────────
    # PLOTTING
    # ─────────────────────────────────────────

    def plot(
        self,
        figsize  : Tuple = (7, 6),
        save     : bool  = True,
        filename : str   = "roc_curve_full.png",
    ):
        """
        Plot the full ROC curve (0 to 1 FPR range).
        Shades the low-FPR region analysed separately by plot_zoomed().
        """
        import matplotlib.pyplot as plt
        from evaluation._plot_style import apply_style, save_figure

        apply_style()
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(self.fpr, self.tpr, color="#58a6ff", linewidth=1.6,
                label=f"{self.label} (AUC={roc_auc_score(self.y_true, self.y_scores):.3f})")
        ax.plot([0, 1], [0, 1], color="#8b949e", linewidth=0.8, linestyle="--",
                label="Random (AUC=0.50)")

        # Shade the low-FPR region examined in plot_zoomed()
        ax.axvspan(0, self.low_fpr_max, alpha=0.10, color="#f0883e",
                   label=f"Low-FPR region (0–{self.low_fpr_max})")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {self.label}", fontsize=11)
        ax.legend(loc="lower right", fontsize=8)
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
        filename : str   = "roc_curve_low_fpr.png",
    ):
        """
        Plot the ROC curve zoomed into the low-FPR region [0, low_fpr_max].
        This is the paper's Figure 4 equivalent (Section IV-B).
        """
        import matplotlib.pyplot as plt
        from evaluation._plot_style import apply_style, save_figure

        apply_style()
        fig, ax = plt.subplots(figsize=figsize)

        df = self.low_fpr_curve_df()
        auc_low = self._low_fpr_auc()

        ax.fill_between(df["fpr"], 0, df["tpr"], alpha=0.15, color="#58a6ff")
        ax.plot(df["fpr"], df["tpr"], color="#58a6ff", linewidth=1.8,
                label=f"{self.label} (AUC={auc_low:.3f})")
        ax.plot([0, self.low_fpr_max], [0, self.low_fpr_max],
                color="#8b949e", linewidth=0.8, linestyle="--", label="Random")

        # Paper reference line
        ax.axhline(PAPER_AUC_LOW_FPR, color="#f0883e", linewidth=0.8,
                   linestyle=":", label=f"Paper reference ({PAPER_AUC_LOW_FPR})")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(
            f"ROC Curve — Low FPR Range [0, {self.low_fpr_max}] — {self.label}",
            fontsize=11,
        )
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.4)
        ax.set_xlim(0, self.low_fpr_max)
        ax.set_ylim(0, 1.02)
        fig.tight_layout()

        if save:
            save_figure(fig, filename)
        return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-PERIOD COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def compare_roc_curves(
    data_dict   : Dict[str, Tuple[np.ndarray, np.ndarray]],
    low_fpr_max : float = LOW_FPR_MAX,
    figsize     : Tuple = (8, 7),
    save        : bool  = True,
    filename    : str   = "roc_comparison.png",
):
    """
    Overlay ROC curves from multiple shock periods (or folds) on one plot.

    Mirrors the paper's comparative analysis across the GFC, IL&FS crisis,
    and COVID-19 onset periods.

    Args:
        data_dict   : {period_name: (y_true, y_scores)}.
        low_fpr_max : Low-FPR cutoff for the inset zoom.
        save        : Save the figure to CFG.FIGURES_DIR.

    Returns:
        (fig, (ax_main, ax_inset), summary_df)
    """
    import matplotlib.pyplot as plt
    from evaluation._plot_style import apply_style, save_figure, COLOR_CYCLE

    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax_inset = fig.add_axes([0.50, 0.18, 0.36, 0.36])   # inset for low-FPR zoom

    rows = []
    for i, (name, (y_true, y_scores)) in enumerate(data_dict.items()):
        try:
            analyzer = ROCAnalyzer(y_true, y_scores, label=name, low_fpr_max=low_fpr_max)
        except ValueError as exc:
            logger.warning(f"Skipping '{name}': {exc}")
            continue

        result = analyzer.analyze()
        rows.append(result)

        colour = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        ax.plot(analyzer.fpr, analyzer.tpr, color=colour, linewidth=1.4,
                label=f"{name} (AUC={result['auc_full']:.3f})")

        low_df = analyzer.low_fpr_curve_df()
        ax_inset.plot(low_df["fpr"], low_df["tpr"], color=colour, linewidth=1.2)

    ax.plot([0, 1], [0, 1], color="#8b949e", linewidth=0.7, linestyle="--", alpha=0.7)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison Across Shock Periods", fontsize=11)
    ax.legend(loc="lower right", fontsize=7.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ax_inset.set_xlim(0, low_fpr_max)
    ax_inset.set_ylim(0, 1)
    ax_inset.set_title(f"Zoom: FPR ∈ [0, {low_fpr_max}]", fontsize=7)
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

    analyzer = ROCAnalyzer(y_true, y_scores, label="Test Shock")
    result   = analyzer.analyze()

    print("ROC Analysis Result:")
    for k, v in result.items():
        print(f"  {k:<20} {v}")

    assert 0 <= result["auc_full"] <= 1
    assert 0 <= result["auc_low_fpr"] <= 1
    assert 0 <= result["optimal_threshold"] <= 1

    df = analyzer.curve_df()
    assert {"fpr", "tpr", "threshold"}.issubset(df.columns)
    print(f"\nCurve points: {len(df)}")

    low_df = analyzer.low_fpr_curve_df()
    assert (low_df["fpr"] <= LOW_FPR_MAX).all()
    print(f"Low-FPR curve points: {len(low_df)}")

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
        fig3, axes3, summary = compare_roc_curves(data, save=True)
        assert len(summary) == 2

        saved = list(tmp.glob("*.png"))
        print(f"\n{len(saved)} figures saved: {[p.name for p in saved]}")
        assert len(saved) >= 2
    finally:
        cfg_mod.FIGURES_DIR = orig_figs
        shutil.rmtree(tmp)

    print("\nAll roc_analysis.py tests PASSED.")