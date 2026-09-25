"""
visualization/centrality_plot.py
----------------------------------
Bar-chart visualizations for centrality metrics: pre-training vs
post-training, for all four centrality metrics (paper Eq. 13–18).

Paper reference: Section IV-C, Figures 6–8 (2008 GFC), 12–14 (2016
elections), 18–20 (COVID-19 onset):

    Figures 6/12/18  — Degree of Connection  (Eq. 13): pre vs post
    Figures 7/13/19  — Betweenness & Closeness Centrality (Eq. 14-15): pre vs post
    Figures 8/14/20  — Degree Centrality  (Eq. 18): pre vs post

The paper's consistent narrative pattern across all three shock periods:
    PRE-TRAINING:  a few stocks (e.g. WFC, KO, PPG for GFC) have high degree,
                   implying their central role.
    POST-TRAINING: degree increases for many stocks, highlighting more
                   interconnected market during shocks.
    BETWEENNESS:   decreases post-training for almost all stocks, indicating
                   a more uniformly connected network (less reliance on bridges).
    CLOSENESS:     increases post-training — a tighter-knit network.
    CLUSTERING:    increases post-training — correlated stock movements.

This module provides five chart types:

    1. paired_bars()            — Paired horizontal bars for ONE metric,
                                  ONE shock period. Direct paper figure equivalent.
    2. all_metrics_grid()       — 2×2 subplot grid covering all four key
                                  centrality metrics in one figure.
    3. cross_shock_metric()     — Same metric across multiple shock periods
                                  (grouped bars) for comparative analysis.
    4. delta_ranked()           — Tickers ranked by Δ (post − pre) for one
                                  metric. Directly quantifies the paper's
                                  qualitative statements.
    5. persistence_chart()      — Scatter: mean_rank vs rank_std from
                                  CentralityAnalyzer.cross_shock_summary(),
                                  classifying systemic hubs vs crisis-specific
                                  stocks.

Usage:
    from visualization.centrality_plot import CentralityPlotter
    plotter = CentralityPlotter(adjacency_store)
    plotter.paired_bars("Global Financial Crisis", metric="betweenness", save=True)
    plotter.all_metrics_grid("Global Financial Crisis", save=True)
    plotter.cross_shock_metric(["GFC", "COVID"], metric="degree", save=True)
    plotter.delta_ranked("Global Financial Crisis", metric="betweenness", save=True)
    plotter.persistence_chart(["GFC", "IL&FS", "COVID"], metric="degree", save=True)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as CFG
from config.nifty50_universe import SECTOR_COLOURS, get_sector_map
from evaluation._plot_style import apply_style, save_figure, COLOR_CYCLE
from graph.adjacency import AdjacencyStore
from network_analysis.centrality import CentralityAnalyzer, CENTRALITY_METRICS

logger = logging.getLogger(__name__)

# Human-readable metric labels with equation references
METRIC_LABELS = {
    "degree"            : "Degree of Connection (Eq. 13)",
    "closeness"         : "Closeness Centrality (Eq. 14)",
    "betweenness"       : "Betweenness Centrality (Eq. 15)",
    "degree_centrality" : "Degree Centrality (Eq. 18)",
}

PRE_COLOUR  = "#58a6ff"   # blue  — pre-training (Granger)
POST_COLOUR = "#f0883e"   # orange — post-training (TGAT predicted)
GAIN_COLOUR = "#3fb950"   # green — positive delta (gained centrality)
LOSS_COLOUR = "#ff7b72"   # red   — negative delta (lost centrality)


class CentralityPlotter:
    """
    Bar-chart visualizations comparing pre- and post-training centrality.

    Wraps CentralityAnalyzer to produce publication-ready plots that
    directly mirror paper Figures 6–8, 12–14, and 18–20.

    Args:
        store : AdjacencyStore with saved pre/post adjacency matrices.
    """

    def __init__(self, store: Optional[AdjacencyStore] = None):
        self.store      = store or AdjacencyStore()
        self.analyzer   = CentralityAnalyzer(self.store)
        self.sector_map = get_sector_map()

    # ─────────────────────────────────────────
    # 1. PAIRED BAR CHART
    # ─────────────────────────────────────────

    def paired_bars(
        self,
        shock_name : str,
        metric     : str   = "betweenness",
        top_n      : int   = 20,
        figsize    : Tuple = (11, 8),
        save       : bool  = True,
        filename   : Optional[str] = None,
    ):
        """
        Horizontal paired bars: pre-training (blue) vs post-training (orange)
        for the top_n tickers by post-training value.

        Reproduces one paper figure panel, e.g.:
            metric="degree"           → paper Figure 6 / 12 / 18
            metric="betweenness"      → paper Figure 7 / 13 / 19
            metric="degree_centrality"→ paper Figure 8 / 14 / 20
            metric="closeness"        → paper Figure 7 / 13 / 19

        Y-tick labels are coloured by sector for quick visual attribution.

        The method auto-generates a one-sentence narrative annotation at
        the bottom of the chart mirroring the paper's description style:
        "Initially, [top_pre] leads. Post-training, [top_post] leads.
         Overall [metric] [increases/decreases] post-training."

        Args:
            top_n : Number of tickers to display.
        """
        apply_style()
        self._validate_metric(metric)

        # Load top-n for each stage
        try:
            pre_df  = self.analyzer.top_n_by_metric(
                shock_name, metric, stage="pre_training",  top_n=top_n)
            post_df = self.analyzer.top_n_by_metric(
                shock_name, metric, stage="post_training", top_n=top_n)
        except FileNotFoundError as exc:
            logger.warning(f"Cannot plot paired_bars: {exc}")
            return None, None

        # Union of tickers from both stages so neither is hidden
        ordered_tickers = list(dict.fromkeys(
            pre_df["ticker"].tolist() + post_df["ticker"].tolist()
        ))[:top_n]

        pre_vals  = pre_df.set_index("ticker")[metric].reindex(ordered_tickers).fillna(0)
        post_vals = post_df.set_index("ticker")[metric].reindex(ordered_tickers).fillna(0)

        # Sort ascending by post-training value (highest appears at top of hbar)
        order     = post_vals.sort_values(ascending=True).index
        pre_vals  = pre_vals.reindex(order)
        post_vals = post_vals.reindex(order)

        fig, ax = plt.subplots(figsize=figsize)
        y = np.arange(len(order))
        h = 0.38

        ax.barh(y + h/2, pre_vals.values,  height=h,
                color=PRE_COLOUR,  label="Pre-training",  alpha=0.88)
        ax.barh(y - h/2, post_vals.values, height=h,
                color=POST_COLOUR, label="Post-training", alpha=0.88)

        # Colour y-tick labels by sector
        ax.set_yticks(y)
        ax.set_yticklabels(order, fontsize=7.5)
        for lbl, ticker in zip(ax.get_yticklabels(), order):
            lbl.set_color(
                SECTOR_COLOURS.get(self.sector_map.get(ticker, "Unknown"), "#c9d1d9")
            )

        ax.set_xlabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(
            f"{METRIC_LABELS.get(metric, metric)}\n"
            f"Pre vs Post Training — {shock_name}",
            fontsize=11,
        )
        ax.legend(fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)

        # Auto-narrative annotation (paper style)
        top_pre   = pre_df.iloc[0]["ticker"]  if not pre_df.empty  else "N/A"
        top_post  = post_df.iloc[0]["ticker"] if not post_df.empty else "N/A"
        direction = "increases" if post_vals.mean() > pre_vals.mean() else "decreases"
        ax.text(
            0.02, 0.01,
            f"Initially: {top_pre} leads. Post-training: {top_post} leads. "
            f"Overall {metric.replace('_',' ')} {direction} post-training.",
            transform=ax.transAxes, fontsize=7,
            color="#8b949e", va="bottom", style="italic",
        )

        fig.tight_layout()
        fname = filename or f"centrality_{metric}_{self._slug(shock_name)}.png"
        if save:
            save_figure(fig, fname)
        return fig, ax

    # ─────────────────────────────────────────
    # 2. ALL-METRICS GRID  (2×2, one shock)
    # ─────────────────────────────────────────

    def all_metrics_grid(
        self,
        shock_name : str,
        top_n      : int   = 12,
        figsize    : Tuple = (18, 14),
        save       : bool  = True,
        filename   : Optional[str] = None,
    ):
        """
        2×2 subplot grid — one subplot per centrality metric.
        Collects the equivalent of paper Figures 6+7+8 into one layout.

        Useful for a single comprehensive centrality overview of one shock period
        without having to generate four separate figures.
        """
        apply_style()
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.patch.set_facecolor("#0f1117")
        axes_flat = axes.flatten()

        for ax, metric in zip(axes_flat, CENTRALITY_METRICS):
            try:
                pre_df  = self.analyzer.top_n_by_metric(
                    shock_name, metric, "pre_training",  top_n)
                post_df = self.analyzer.top_n_by_metric(
                    shock_name, metric, "post_training", top_n)
            except FileNotFoundError:
                ax.text(
                    0.5, 0.5, f"No data\n{metric}",
                    transform=ax.transAxes,
                    ha="center", va="center", color="#8b949e",
                )
                ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=9)
                continue

            ordered = list(dict.fromkeys(
                pre_df["ticker"].tolist() + post_df["ticker"].tolist()
            ))[:top_n]
            pre_v  = pre_df.set_index("ticker")[metric].reindex(ordered).fillna(0)
            post_v = post_df.set_index("ticker")[metric].reindex(ordered).fillna(0)
            order  = post_v.sort_values(ascending=True).index
            pre_v  = pre_v.reindex(order)
            post_v = post_v.reindex(order)

            y, h = np.arange(len(order)), 0.38
            ax.barh(y + h/2, pre_v.values,  height=h,
                    color=PRE_COLOUR,  alpha=0.85, label="Pre")
            ax.barh(y - h/2, post_v.values, height=h,
                    color=POST_COLOUR, alpha=0.85, label="Post")
            ax.set_yticks(y)
            ax.set_yticklabels(order, fontsize=6.5)
            for lbl, ticker in zip(ax.get_yticklabels(), order):
                lbl.set_color(
                    SECTOR_COLOURS.get(self.sector_map.get(ticker, "Unknown"), "#c9d1d9")
                )
            ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, axis="x", alpha=0.3)

        fig.suptitle(
            f"Centrality Metrics — Pre vs Post Training\n{shock_name}",
            fontsize=12, y=1.01,
        )
        fig.tight_layout()
        fname = filename or f"centrality_grid_{self._slug(shock_name)}.png"
        if save:
            save_figure(fig, fname)
        return fig, axes

    # ─────────────────────────────────────────
    # 3. CROSS-SHOCK METRIC  (same metric, multiple shocks)
    # ─────────────────────────────────────────

    def cross_shock_metric(
        self,
        shock_names : List[str],
        metric      : str   = "betweenness",
        stage       : str   = "post_training",
        top_n       : int   = 15,
        figsize     : Tuple = (13, 7),
        save        : bool  = True,
        filename    : Optional[str] = None,
    ):
        """
        Grouped horizontal bars: one shock period per colour, showing how
        the same centrality metric varies across the target shock periods.

        Answers: "Which stocks are consistently high-centrality across
        ALL shocks vs. which become central only during specific crises?"

        Uses post-training values by default (TGAT-learned structure)
        to emphasise the model's comparative findings.
        """
        apply_style()
        self._validate_metric(metric)

        all_data: Dict[str, pd.Series] = {}
        for shock_name in shock_names:
            try:
                df = self.analyzer.top_n_by_metric(
                    shock_name, metric, stage, top_n)
                all_data[shock_name] = df.set_index("ticker")[metric]
            except FileNotFoundError:
                logger.warning(f"Skipping '{shock_name}' — no {stage} matrix.")

        if not all_data:
            logger.warning("No data for cross_shock_metric plot.")
            return None, None

        # Union of top tickers across all shocks
        all_tickers = list(dict.fromkeys(
            t for series in all_data.values() for t in series.index
        ))[:top_n]

        wide = pd.DataFrame(
            {k: v.reindex(all_tickers).fillna(0) for k, v in all_data.items()},
            index=all_tickers,
        )
        # Sort by mean across shocks so most consistently-central tickers top the chart
        wide = wide.loc[wide.mean(axis=1).sort_values(ascending=True).index]

        n_shocks = len(all_data)
        total_h  = 0.75
        bar_h    = total_h / n_shocks
        y        = np.arange(len(wide))

        fig, ax = plt.subplots(figsize=figsize)
        for i, (shock_name, series) in enumerate(all_data.items()):
            vals   = series.reindex(wide.index).fillna(0)
            offset = (i - n_shocks / 2 + 0.5) * bar_h
            ax.barh(
                y + offset, vals.values,
                height=bar_h * 0.85,
                color=COLOR_CYCLE[i % len(COLOR_CYCLE)],
                label=shock_name[:45], alpha=0.88,
            )

        ax.set_yticks(y)
        ax.set_yticklabels(wide.index, fontsize=7.5)
        for lbl, ticker in zip(ax.get_yticklabels(), wide.index):
            lbl.set_color(
                SECTOR_COLOURS.get(self.sector_map.get(ticker, "Unknown"), "#c9d1d9")
            )

        ax.set_xlabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(
            f"{METRIC_LABELS.get(metric, metric)}\n"
            f"Cross-Shock Comparison [{stage.replace('_', ' ').title()}]",
            fontsize=11,
        )
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()

        fname = filename or f"centrality_cross_{metric}_{stage}.png"
        if save:
            save_figure(fig, fname)
        return fig, ax

    # ─────────────────────────────────────────
    # 4. DELTA RANKED  (who gained / lost centrality)
    # ─────────────────────────────────────────

    def delta_ranked(
        self,
        shock_name : str,
        metric     : str   = "betweenness",
        top_n      : int   = 20,
        figsize    : Tuple = (11, 8),
        save       : bool  = True,
        filename   : Optional[str] = None,
    ):
        """
        Ranked horizontal bar chart of Δ (post − pre) per ticker.

        Green bars = centrality GAINED post-training.
        Red   bars = centrality LOST post-training.

        Directly quantifies the paper's qualitative findings such as:
        "Post-training, betweenness centrality decreases significantly
         for almost all stocks, possibly indicating a more uniformly
         connected network. These changes are illustrated in Fig. 7."

        Shows top top_n//2 gainers and top top_n//2 losers so both
        directions of change are visible.
        """
        apply_style()
        self._validate_metric(metric)

        try:
            shift = self.analyzer.pre_post_shift_report(shock_name)
        except FileNotFoundError as exc:
            logger.warning(f"Cannot plot delta_ranked: {exc}")
            return None, None

        delta_col = f"{metric}_delta"
        if delta_col not in shift.columns:
            raise ValueError(
                f"Column '{delta_col}' not found. "
                f"Available delta columns: "
                f"{[c for c in shift.columns if c.endswith('_delta')]}"
            )

        delta = shift[delta_col].sort_values(ascending=False)
        n_show = top_n // 2

        gainers = delta.head(n_show)
        losers  = delta.tail(n_show).sort_values(ascending=True)
        display = pd.concat([gainers, losers])

        colours = [GAIN_COLOUR if v >= 0 else LOSS_COLOUR for v in display.values]

        fig, ax = plt.subplots(figsize=figsize)
        y = np.arange(len(display))
        ax.barh(y, display.values, color=colours, alpha=0.88)
        ax.axvline(0, color="#8b949e", linewidth=0.8)

        ax.set_yticks(y)
        ax.set_yticklabels(display.index, fontsize=7.5)
        for lbl, ticker in zip(ax.get_yticklabels(), display.index):
            lbl.set_color(
                SECTOR_COLOURS.get(self.sector_map.get(ticker, "Unknown"), "#c9d1d9")
            )

        ax.set_xlabel(
            f"Δ {METRIC_LABELS.get(metric, metric)}  (Post − Pre Training)"
        )
        ax.set_title(
            f"Centrality Change — Top Gainers & Losers\n"
            f"{METRIC_LABELS.get(metric, metric)} — {shock_name}",
            fontsize=11,
        )
        ax.legend(
            handles=[
                mpatches.Patch(color=GAIN_COLOUR, label="Gained centrality"),
                mpatches.Patch(color=LOSS_COLOUR, label="Lost centrality"),
            ],
            fontsize=8,
        )
        ax.grid(True, axis="x", alpha=0.3)

        # Annotation: fraction of stocks that decreased (paper's finding)
        frac_decrease = (delta < 0).mean()
        ax.text(
            0.98, 0.01,
            f"{frac_decrease:.0%} of stocks lost {metric.replace('_',' ')} post-training.",
            transform=ax.transAxes, fontsize=7,
            color="#8b949e", va="bottom", ha="right", style="italic",
        )

        fig.tight_layout()
        fname = filename or f"centrality_delta_{metric}_{self._slug(shock_name)}.png"
        if save:
            save_figure(fig, fname)
        return fig, ax

    # ─────────────────────────────────────────
    # 5. PERSISTENCE CHART  (systemic hub identification)
    # ─────────────────────────────────────────

    def persistence_chart(
        self,
        shock_names : List[str],
        metric      : str   = "degree",
        stage       : str   = "post_training",
        top_n       : int   = 25,
        figsize     : Tuple = (10, 8),
        save        : bool  = True,
        filename    : Optional[str] = None,
    ):
        """
        Scatter plot: mean centrality rank (y-axis) vs rank standard deviation
        (x-axis) for each ticker across all given shock periods.

        Quadrant interpretation:
            Low mean rank + Low std  → SYSTEMIC HUBS (consistently central)
            Low mean rank + High std → CRISIS-SPECIFIC hubs (central in some shocks)
            High mean rank           → PERIPHERAL stocks (rarely central)

        Complements centrality_persistence() from ShockComparator with a
        visual representation of the rank-stability argument.
        """
        apply_style()
        self._validate_metric(metric)

        cross = self.analyzer.cross_shock_summary(
            shock_names, metric=metric, stage=stage
        )
        if cross.empty:
            logger.warning("No data for persistence_chart.")
            return None, None

        # Show only the top_n most central on average
        cross = cross.head(top_n)

        N = len(cross)
        # Colour by persistence class using same thresholds as ShockComparator
        median_rank = cross["mean_rank"].median()
        std_thresh  = cross["rank_std"].median()

        def _cls(row):
            if row["mean_rank"] <= N * 0.20 and row["rank_std"] <= std_thresh:
                return "systemic_hub"
            elif row["mean_rank"] <= N * 0.30 and row["rank_std"] > std_thresh:
                return "crisis_specific"
            return "peripheral"

        cross["cls"] = cross.apply(_cls, axis=1)

        cls_colour = {
            "systemic_hub"   : "#f0883e",
            "crisis_specific": "#58a6ff",
            "peripheral"     : "#8b949e",
        }

        fig, ax = plt.subplots(figsize=figsize)
        for cls_name, grp in cross.groupby("cls"):
            ax.scatter(
                grp["rank_std"], grp["mean_rank"],
                color=cls_colour[cls_name],
                label=cls_name.replace("_", " ").title(),
                alpha=0.85, s=80, zorder=3,
            )

        # Label the most interesting stocks (systemic hubs)
        hubs = cross[cross["cls"] == "systemic_hub"]
        for _, row in hubs.iterrows():
            ax.annotate(
                row["ticker"],
                xy=(row["rank_std"], row["mean_rank"]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7, color="#f0883e",
            )

        # Quadrant lines
        ax.axhline(N * 0.20, color="#30363d", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.axvline(std_thresh, color="#30363d", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.text(std_thresh * 1.02, 1, "Low std\n(consistent)", fontsize=7, color="#8b949e")
        ax.text(std_thresh * 1.02, N * 0.20 * 1.05,
                "High rank\n(peripheral)", fontsize=7, color="#8b949e")

        ax.set_xlabel("Rank Std Deviation (low = consistent across shocks)")
        ax.set_ylabel("Mean Centrality Rank (lower = more central)")
        ax.invert_yaxis()   # rank 1 = most central → top of chart
        ax.set_title(
            f"Centrality Persistence — {METRIC_LABELS.get(metric, metric)}\n"
            f"{len(shock_names)} shock periods · {stage.replace('_',' ').title()}",
            fontsize=11,
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fname = filename or f"centrality_persistence_{metric}_{stage}.png"
        if save:
            save_figure(fig, fname)
        return fig, ax

    # ─────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────

    def _validate_metric(self, metric: str) -> None:
        if metric not in CENTRALITY_METRICS:
            raise ValueError(
                f"metric must be one of {CENTRALITY_METRICS}, got '{metric}'."
            )

    @staticmethod
    def _slug(name: str) -> str:
        return (
            name.lower()
            .replace(" ", "_")
            .replace("(", "").replace(")", "")
            .replace("&", "and")[:55]
        )


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, shutil
    logging.basicConfig(level="INFO", format=CFG.LOG_FORMAT)
    np.random.seed(13)

    from config.nifty50_universe import get_tickers
    tickers = get_tickers()
    N       = len(tickers)

    tmp = Path(tempfile.mkdtemp())
    import graph.adjacency as adj_mod
    orig_pre  = adj_mod.PRE_TRAINING_DIR
    orig_post = adj_mod.POST_TRAINING_DIR
    orig_tk   = adj_mod.TICKER_FILE
    adj_mod.PRE_TRAINING_DIR  = tmp / "pre_training"
    adj_mod.POST_TRAINING_DIR = tmp / "post_training"
    adj_mod.TICKER_FILE       = tmp / "tickers.json"

    try:
        from graph.adjacency import AdjacencyStore
        store = AdjacencyStore()
        shock_names = [
            "Global Financial Crisis",
            "COVID-19 Pandemic Onset",
        ]
        for name in shock_names:
            pre  = (np.random.rand(N, N) > 0.65).astype(np.float32)
            np.fill_diagonal(pre, 1.0)
            post = (np.random.rand(N, N) > 0.55).astype(np.float32)
            np.fill_diagonal(post, 1.0)
            store.save_all({name: pre},  tickers, stage="pre_training")
            store.save_all({name: post}, tickers, stage="post_training")

        import config.settings as cfg_mod
        orig_figs = cfg_mod.FIGURES_DIR
        cfg_mod.FIGURES_DIR = tmp / "figs"
        (tmp / "figs").mkdir(parents=True, exist_ok=True)

        try:
            plotter = CentralityPlotter(store)
            shock   = shock_names[0]

            # 1. paired_bars — one metric, one shock
            fig1, ax1 = plotter.paired_bars(shock, metric="betweenness", save=True)
            assert fig1 is not None
            print("paired_bars            OK : betweenness")

            # Check all four metrics work
            for m in CENTRALITY_METRICS:
                fig, _ = plotter.paired_bars(shock, metric=m, save=True)
                assert fig is not None
            print("paired_bars all metrics OK")

            # 2. all_metrics_grid — 2×2 subplot
            fig2, axes2 = plotter.all_metrics_grid(shock, top_n=10, save=True)
            assert fig2 is not None
            print("all_metrics_grid       OK")

            # 3. cross_shock_metric — multiple shocks
            fig3, ax3 = plotter.cross_shock_metric(
                shock_names, metric="degree", stage="post_training", save=True
            )
            assert fig3 is not None
            print("cross_shock_metric     OK")

            # 4. delta_ranked — gain/loss chart
            fig4, ax4 = plotter.delta_ranked(
                shock, metric="betweenness", top_n=20, save=True
            )
            assert fig4 is not None
            print("delta_ranked           OK")

            # 5. persistence_chart — scatter of rank stability
            fig5, ax5 = plotter.persistence_chart(
                shock_names, metric="degree", stage="post_training", save=True
            )
            assert fig5 is not None
            print("persistence_chart      OK")

            # Total: 4 (one per metric, paired_bars) + 1 grid + 1 cross + 1 delta + 1 persist = 8
            saved = list((tmp / "figs").glob("*.png"))
            print(f"Figures saved          : {len(saved)}")
            assert len(saved) == 8, f"Expected 8 PNGs, got {len(saved)}"

        finally:
            cfg_mod.FIGURES_DIR = orig_figs

        print("\nAll centrality_plot.py tests PASSED.")

    finally:
        adj_mod.PRE_TRAINING_DIR  = orig_pre
        adj_mod.POST_TRAINING_DIR = orig_post
        adj_mod.TICKER_FILE       = orig_tk
        shutil.rmtree(tmp)