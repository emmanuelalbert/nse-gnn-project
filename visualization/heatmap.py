"""
visualization/heatmap.py
Adjacency matrix heatmaps (paper Figure 2) and correlation heatmaps.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as CFG
from config.nifty50_universe import get_sector_map, SECTOR_COLOURS
from evaluation._plot_style import apply_style, save_figure
from graph.adjacency import AdjacencyStore
from shock_detection.detector import ShockPeriod

logger = logging.getLogger(__name__)

CMAP_BINARY    = "Blues"
CMAP_PREDICTED = "YlOrRd"
CMAP_DIFF      = "RdYlGn"
CMAP_CORR      = "coolwarm"


class HeatmapPlotter:
    """Adjacency matrix and correlation heatmap visualizations."""

    def __init__(self, store: Optional[AdjacencyStore] = None):
        self.store      = store or AdjacencyStore()
        self.sector_map = get_sector_map()

    def adjacency_heatmap(
        self,
        shock_name : str,
        stage      : str   = "pre_training",
        annotate   : bool  = False,
        figsize    : Tuple = (12, 10),
        save       : bool  = True,
        filename   : Optional[str] = None,
    ):
        """Single N×N adjacency heatmap — paper Figure 2 equivalent."""
        apply_style()
        matrix, tickers = self.store.load(shock_name, stage=stage)
        N = len(tickers)
        is_binary = set(np.unique(matrix[np.eye(N) == 0])).issubset({0.0, 1.0})
        cmap = CMAP_BINARY if is_binary else CMAP_PREDICTED

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("GC coefficient (Eq. 4)" if is_binary else "Predicted probability", fontsize=9)

        K = max(1, N // 20)
        tick_pos = list(range(0, N, K))
        ax.set_xticks(tick_pos); ax.set_xticklabels([tickers[i] for i in tick_pos], rotation=90, fontsize=6)
        ax.set_yticks(tick_pos); ax.set_yticklabels([tickers[i] for i in tick_pos], fontsize=6)

        if annotate and N <= 30:
            for i in range(N):
                for j in range(N):
                    ax.text(j, i, f"{matrix[i,j]:.0f}", ha="center", va="center",
                            fontsize=4.5, color="white" if matrix[i,j] > 0.5 else "#333")

        density = (matrix.sum() - np.trace(matrix)) / max(1, N * (N-1))
        ax.set_title(
            f"Adjacency Matrix — {shock_name}\n"
            f"[{stage.replace('_',' ').title()}]  |  density={density:.3f}", fontsize=11)
        ax.set_xlabel("Effect (Column j)")
        ax.set_ylabel("Cause (Row i)")
        fig.tight_layout()

        fname = filename or f"heatmap_{self._slug(shock_name)}_{stage}.png"
        if save: save_figure(fig, fname)
        return fig, ax

    def pre_post_diff_heatmap(
        self,
        shock_name : str,
        figsize    : Tuple = (22, 8),
        save       : bool  = True,
        filename   : Optional[str] = None,
    ):
        """Three-panel: pre | post | Δ(post-pre). Green=added, Red=removed."""
        apply_style()
        try:
            pre_mat,  tickers = self.store.load(shock_name, stage="pre_training")
            post_mat, _       = self.store.load(shock_name, stage="post_training")
        except FileNotFoundError as exc:
            logger.warning(f"Cannot plot diff: {exc}")
            return None, None

        N    = len(tickers)
        diff = post_mat - pre_mat
        panels = [
            (pre_mat,  CMAP_BINARY,    0,  1, "Pre-Training  (Granger)"),
            (post_mat, CMAP_PREDICTED, 0,  1, "Post-Training (TGAT Predicted)"),
            (diff,     CMAP_DIFF,      -1, 1, "Δ Post − Pre  (green=added, red=removed)"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        K = max(1, N // 15)
        tick_pos    = list(range(0, N, K))
        tick_labels = [tickers[i] for i in tick_pos]

        for ax, (mat, cmap, vmin, vmax, title) in zip(axes, panels):
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            ax.set_xticks(tick_pos); ax.set_xticklabels(tick_labels, rotation=90, fontsize=5.5)
            ax.set_yticks(tick_pos); ax.set_yticklabels(tick_labels, fontsize=5.5)
            ax.set_title(title, fontsize=9)

        n_added   = int((diff > 0).sum())
        n_removed = int((diff < 0).sum())
        fig.suptitle(
            f"Adjacency Comparison — {shock_name}\n"
            f"{n_added} edges added | {n_removed} removed", fontsize=11, y=1.02)
        fig.tight_layout()

        fname = filename or f"heatmap_diff_{self._slug(shock_name)}.png"
        if save: save_figure(fig, fname)
        return fig, axes

    def sector_block_heatmap(
        self,
        shock_name : str,
        stage      : str   = "pre_training",
        figsize    : Tuple = (13, 11),
        save       : bool  = True,
        filename   : Optional[str] = None,
    ):
        """Adjacency matrix with rows/columns sorted by sector, showing block causal structure."""
        apply_style()
        matrix, tickers = self.store.load(shock_name, stage=stage)
        N = len(tickers)

        sectors = [self.sector_map.get(t, "Unknown") for t in tickers]
        sector_order  = sorted(set(sectors))
        sorted_indices = []
        sector_starts  = {}
        for sec in sector_order:
            idxs = [i for i, s in enumerate(sectors) if s == sec]
            sector_starts[sec] = len(sorted_indices)
            sorted_indices.extend(idxs)

        mat_sorted  = matrix[np.ix_(sorted_indices, sorted_indices)]
        tick_labels = [tickers[i] for i in sorted_indices]
        is_binary   = set(np.unique(matrix[np.eye(N) == 0])).issubset({0.0, 1.0})

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(mat_sorted, cmap=CMAP_BINARY if is_binary else CMAP_PREDICTED,
                       vmin=0, vmax=1, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

        for sec, start in sector_starts.items():
            if start > 0:
                ax.axhline(start - 0.5, color="#f0883e", linewidth=0.8, alpha=0.6)
                ax.axvline(start - 0.5, color="#f0883e", linewidth=0.8, alpha=0.6)
            n_in_sec = sum(1 for s in sectors if s == sec)
            mid = start + n_in_sec / 2 - 0.5
            clr = SECTOR_COLOURS.get(sec, "#c9d1d9")
            ax.text(mid, -2, sec[:12], ha="center", fontsize=6, color=clr, rotation=45)
            ax.text(-2,  mid, sec[:12], ha="right",  fontsize=6, color=clr)

        K = max(1, N // 20)
        tick_pos = list(range(0, N, K))
        ax.set_xticks(tick_pos); ax.set_xticklabels([tick_labels[i] for i in tick_pos], rotation=90, fontsize=5.5)
        ax.set_yticks(tick_pos); ax.set_yticklabels([tick_labels[i] for i in tick_pos], fontsize=5.5)

        density = (mat_sorted.sum() - np.trace(mat_sorted)) / max(1, N*(N-1))
        ax.set_title(
            f"Sector-Block Adjacency — {shock_name}\n"
            f"[{stage.replace('_',' ').title()}]  |  density={density:.3f}", fontsize=11)
        fig.tight_layout()

        fname = filename or f"heatmap_sector_{self._slug(shock_name)}_{stage}.png"
        if save: save_figure(fig, fname)
        return fig, ax

    def correlation_heatmap(
        self,
        returns    : pd.DataFrame,
        shock      : ShockPeriod,
        figsize    : Tuple = (13, 11),
        save       : bool  = True,
        filename   : Optional[str] = None,
    ):
        """Pearson correlation of log returns over a shock window, sorted by sector."""
        apply_style()
        mask   = (returns.index >= pd.Timestamp(shock.start)) & (returns.index <= pd.Timestamp(shock.end))
        period = returns.loc[mask]
        if period.empty:
            raise ValueError(f"No data for [{shock.start} → {shock.end}]")

        corr    = period.corr()
        tickers = list(corr.index)
        N       = len(tickers)
        sectors = [self.sector_map.get(t, "Unknown") for t in tickers]
        sorted_indices = []
        for sec in sorted(set(sectors)):
            sorted_indices.extend([i for i, s in enumerate(sectors) if s == sec])
        sorted_tickers = [tickers[i] for i in sorted_indices]
        corr_sorted    = corr.iloc[sorted_indices, sorted_indices]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(corr_sorted.values, cmap=CMAP_CORR, vmin=-1, vmax=1, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Pearson r")

        K = max(1, N // 20)
        tick_pos = list(range(0, N, K))
        ax.set_xticks(tick_pos); ax.set_xticklabels([sorted_tickers[i] for i in tick_pos], rotation=90, fontsize=5.5)
        ax.set_yticks(tick_pos); ax.set_yticklabels([sorted_tickers[i] for i in tick_pos], fontsize=5.5)

        mean_corr = corr.values[~np.eye(N, dtype=bool)].mean()
        ax.set_title(
            f"Return Correlation — {shock.name}\n"
            f"[{shock.start} → {shock.end}]  |  mean r={mean_corr:.3f}", fontsize=11)
        fig.tight_layout()

        fname = filename or f"corr_{self._slug(shock.name)}.png"
        if save: save_figure(fig, fname)
        return fig, ax

    @staticmethod
    def _slug(name: str) -> str:
        return name.lower().replace(" ", "_").replace("(","").replace(")","")[:55]


if __name__ == "__main__":
    import tempfile, shutil
    logging.basicConfig(level="INFO", format=CFG.LOG_FORMAT)
    np.random.seed(17)
    from config.nifty50_universe import get_tickers
    tickers = get_tickers(); N = len(tickers)
    tmp = Path(tempfile.mkdtemp())
    import graph.adjacency as adj_mod
    orig_pre, orig_post, orig_tk = adj_mod.PRE_TRAINING_DIR, adj_mod.POST_TRAINING_DIR, adj_mod.TICKER_FILE
    adj_mod.PRE_TRAINING_DIR  = tmp/"pre"; adj_mod.POST_TRAINING_DIR = tmp/"post"; adj_mod.TICKER_FILE = tmp/"t.json"
    try:
        store = AdjacencyStore()
        pre  = (np.random.rand(N,N)>0.65).astype(np.float32); np.fill_diagonal(pre,1.0)
        post = (np.random.rand(N,N)>0.55).astype(np.float32); np.fill_diagonal(post,1.0)
        store.save_all({"GFC": pre}, tickers, stage="pre_training")
        store.save_all({"GFC": post}, tickers, stage="post_training")
        dates   = pd.bdate_range("2008-09-15", "2008-10-10")
        returns = pd.DataFrame(np.random.normal(0,0.015,(len(dates),N)), index=dates, columns=tickers)
        shock   = ShockPeriod("GFC","2008-09-15","2008-10-10",18,-0.03,-0.08,4.0)
        import config.settings as c; orig_figs=c.FIGURES_DIR; c.FIGURES_DIR=tmp/"figs"
        try:
            p = HeatmapPlotter(store)
            assert p.adjacency_heatmap("GFC", save=True)[0] is not None;       print("adjacency_heatmap      OK")
            assert p.pre_post_diff_heatmap("GFC", save=True)[0] is not None;   print("pre_post_diff_heatmap  OK")
            assert p.sector_block_heatmap("GFC", save=True)[0] is not None;    print("sector_block_heatmap   OK")
            assert p.correlation_heatmap(returns, shock, save=True)[0] is not None; print("correlation_heatmap    OK")
            saved = list((tmp/"figs").glob("*.png"))
            print(f"Figures saved          : {len(saved)}")
            assert len(saved) == 4
        finally:
            c.FIGURES_DIR = orig_figs
        print("\nAll heatmap.py tests PASSED.")
    finally:
        adj_mod.PRE_TRAINING_DIR=orig_pre; adj_mod.POST_TRAINING_DIR=orig_post; adj_mod.TICKER_FILE=orig_tk
        shutil.rmtree(tmp)