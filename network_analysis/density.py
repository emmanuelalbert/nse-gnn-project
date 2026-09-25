"""
network_analysis/density.py
-----------------------------
Network density and clustering coefficient reporting across shock periods.

Paper reference: Section III-E (Eq. 16–17) and Section IV-C

    Eq. 16 — Average Clustering Coefficient:
        C̄ = (1/|V|) Σ_{v∈V} C(v)
        "Reveals clustering tendencies in the market, indicating collective
         movements during shocks."

    Eq. 17 — Network Density:
        Density = 2|E| / (|V|(|V|-1))
        "Indicates the overall interconnectivity of the market, highlighting
         network robustness during shocks."

    Paper narrative pattern (repeated for each of the 3 analysed periods,
    e.g. Section IV-C-1-c, "Network Indicators"):

        "Pre-training, the network shows a density of 0.57 and a clustering
         coefficient of 0.53. Post-training, the density slightly decreases
         to 0.49, while the clustering coefficient increases to 0.70. These
         changes suggest a more tightly clustered network, typical in times
         of market shocks where correlations between stocks tend to increase."

    This is a consistent finding across all three paper shock periods:
        density   : decreases slightly post-training
        clustering: increases post-training
    interpreted as the TGCN/TGAT learning a more "tightly-knit" causal
    structure than the raw Granger matrix, concentrated among fewer but
    more cohesive groups of stocks.

graph/adjacency.py::MatrixMetrics.density() and .clustering() already
compute the raw scalar values for one matrix. This module is the
ANALYTICAL layer: it runs these across every shock period (pre vs post),
reproduces the paper's exact narrative table format, and visualises the
density/clustering trend across the full shock history (not just the
3-4 target periods).

Usage:
    from network_analysis.density import DensityAnalyzer
    analyzer = DensityAnalyzer(adjacency_store)
    report   = analyzer.network_indicators_report("Global Financial Crisis")
    all_rpt  = analyzer.all_shocks_report(shock_names)
    analyzer.plot_density_clustering("Global Financial Crisis", save=True)
    analyzer.plot_trend(shock_names, save=True)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as CFG
from graph.adjacency import AdjacencyStore, MatrixMetrics

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DENSITY ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

class DensityAnalyzer:
    """
    Network density and clustering coefficient analysis across shock periods,
    pre- and post-training (paper Eq. 16–17, Section IV-C "Network Indicators").

    Args:
        store : AdjacencyStore instance with saved pre/post matrices.
    """

    def __init__(self, store: Optional[AdjacencyStore] = None):
        self.store = store or AdjacencyStore()

    # ─────────────────────────────────────────
    # SINGLE-PERIOD "NETWORK INDICATORS" REPORT
    # ─────────────────────────────────────────

    def network_indicators_report(
        self,
        shock_name : str,
    ) -> Dict[str, float]:
        """
        Reproduce the paper's "Network Indicators" sub-section for one
        shock period (Section IV-C-1-c, IV-C-2-c, IV-C-3-c).

        Example paper text reproduced by this method's output:
            "Pre-training, the network shows a density of 0.57 and a
             clustering coefficient of 0.53. Post-training, the density
             slightly decreases to 0.49, while the clustering coefficient
             increases to 0.70."

        Returns:
            {
                "shock_name"           : str,
                "density_pre"          : float,
                "density_post"         : float,
                "density_delta"        : float,
                "clustering_pre"       : float,
                "clustering_post"      : float,
                "clustering_delta"     : float,
                "interpretation"       : str — auto-generated narrative,
            }
        """
        pre_mat, tickers = self.store.load(shock_name, stage="pre_training")
        try:
            post_mat, _ = self.store.load(shock_name, stage="post_training")
        except FileNotFoundError:
            logger.warning(
                f"No post-training matrix for '{shock_name}' — "
                "density/clustering will be pre-training only."
            )
            post_mat = None

        pre_m = MatrixMetrics(pre_mat, tickers)
        density_pre        = pre_m.density()
        _, clustering_pre  = pre_m.clustering()

        result = {
            "shock_name"      : shock_name,
            "density_pre"     : round(density_pre, 4),
            "clustering_pre"  : round(clustering_pre, 4),
        }

        if post_mat is not None:
            post_m = MatrixMetrics(post_mat, tickers)
            density_post       = post_m.density()
            _, clustering_post = post_m.clustering()

            result.update({
                "density_post"       : round(density_post, 4),
                "density_delta"      : round(density_post - density_pre, 4),
                "clustering_post"    : round(clustering_post, 4),
                "clustering_delta"   : round(clustering_post - clustering_pre, 4),
            })
            result["interpretation"] = self._generate_narrative(result)

        logger.info(
            f"Network indicators '{shock_name}' | "
            f"density: {result.get('density_pre','?')} → {result.get('density_post','?')} | "
            f"clustering: {result.get('clustering_pre','?')} → {result.get('clustering_post','?')}"
        )
        return result

    def _generate_narrative(self, result: Dict) -> str:
        """
        Auto-generate a paper-style narrative sentence describing the
        density/clustering shift, matching the pattern seen in all three
        of the paper's analysed periods.
        """
        d_delta = result["density_delta"]
        c_delta = result["clustering_delta"]

        d_dir = "decreases" if d_delta < 0 else ("increases" if d_delta > 0 else "remains stable")
        c_dir = "increases" if c_delta > 0 else ("decreases" if c_delta < 0 else "remains stable")

        tightening = c_delta > 0 and d_delta <= 0
        tightening_phrase = (
            "These changes suggest a more tightly clustered network, "
            "typical in times of market shocks where correlations between "
            "stocks tend to increase."
            if tightening else
            "These changes suggest a shift in network cohesion that does "
            "not follow the typical shock-tightening pattern."
        )

        return (
            f"Pre-training, the network shows a density of {result['density_pre']:.2f} "
            f"and a clustering coefficient of {result['clustering_pre']:.2f}. "
            f"Post-training, the density {d_dir} to {result['density_post']:.2f}, "
            f"while the clustering coefficient {c_dir} to {result['clustering_post']:.2f}. "
            f"{tightening_phrase}"
        )

    # ─────────────────────────────────────────
    # MULTI-SHOCK BATCH REPORT
    # ─────────────────────────────────────────

    def all_shocks_report(
        self,
        shock_names : List[str],
    ) -> pd.DataFrame:
        """
        Run network_indicators_report() for multiple shock periods and
        return a tabular summary.

        Returns:
            DataFrame with one row per shock:
                shock_name, density_pre, density_post, density_delta,
                clustering_pre, clustering_post, clustering_delta,
                tightening (bool — matches the paper's typical shock pattern)
        """
        rows = []
        for name in shock_names:
            try:
                r = self.network_indicators_report(name)
            except FileNotFoundError as exc:
                logger.warning(f"Skipping '{name}': {exc}")
                continue
            if "density_post" not in r:
                continue
            row = {k: v for k, v in r.items() if k != "interpretation"}
            row["tightening"] = (
                row["clustering_delta"] > 0 and row["density_delta"] <= 0
            )
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            n_tightening = df["tightening"].sum()
            logger.info(
                f"all_shocks_report: {n_tightening}/{len(df)} shocks show "
                "the paper's typical 'tightening' pattern (clustering↑, density↓)."
            )
        return df

    # ─────────────────────────────────────────
    # TREND ACROSS FULL SHOCK HISTORY
    # ─────────────────────────────────────────

    def density_trend(
        self,
        shock_names : List[str],
        stage       : str = "pre_training",
    ) -> pd.DataFrame:
        """
        Compute density and clustering for every shock in chronological
        order, for trend analysis across the full 2005-present history
        (not just the 3-4 target periods).

        Args:
            shock_names : Ordered list of shock period names.
            stage       : "pre_training" | "post_training".

        Returns:
            DataFrame: shock_name, density, clustering, n_edges.
        """
        rows = []
        for name in shock_names:
            try:
                matrix, tickers = self.store.load(name, stage=stage)
            except FileNotFoundError:
                continue
            m = MatrixMetrics(matrix, tickers)
            _, avg_clust = m.clustering()
            rows.append({
                "shock_name" : name,
                "density"    : round(m.density(), 4),
                "clustering" : round(avg_clust, 4),
                "n_edges"    : int(matrix.sum() - np.trace(matrix)),
            })
        return pd.DataFrame(rows)

    # ─────────────────────────────────────────
    # PLOTTING
    # ─────────────────────────────────────────

    def plot_density_clustering(
        self,
        shock_name : str,
        figsize    : Tuple = (8, 5),
        save       : bool  = True,
        filename   : Optional[str] = None,
    ):
        """
        Paired bar chart of density and clustering coefficient,
        pre vs post training, for one shock period.
        Reproduces paper Figures 9, 15, 21.
        """
        import matplotlib.pyplot as plt
        from evaluation._plot_style import apply_style, save_figure

        apply_style()
        r = self.network_indicators_report(shock_name)

        if "density_post" not in r:
            raise FileNotFoundError(
                f"No post-training data for '{shock_name}' — cannot plot comparison."
            )

        fig, ax = plt.subplots(figsize=figsize)
        labels  = ["Density", "Clustering\nCoefficient"]
        pre_vals  = [r["density_pre"],  r["clustering_pre"]]
        post_vals = [r["density_post"], r["clustering_post"]]

        x = np.arange(len(labels))
        w = 0.32

        bars1 = ax.bar(x - w/2, pre_vals,  width=w, color="#58a6ff",
                       label="Pre-training",  alpha=0.88)
        bars2 = ax.bar(x + w/2, post_vals, width=w, color="#f0883e",
                       label="Post-training", alpha=0.88)

        for bars in (bars1, bars2):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Value")
        ax.set_ylim(0, max(pre_vals + post_vals) * 1.25)
        ax.set_title(f"Network Indicators — {shock_name}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.4)
        fig.tight_layout()

        if save:
            fname = filename or f"network_indicators_{self._slug(shock_name)}.png"
            save_figure(fig, fname)
        return fig, ax

    def plot_trend(
        self,
        shock_names : List[str],
        stage       : str = "pre_training",
        figsize     : Tuple = (13, 5),
        save        : bool  = True,
        filename    : str   = "density_clustering_trend.png",
    ):
        """
        Line plot of density and clustering coefficient across the full
        chronological shock history. Reveals whether the market has become
        structurally more/less interconnected over the 2005-present period.
        """
        import matplotlib.pyplot as plt
        from evaluation._plot_style import apply_style, save_figure

        apply_style()
        df = self.density_trend(shock_names, stage=stage)

        if df.empty:
            logger.warning("No data for density trend plot.")
            return None, None

        fig, ax1 = plt.subplots(figsize=figsize)
        x = np.arange(len(df))

        ax1.plot(x, df["density"], color="#58a6ff", marker="o", markersize=4,
                 linewidth=1.4, label="Density")
        ax1.set_ylabel("Density", color="#58a6ff")
        ax1.tick_params(axis="y", labelcolor="#58a6ff")

        ax2 = ax1.twinx()
        ax2.plot(x, df["clustering"], color="#f0883e", marker="s", markersize=4,
                 linewidth=1.4, label="Clustering Coefficient")
        ax2.set_ylabel("Clustering Coefficient", color="#f0883e")
        ax2.tick_params(axis="y", labelcolor="#f0883e")

        ax1.set_xticks(x)
        ax1.set_xticklabels(df["shock_name"], rotation=60, ha="right", fontsize=7)
        ax1.set_title(
            f"Density & Clustering Coefficient Across Shock History ({stage})",
            fontsize=11,
        )
        ax1.grid(True, alpha=0.3)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

        fig.tight_layout()

        if save:
            save_figure(fig, filename)
        return fig, (ax1, ax2)

    # ─────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────

    @staticmethod
    def _slug(name: str) -> str:
        return name.lower().replace(" ", "_").replace("(", "").replace(")", "")[:60]


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, shutil
    logging.basicConfig(level="INFO", format=CFG.LOG_FORMAT)
    np.random.seed(42)

    from config.nifty50_universe import get_tickers
    tickers = get_tickers()
    N       = len(tickers)

    tmp = Path(tempfile.mkdtemp())
    import graph.adjacency as adj_mod
    orig_pre, orig_post, orig_tk = (
        adj_mod.PRE_TRAINING_DIR, adj_mod.POST_TRAINING_DIR, adj_mod.TICKER_FILE
    )
    adj_mod.PRE_TRAINING_DIR  = tmp / "pre_training"
    adj_mod.POST_TRAINING_DIR = tmp / "post_training"
    adj_mod.TICKER_FILE       = tmp / "tickers.json"

    try:
        store = AdjacencyStore()

        # Simulate the paper's typical pattern: density↓, clustering↑ post-training
        shock_names = ["Shock_A", "Shock_B", "Shock_C"]
        for i, name in enumerate(shock_names):
            pre_density  = 0.55 - i * 0.05
            post_density = pre_density - 0.05

            pre_mat  = (np.random.rand(N, N) < pre_density).astype(np.float32)
            np.fill_diagonal(pre_mat, 1.0)
            # Post matrix: fewer random edges but more triangles (force via block structure)
            post_mat = (np.random.rand(N, N) < post_density).astype(np.float32)
            # Add clustering: connect first 15 nodes densely (simulates a tight cluster)
            post_mat[:15, :15] = (np.random.rand(15, 15) < 0.7).astype(np.float32)
            np.fill_diagonal(post_mat, 1.0)

            store.save_all({name: pre_mat},  tickers, stage="pre_training")
            store.save_all({name: post_mat}, tickers, stage="post_training")

        analyzer = DensityAnalyzer(store)

        # network_indicators_report
        r = analyzer.network_indicators_report("Shock_A")
        assert "density_delta" in r
        print(f"network_indicators_report OK")
        print(f"  Narrative: {r['interpretation'][:100]}...")

        # all_shocks_report
        all_rpt = analyzer.all_shocks_report(shock_names)
        assert len(all_rpt) == 3
        print(f"\nall_shocks_report  OK : {len(all_rpt)} shocks")
        print(all_rpt[["shock_name", "density_pre", "density_post",
                       "clustering_pre", "clustering_post", "tightening"]].to_string(index=False))

        # density_trend
        trend = analyzer.density_trend(shock_names, stage="pre_training")
        assert len(trend) == 3
        print(f"\ndensity_trend      OK : {trend.shape}")

        # Plotting
        import config.settings as cfg_mod
        orig_figs = cfg_mod.FIGURES_DIR
        cfg_mod.FIGURES_DIR = tmp / "figs"
        try:
            fig1, ax1 = analyzer.plot_density_clustering("Shock_A", save=True)
            fig2, ax2 = analyzer.plot_trend(shock_names, save=True)
            saved = list((tmp / "figs").glob("*.png"))
            print(f"\nplotting           OK : {len(saved)} figures saved")
            assert len(saved) == 2
        finally:
            cfg_mod.FIGURES_DIR = orig_figs

        print("\nAll density.py tests PASSED.")
    finally:
        adj_mod.PRE_TRAINING_DIR  = orig_pre
        adj_mod.POST_TRAINING_DIR = orig_post
        adj_mod.TICKER_FILE       = orig_tk
        shutil.rmtree(tmp)