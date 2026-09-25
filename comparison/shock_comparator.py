"""
comparison/shock_comparator.py
---------------------------------
Top-level synthesis layer: compares pre/post-training network structure
across MULTIPLE shock periods, reproducing the paper's Section IV-D
("Comparative Analysis Across Shock Periods").

Paper reference: Section IV-D

    "Our comparative analysis of the 2008 Financial Crisis, the 2016 U.S.
     Presidential Elections, and the onset of the COVID-19 pandemic offers
     a detailed understanding of how different types of market shocks
     propagate across sectors. Each event's unique characteristics —
     whether financial, political, or health-related — significantly
     influenced market behavior and dynamics."

    "The systemic nature of each shock is evident in the increased
     interconnectedness of the market, as shown by the after-training
     network analysis."

This module sits ABOVE network_analysis/ — it does not recompute centrality,
density, or shock metrics; it orchestrates CentralityAnalyzer, DensityAnalyzer
and AdjacencyStore.compare() across the full set of target shock periods and
produces the cross-period synthesis tables and narrative text that the
paper's discussion sections are built from:

    1. network_diff_summary()      — one row per shock: edges added/removed/
                                      stable, density delta, clustering delta
                                      (paper Section IV-A, "Factors Influencing
                                      Market Shock Propagation")
    2. interconnectedness_ranking() — which shock produced the MOST
                                      interconnected post-training network
                                      (paper: "systemic nature of each shock")
    3. propagation_mechanism_report() — classifies each shock's propagation
                                      pattern using the paper's own taxonomy
                                      (Direct Exposure / Indirect Exposure /
                                      Market Sentiment / Policy Response,
                                      Section IV-A-3)
    4. centrality_persistence()    — which tickers gained centrality
                                      consistently across ALL shocks vs.
                                      shock-specific hubs (synthesises
                                      CentralityAnalyzer.cross_shock_summary())
    5. generate_comparative_narrative() — auto-generated prose paragraph
                                      mirroring paper Section IV-D's
                                      structure, for each shock period.

Usage:
    from comparison.shock_comparator import ShockComparator
    comparator = ShockComparator(adjacency_store)
    summary    = comparator.network_diff_summary(shock_names)
    ranking    = comparator.interconnectedness_ranking(shock_names)
    narrative  = comparator.generate_comparative_narrative(shock_names)
    comparator.plot_network_diff_summary(shock_names, save=True)
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
from network_analysis.centrality import CentralityAnalyzer
from network_analysis.density import DensityAnalyzer
from shock_detection.detector import ShockPeriod
from shock_detection.shock_registry import ShockRegistry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PROPAGATION MECHANISM TAXONOMY  (paper Section IV-A-3)
# ─────────────────────────────────────────────────────────────────────────────

# The paper identifies four qualitative mechanisms of shock propagation.
# We map them to quantitative signatures computable from the pre/post
# network comparison, so each shock period can be auto-classified rather
# than manually labelled.
PROPAGATION_MECHANISMS = {
    "direct_exposure": (
        "Direct Exposure — firms directly linked to the shock's origin are "
        "first impacted. Signature: high pre-training degree concentrated "
        "in a small number of stocks/sectors."
    ),
    "indirect_exposure": (
        "Indirect Exposure — companies with financial/operational ties to "
        "directly-impacted industries experience secondary effects. "
        "Signature: post-training degree increase concentrated in stocks "
        "NOT in the pre-training top-decile."
    ),
    "market_sentiment": (
        "Market Sentiment — diminished investor confidence triggers broad "
        "sell-offs beyond initially affected sectors. Signature: large "
        "increase in network density and clustering post-training, spread "
        "broadly across sectors rather than concentrated."
    ),
    "policy_response": (
        "Policy Response — government/central bank actions shape shock "
        "severity and duration. Signature: post-training betweenness "
        "DECREASE (paper's consistent finding: a shift from a few pivotal "
        "stocks to more uniformly distributed influence)."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# SHOCK COMPARATOR
# ─────────────────────────────────────────────────────────────────────────────

class ShockComparator:
    """
    Cross-shock-period network comparison and narrative synthesis.

    Args:
        store    : AdjacencyStore with saved pre/post matrices for all
                   shocks to be compared.
        registry : Optional ShockRegistry, used to pull shock metadata
                   (avg_return, sigma_breach, type) for richer narratives.
    """

    def __init__(
        self,
        store    : Optional[AdjacencyStore]  = None,
        registry : Optional[ShockRegistry]   = None,
    ):
        self.store      = store or AdjacencyStore()
        self.registry   = registry or ShockRegistry()
        self.centrality = CentralityAnalyzer(self.store)
        self.density    = DensityAnalyzer(self.store)

    # ─────────────────────────────────────────
    # 1. NETWORK DIFF SUMMARY
    # ─────────────────────────────────────────

    def network_diff_summary(
        self,
        shock_names : List[str],
    ) -> pd.DataFrame:
        """
        One row per shock period summarising the pre→post training network
        transformation: edges added/removed/stable, density delta, clustering
        delta, and the paper's "tightening" classification.

        Returns:
            DataFrame: shock_name, n_edges_pre, n_edges_post,
                      added_edges, removed_edges, stable_edges,
                      density_pre, density_post, density_delta,
                      clustering_pre, clustering_post, clustering_delta,
                      tightening (bool)
        """
        rows = []
        for name in shock_names:
            try:
                comparison = self.store.compare(name)
            except FileNotFoundError as exc:
                logger.warning(f"Skipping '{name}': {exc}")
                continue

            if comparison.get("post_matrix") is None:
                logger.warning(f"Skipping '{name}': no post-training matrix.")
                continue

            pre_m  = comparison["pre_metrics"]
            post_m = comparison["post_metrics"]
            _, clust_pre  = pre_m.clustering()
            _, clust_post = post_m.clustering()

            density_pre  = pre_m.density()
            density_post = post_m.density()

            rows.append({
                "shock_name"       : name,
                "n_edges_pre"      : int(comparison["pre_matrix"].sum() - np.trace(comparison["pre_matrix"])),
                "n_edges_post"     : int(comparison["post_matrix"].sum() - np.trace(comparison["post_matrix"])),
                "added_edges"      : comparison["added_edges"],
                "removed_edges"    : comparison["removed_edges"],
                "stable_edges"     : comparison["stable_edges"],
                "density_pre"      : round(density_pre, 4),
                "density_post"     : round(density_post, 4),
                "density_delta"    : round(density_post - density_pre, 4),
                "clustering_pre"   : round(clust_pre, 4),
                "clustering_post"  : round(clust_post, 4),
                "clustering_delta" : round(clust_post - clust_pre, 4),
                "tightening"       : bool(clust_post > clust_pre and density_post <= density_pre),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            n_tight = df["tightening"].sum()
            logger.info(
                f"network_diff_summary: {len(df)} shocks compared, "
                f"{n_tight}/{len(df)} show paper's typical tightening pattern."
            )
        return df

    # ─────────────────────────────────────────
    # 2. INTERCONNECTEDNESS RANKING
    # ─────────────────────────────────────────

    def interconnectedness_ranking(
        self,
        shock_names : List[str],
    ) -> pd.DataFrame:
        """
        Rank shock periods by how much MORE interconnected the post-training
        network became relative to pre-training — i.e. which crisis produced
        the most "systemic" pattern of causal propagation.

        Paper: "The systemic nature of each shock is evident in the
        increased interconnectedness of the market, as shown by the
        after-training network analysis."

        Returns:
            DataFrame: shock_name, density_delta, clustering_delta,
                      edges_added, interconnectedness_score (composite),
                      sorted descending by interconnectedness_score.
        """
        diff_df = self.network_diff_summary(shock_names)
        if diff_df.empty:
            return diff_df

        # Composite score: equally-weighted z-scores of density delta,
        # clustering delta, and edges added — captures "how much MORE
        # interconnected" along all three signals at once.
        def _z(s: pd.Series) -> pd.Series:
            std = s.std(ddof=0)
            return (s - s.mean()) / std if std > 0 else pd.Series(0.0, index=s.index)

        score = (
            _z(diff_df["density_delta"]) +
            _z(diff_df["clustering_delta"]) +
            _z(diff_df["added_edges"].astype(float))
        ) / 3.0

        result = diff_df[["shock_name", "density_delta", "clustering_delta", "added_edges"]].copy()
        result["interconnectedness_score"] = score.round(4)
        result = result.sort_values("interconnectedness_score", ascending=False).reset_index(drop=True)
        result.index += 1
        result.index.name = "rank"

        if not result.empty:
            logger.info(
                f"Most systemic shock: '{result.iloc[0]['shock_name']}' "
                f"(score={result.iloc[0]['interconnectedness_score']:.3f})"
            )
        return result

    # ─────────────────────────────────────────
    # 3. PROPAGATION MECHANISM CLASSIFICATION
    # ─────────────────────────────────────────

    def propagation_mechanism_report(
        self,
        shock_name : str,
    ) -> Dict:
        """
        Classify the dominant propagation mechanism for one shock period
        using the paper's own taxonomy (Section IV-A-3), inferred from
        quantitative signatures in the pre/post network comparison.

        Returns:
            {
                "shock_name"          : str,
                "dominant_mechanism"  : str — key into PROPAGATION_MECHANISMS,
                "mechanism_description": str,
                "evidence"            : Dict — the quantitative signals used,
            }
        """
        diff_row = self.network_diff_summary([shock_name])
        if diff_row.empty:
            raise FileNotFoundError(f"No comparable data for '{shock_name}'.")
        diff_row = diff_row.iloc[0]

        shift = self.centrality.pre_post_shift_report(shock_name)
        betweenness_decreased = (shift["betweenness_delta"] < 0).mean() > 0.5

        # Concentration of pre-training degree (Gini-like: top-decile share)
        pre_degree = shift["degree_pre"].sort_values(ascending=False)
        n_top = max(1, len(pre_degree) // 10)
        top_decile_share = (
            pre_degree.head(n_top).sum() / pre_degree.sum()
            if pre_degree.sum() > 0 else 0.0
        )

        evidence = {
            "density_delta"           : float(diff_row["density_delta"]),
            "clustering_delta"        : float(diff_row["clustering_delta"]),
            "betweenness_decreased_pct": round(float((shift["betweenness_delta"] < 0).mean()), 3),
            "pre_training_top_decile_degree_share": round(float(top_decile_share), 3),
        }

        # Decision logic, ordered by the paper's typical pattern strength:
        if betweenness_decreased and diff_row["tightening"]:
            mechanism = "policy_response"
        elif diff_row["clustering_delta"] > 0.10:   # large broad-based tightening
            mechanism = "market_sentiment"
        elif top_decile_share > 0.40:                 # concentrated initial exposure
            mechanism = "direct_exposure"
        else:
            mechanism = "indirect_exposure"

        result = {
            "shock_name"           : shock_name,
            "dominant_mechanism"   : mechanism,
            "mechanism_description": PROPAGATION_MECHANISMS[mechanism],
            "evidence"             : evidence,
        }
        logger.info(
            f"Propagation mechanism '{shock_name}': {mechanism} "
            f"(betweenness↓={evidence['betweenness_decreased_pct']:.0%}, "
            f"clustering Δ={evidence['clustering_delta']:.3f})"
        )
        return result

    def propagation_mechanism_summary(
        self,
        shock_names : List[str],
    ) -> pd.DataFrame:
        """Batch propagation_mechanism_report() across multiple shocks."""
        rows = []
        for name in shock_names:
            try:
                r = self.propagation_mechanism_report(name)
            except FileNotFoundError as exc:
                logger.warning(f"Skipping '{name}': {exc}")
                continue
            row = {"shock_name": r["shock_name"], "dominant_mechanism": r["dominant_mechanism"]}
            row.update(r["evidence"])
            rows.append(row)
        return pd.DataFrame(rows)

    # ─────────────────────────────────────────
    # 4. CENTRALITY PERSISTENCE  (cross-shock hub identification)
    # ─────────────────────────────────────────

    def centrality_persistence(
        self,
        shock_names : List[str],
        metric      : str = "betweenness",
        stage       : str = "post_training",
    ) -> pd.DataFrame:
        """
        Thin wrapper synthesising CentralityAnalyzer.cross_shock_summary()
        into the comparison package's persistence framing: which tickers
        are systemically important across ALL crises vs. which only
        emerge as hubs during specific events.

        Returns:
            DataFrame: ticker, sector, mean_rank, rank_std, persistence_class
                      where persistence_class ∈
                          {"systemic_hub", "crisis_specific", "peripheral"}
        """
        cross = self.centrality.cross_shock_summary(shock_names, metric=metric, stage=stage)
        if cross.empty:
            return cross

        n      = len(cross)
        median_rank = cross["mean_rank"].median()
        std_thresh  = cross["rank_std"].median()

        def _classify(row) -> str:
            if row["mean_rank"] <= n * 0.20 and row["rank_std"] <= std_thresh:
                return "systemic_hub"          # consistently central
            elif row["mean_rank"] <= n * 0.30 and row["rank_std"] > std_thresh:
                return "crisis_specific"        # central in some, not others
            else:
                return "peripheral"

        cross["persistence_class"] = cross.apply(_classify, axis=1)

        counts = cross["persistence_class"].value_counts()
        logger.info(
            f"centrality_persistence ({metric}, {stage}): "
            f"{counts.to_dict()}"
        )
        return cross

    # ─────────────────────────────────────────
    # 5. NARRATIVE GENERATION
    # ─────────────────────────────────────────

    def generate_comparative_narrative(
        self,
        shock_names : List[str],
    ) -> str:
        """
        Auto-generate a paper-style comparative narrative paragraph,
        mirroring the structure of paper Section IV-D.

        Returns:
            Multi-paragraph string summarising the comparative findings
            across all given shock periods.
        """
        diff_df  = self.network_diff_summary(shock_names)
        ranking  = self.interconnectedness_ranking(shock_names)
        mech_df  = self.propagation_mechanism_summary(shock_names)

        if diff_df.empty:
            return "Insufficient data to generate comparative narrative."

        lines = []
        lines.append(
            f"Our comparative analysis of {len(diff_df)} distinct shock periods "
            f"offers a detailed understanding of how different types of market "
            f"shocks propagate across the Nifty 50 market. Each event's unique "
            f"characteristics significantly influenced market behavior and "
            f"network dynamics."
        )

        if not ranking.empty:
            top = ranking.iloc[0]
            lines.append(
                f"\nThe systemic nature of each shock is evident in the increased "
                f"interconnectedness of the market post-training. '{top['shock_name']}' "
                f"produced the most systemic propagation pattern (interconnectedness "
                f"score={top['interconnectedness_score']:.3f}), with a density change "
                f"of {top['density_delta']:+.4f} and clustering change of "
                f"{top['clustering_delta']:+.4f}."
            )

        n_tightening = diff_df["tightening"].sum()
        lines.append(
            f"\n{n_tightening} of {len(diff_df)} shock periods exhibited the "
            f"characteristic 'tightening' pattern — increased clustering "
            f"coefficient alongside stable or decreasing density — consistent "
            f"with markets becoming more cohesive during periods of financial "
            f"distress, where correlations between stocks tend to increase."
        )

        if not mech_df.empty:
            mech_counts = mech_df["dominant_mechanism"].value_counts()
            mech_summary = ", ".join(
                f"{PROPAGATION_MECHANISMS[m].split('—')[0].strip()} ({c})"
                for m, c in mech_counts.items()
            )
            lines.append(
                f"\nClassifying propagation mechanisms across all periods: "
                f"{mech_summary}."
            )

        return "\n".join(lines)

    # ─────────────────────────────────────────
    # PLOTTING
    # ─────────────────────────────────────────

    def plot_network_diff_summary(
        self,
        shock_names : List[str],
        figsize     : Tuple = (12, 6),
        save        : bool  = True,
        filename    : str   = "network_diff_summary.png",
    ):
        """
        Grouped bar chart: density delta and clustering delta per shock
        period, side by side. Visual companion to network_diff_summary().
        """
        import matplotlib.pyplot as plt
        from evaluation._plot_style import apply_style, save_figure

        apply_style()
        df = self.network_diff_summary(shock_names)
        if df.empty:
            logger.warning("No data to plot.")
            return None, None

        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(df))
        w = 0.35

        ax.bar(x - w/2, df["density_delta"],    width=w, color="#58a6ff",
               label="Density Δ", alpha=0.88)
        ax.bar(x + w/2, df["clustering_delta"], width=w, color="#f0883e",
               label="Clustering Δ", alpha=0.88)
        ax.axhline(0, color="#8b949e", linewidth=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(df["shock_name"], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Pre → Post Training Delta")
        ax.set_title("Network Structure Change Across Shock Periods", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.4)
        fig.tight_layout()

        if save:
            save_figure(fig, filename)
        return fig, ax

    def plot_interconnectedness_ranking(
        self,
        shock_names : List[str],
        figsize     : Tuple = (10, 5),
        save        : bool  = True,
        filename    : str   = "interconnectedness_ranking.png",
    ):
        """Horizontal bar chart ranking shocks by interconnectedness score."""
        import matplotlib.pyplot as plt
        from evaluation._plot_style import apply_style, save_figure

        apply_style()
        df = self.interconnectedness_ranking(shock_names)
        if df.empty:
            logger.warning("No data to plot.")
            return None, None

        fig, ax = plt.subplots(figsize=figsize)
        colours = plt.cm.YlOrRd(
            (df["interconnectedness_score"] - df["interconnectedness_score"].min())
            / max(1e-9, df["interconnectedness_score"].max() - df["interconnectedness_score"].min())
        )
        ax.barh(df["shock_name"], df["interconnectedness_score"], color=colours, alpha=0.9)
        ax.invert_yaxis()
        ax.set_xlabel("Interconnectedness Score (composite z-score)")
        ax.set_title("Shock Periods Ranked by Systemic Interconnectedness", fontsize=11)
        ax.grid(True, axis="x", alpha=0.4)
        fig.tight_layout()

        if save:
            save_figure(fig, filename)
        return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, shutil
    logging.basicConfig(level="INFO", format=CFG.LOG_FORMAT)
    np.random.seed(11)

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
        shock_names = ["Global Financial Crisis", "IL&FS Crisis", "COVID-19 Onset"]

        for i, name in enumerate(shock_names):
            pre_density = 0.50 + i * 0.03
            pre_mat = (np.random.rand(N, N) < pre_density).astype(np.float32)
            np.fill_diagonal(pre_mat, 1.0)

            # Simulate paper's typical pattern with varying intensity per shock
            post_mat = pre_mat.copy()
            # Add clustering structure (denser block among first 20 nodes)
            block = (np.random.rand(20, 20) < (0.6 + i * 0.1)).astype(np.float32)
            post_mat[:20, :20] = np.maximum(post_mat[:20, :20], block)
            # Reduce overall density slightly by removing some random edges
            removal_mask = np.random.rand(N, N) < 0.05
            post_mat[removal_mask] = 0
            np.fill_diagonal(post_mat, 1.0)

            store.save_all({name: pre_mat},  tickers, stage="pre_training")
            store.save_all({name: post_mat}, tickers, stage="post_training")

        comparator = ShockComparator(store)

        # 1. network_diff_summary
        diff_df = comparator.network_diff_summary(shock_names)
        assert len(diff_df) == 3
        print(f"network_diff_summary       OK : {diff_df.shape}")
        print(diff_df[["shock_name", "density_delta", "clustering_delta", "tightening"]].to_string(index=False))

        # 2. interconnectedness_ranking
        ranking = comparator.interconnectedness_ranking(shock_names)
        assert len(ranking) == 3
        print(f"\ninterconnectedness_ranking OK : top={ranking.iloc[0]['shock_name']}")

        # 3. propagation_mechanism_report
        mech = comparator.propagation_mechanism_report(shock_names[0])
        assert mech["dominant_mechanism"] in PROPAGATION_MECHANISMS
        print(f"propagation_mechanism      OK : {mech['dominant_mechanism']}")

        mech_summary = comparator.propagation_mechanism_summary(shock_names)
        assert len(mech_summary) == 3
        print(f"propagation_mechanism_summary OK : {mech_summary.shape}")

        # 4. centrality_persistence
        persist = comparator.centrality_persistence(shock_names, metric="degree")
        assert "persistence_class" in persist.columns
        print(f"centrality_persistence     OK : {persist['persistence_class'].value_counts().to_dict()}")

        # 5. narrative
        narrative = comparator.generate_comparative_narrative(shock_names)
        assert len(narrative) > 100
        print(f"\ngenerate_comparative_narrative OK :\n{narrative}")

        # Plotting
        import config.settings as cfg_mod
        orig_figs = cfg_mod.FIGURES_DIR
        cfg_mod.FIGURES_DIR = tmp / "figs"
        try:
            fig1, ax1 = comparator.plot_network_diff_summary(shock_names, save=True)
            fig2, ax2 = comparator.plot_interconnectedness_ranking(shock_names, save=True)
            saved = list((tmp / "figs").glob("*.png"))
            print(f"\nplotting                   OK : {len(saved)} figures saved")
            assert len(saved) == 2
        finally:
            cfg_mod.FIGURES_DIR = orig_figs

        print("\nAll shock_comparator.py tests PASSED.")
    finally:
        adj_mod.PRE_TRAINING_DIR  = orig_pre
        adj_mod.POST_TRAINING_DIR = orig_post
        adj_mod.TICKER_FILE       = orig_tk
        shutil.rmtree(tmp)