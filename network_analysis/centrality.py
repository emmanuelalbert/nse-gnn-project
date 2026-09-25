"""
network_analysis/centrality.py
--------------------------------
Paper-style centrality reporting layer built on top of graph.adjacency.MatrixMetrics.

Paper reference: Section III-E (Eq. 13–15, 18) and Section IV-C
    "We analyze network structures before and after model training and
     evaluate company characteristics such as linkage degree and centrality
     measures."

    Eq. 13 — Degree of Connection
    Eq. 14 — Closeness Centrality
    Eq. 15 — Betweenness Centrality
    Eq. 18 — Degree Centrality

graph/adjacency.py::MatrixMetrics already implements the raw per-matrix
computation of each metric. This module is the ANALYTICAL layer on top:
it runs MatrixMetrics across every shock period (pre vs. post training),
ranks tickers, attaches sector labels, and reproduces the paper's narrative
structure (e.g. paper Section IV-C-1-b:
    "Betweenness Centrality: Initially, stocks like ECL, PPG, and CAT
     showed higher betweenness centrality... Post-training, this measure
     decreases significantly for almost all stocks.").

Responsibilities:
    1. top_n_by_metric()        — paper-style "Initially, stocks like X, Y, Z
                                   showed higher [metric]" tables.
    2. pre_post_shift_report()  — per-ticker delta table for one shock period.
    3. cross_shock_summary()    — how each ticker's centrality rank changes
                                   across ALL shock periods (who is consistently
                                   central vs. who becomes central only during
                                   crises).
    4. sector_centrality()      — average centrality per sector, paper's
                                   sector-level narrative.
    5. plot_centrality_shift()  — paper Figures 6-8 / 12-14 / 18-20 equivalent
                                   (before/after bar charts).

Usage:
    from network_analysis.centrality import CentralityAnalyzer
    analyzer = CentralityAnalyzer(adjacency_store)
    report   = analyzer.pre_post_shift_report("Global Financial Crisis")
    top      = analyzer.top_n_by_metric("Global Financial Crisis", "betweenness", stage="pre_training")
    cross    = analyzer.cross_shock_summary(shock_names)
    analyzer.plot_centrality_shift("Global Financial Crisis", metric="degree", save=True)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as CFG
from config.nifty50_universe import get_sector_map, SECTOR_COLOURS
from graph.adjacency import AdjacencyStore, MatrixMetrics

logger = logging.getLogger(__name__)

CENTRALITY_METRICS = ["degree", "degree_centrality", "closeness", "betweenness"]


# ─────────────────────────────────────────────────────────────────────────────
# CENTRALITY ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

class CentralityAnalyzer:
    """
    Paper-style centrality analysis across shock periods, pre- and
    post-training.

    Args:
        store : AdjacencyStore instance with saved pre/post matrices.
    """

    def __init__(self, store: Optional[AdjacencyStore] = None):
        self.store      = store or AdjacencyStore()
        self.sector_map = get_sector_map()

    # ─────────────────────────────────────────
    # TOP-N RANKING  (paper narrative style)
    # ─────────────────────────────────────────

    def top_n_by_metric(
        self,
        shock_name : str,
        metric     : str = "degree",
        stage      : str = "pre_training",
        top_n      : int = 10,
    ) -> pd.DataFrame:
        """
        Return the top_n tickers ranked by a centrality metric for one
        shock period and training stage.

        Reproduces tables like paper Section IV-C-1-b:
            "Initially, stocks like ECL, PPG, and CAT showed higher
             betweenness centrality, implying their role as bridges
             in the network."

        Args:
            shock_name : Shock period name.
            metric     : One of CENTRALITY_METRICS.
            stage      : "pre_training" | "post_training".
            top_n      : Number of top tickers to return.

        Returns:
            DataFrame: ticker, sector, {metric}, rank.
        """
        self._validate_metric(metric)
        matrix, tickers = self.store.load(shock_name, stage=stage)
        m = MatrixMetrics(matrix, tickers)
        values = self._get_metric_series(m, metric)

        df = pd.DataFrame({
            "ticker" : values.index,
            "sector" : [self.sector_map.get(t, "Unknown") for t in values.index],
            metric   : values.values,
        }).sort_values(metric, ascending=False).reset_index(drop=True)
        df.index += 1
        df.index.name = "rank"

        top = df.head(top_n)
        logger.info(
            f"Top {top_n} by {metric} [{stage}] for '{shock_name}': "
            f"{', '.join(top['ticker'].tolist()[:5])}..."
        )
        return top

    # ─────────────────────────────────────────
    # PRE/POST SHIFT REPORT  (single shock period)
    # ─────────────────────────────────────────

    def pre_post_shift_report(
        self,
        shock_name : str,
    ) -> pd.DataFrame:
        """
        Full pre-vs-post centrality comparison for one shock period.
        This is the core analytical output mirroring paper Figures 6-8,
        12-14, 18-20.

        Returns:
            DataFrame indexed by ticker with columns:
                sector,
                {metric}_pre, {metric}_post, {metric}_delta  (for each metric in CENTRALITY_METRICS)
            Sorted by degree_delta descending (biggest centrality gainers first).
        """
        comparison = self.store.compare(shock_name)

        if comparison.get("post_matrix") is None:
            raise FileNotFoundError(
                f"No post-training matrix for '{shock_name}'. "
                "Run model inference and save the predicted adjacency first."
            )

        delta_df = MatrixMetrics.compare(
            comparison["pre_matrix"],
            comparison["post_matrix"],
            comparison["tickers"],
        )

        delta_df.insert(0, "sector",
                        [self.sector_map.get(t, "Unknown") for t in delta_df.index])

        delta_df = delta_df.sort_values("degree_delta", ascending=False)

        logger.info(
            f"Pre/post shift report '{shock_name}': "
            f"{comparison['added_edges']} edges added, "
            f"{comparison['removed_edges']} removed, "
            f"{comparison['stable_edges']} stable"
        )
        return delta_df

    # ─────────────────────────────────────────
    # CROSS-SHOCK SUMMARY
    # ─────────────────────────────────────────

    def cross_shock_summary(
        self,
        shock_names : List[str],
        metric      : str = "betweenness",
        stage       : str = "pre_training",
    ) -> pd.DataFrame:
        """
        Track how each ticker's centrality RANK changes across multiple
        shock periods. Identifies tickers that are:
          - Consistently central (low rank variance across shocks)
          - Crisis-specific hubs (high rank only during certain shocks)

        Args:
            shock_names : List of shock period names to compare.
            metric      : Centrality metric to track.
            stage       : "pre_training" | "post_training".

        Returns:
            DataFrame: ticker, sector, {shock_1}_rank, {shock_2}_rank, ...,
                      mean_rank, rank_std (lower std = more consistently central).
        """
        self._validate_metric(metric)
        rank_data: Dict[str, Dict] = {}

        for shock_name in shock_names:
            try:
                matrix, tickers = self.store.load(shock_name, stage=stage)
            except FileNotFoundError:
                logger.warning(f"Skipping '{shock_name}' — no {stage} matrix saved.")
                continue

            m      = MatrixMetrics(matrix, tickers)
            values = self._get_metric_series(m, metric)
            ranks  = values.rank(ascending=False, method="min")

            for ticker, rank in ranks.items():
                rank_data.setdefault(ticker, {})[shock_name] = int(rank)

        if not rank_data:
            logger.warning("No shocks had saved matrices — returning empty summary.")
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(rank_data, orient="index")
        df.insert(0, "sector", [self.sector_map.get(t, "Unknown") for t in df.index])

        rank_cols = [c for c in df.columns if c != "sector"]
        df["mean_rank"] = df[rank_cols].mean(axis=1, skipna=True).round(2)
        df["rank_std"]  = df[rank_cols].std(axis=1, skipna=True, ddof=0).round(2)

        df = df.sort_values("mean_rank").reset_index().rename(columns={"index": "ticker"})

        logger.info(
            f"Cross-shock summary ({metric}, {stage}) across {len(shock_names)} shocks: "
            f"most consistently central = {df.iloc[0]['ticker']}"
        )
        return df

    # ─────────────────────────────────────────
    # SECTOR-LEVEL CENTRALITY
    # ─────────────────────────────────────────

    def sector_centrality(
        self,
        shock_name : str,
        stage      : str = "pre_training",
    ) -> pd.DataFrame:
        """
        Aggregate centrality metrics to the sector level for one shock period.

        Reproduces the paper's sector-level narrative (e.g. Section IV-C-3:
        "Financials (JPM, WFC, and BAC) and Utilities ... showed negative
         average returns, highlighting their vulnerability").

        Returns:
            DataFrame: sector, n_tickers, mean_degree, mean_closeness,
                      mean_betweenness, mean_degree_centrality.
        """
        matrix, tickers = self.store.load(shock_name, stage=stage)
        m  = MatrixMetrics(matrix, tickers)
        df = m.all_metrics()
        df["sector"] = [self.sector_map.get(t, "Unknown") for t in df.index]

        sector_df = (
            df.groupby("sector")[CENTRALITY_METRICS]
            .mean()
            .add_prefix("mean_")
            .round(4)
        )
        sector_df["n_tickers"] = df.groupby("sector").size()
        sector_df = sector_df.sort_values("mean_betweenness", ascending=False)

        return sector_df.reset_index()

    def sector_centrality_shift(
        self,
        shock_name : str,
    ) -> pd.DataFrame:
        """
        Sector-level pre vs. post training centrality comparison.
        Aggregates the ticker-level pre_post_shift_report() to sector means.
        """
        ticker_df = self.pre_post_shift_report(shock_name)

        delta_cols = [c for c in ticker_df.columns if c.endswith("_delta")]
        sector_df  = ticker_df.groupby("sector")[delta_cols].mean().round(4)
        sector_df["n_tickers"] = ticker_df.groupby("sector").size()

        return sector_df.sort_values("degree_delta", ascending=False).reset_index()

    # ─────────────────────────────────────────
    # MULTI-SHOCK BATCH REPORT
    # ─────────────────────────────────────────

    def all_target_shocks_report(
        self,
        target_shock_names : Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Run pre_post_shift_report() for all target shock periods at once.
        Convenience method for the comparative analysis notebook.

        Returns:
            {shock_name: shift_report_df}
        """
        names = target_shock_names or [t["name"] for t in CFG.TARGET_SHOCK_PERIODS]
        reports = {}
        for name in names:
            try:
                reports[name] = self.pre_post_shift_report(name)
            except FileNotFoundError as exc:
                logger.warning(f"Skipping '{name}': {exc}")
        return reports

    # ─────────────────────────────────────────
    # PLOTTING
    # ─────────────────────────────────────────

    def plot_centrality_shift(
        self,
        shock_name : str,
        metric     : str = "degree",
        top_n      : int = 15,
        figsize    : Tuple = (10, 7),
        save       : bool  = True,
        filename   : Optional[str] = None,
    ):
        """
        Paired bar chart: pre-training vs post-training centrality for the
        top_n tickers by |delta|. Reproduces paper Figures 6, 8, 12, 14, 18, 20.
        """
        import matplotlib.pyplot as plt
        from evaluation._plot_style import apply_style, save_figure

        self._validate_metric(metric)
        apply_style()

        report = self.pre_post_shift_report(shock_name)
        report = report.reindex(
            report[f"{metric}_delta"].abs().sort_values(ascending=False).index
        ).head(top_n)

        fig, ax = plt.subplots(figsize=figsize)
        y_pos   = np.arange(len(report))
        bar_h   = 0.38

        ax.barh(y_pos + bar_h/2, report[f"{metric}_pre"],  height=bar_h,
                color="#58a6ff", label="Pre-training", alpha=0.85)
        ax.barh(y_pos - bar_h/2, report[f"{metric}_post"], height=bar_h,
                color="#f0883e", label="Post-training", alpha=0.85)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(report.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_title(
            f"{metric.replace('_',' ').title()} — Pre vs Post Training\n{shock_name}",
            fontsize=11,
        )
        ax.legend(fontsize=8)
        ax.grid(True, axis="x", alpha=0.4)
        fig.tight_layout()

        if save:
            fname = filename or f"centrality_shift_{metric}_{self._slug(shock_name)}.png"
            save_figure(fig, fname)
        return fig, ax

    def plot_sector_centrality(
        self,
        shock_name : str,
        stage      : str = "pre_training",
        figsize    : Tuple = (9, 6),
        save       : bool  = True,
        filename   : Optional[str] = None,
    ):
        """Bar chart of mean betweenness centrality by sector for one shock/stage."""
        import matplotlib.pyplot as plt
        from evaluation._plot_style import apply_style, save_figure

        apply_style()
        df = self.sector_centrality(shock_name, stage=stage)

        fig, ax = plt.subplots(figsize=figsize)
        colours = [SECTOR_COLOURS.get(s, "#8b949e") for s in df["sector"]]
        ax.barh(df["sector"], df["mean_betweenness"], color=colours, alpha=0.88)
        ax.set_xlabel("Mean Betweenness Centrality")
        ax.set_title(f"Sector Centrality — {shock_name} ({stage})", fontsize=11)
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.4)
        fig.tight_layout()

        if save:
            fname = filename or f"sector_centrality_{stage}_{self._slug(shock_name)}.png"
            save_figure(fig, fname)
        return fig, ax

    # ─────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────

    def _get_metric_series(self, m: MatrixMetrics, metric: str) -> pd.Series:
        if metric == "degree":
            return m.degree()
        elif metric == "degree_centrality":
            return m.degree_centrality()
        elif metric == "closeness":
            return m.closeness()
        elif metric == "betweenness":
            return m.betweenness()
        raise ValueError(f"Unknown metric '{metric}'.")

    def _validate_metric(self, metric: str) -> None:
        if metric not in CENTRALITY_METRICS:
            raise ValueError(
                f"metric must be one of {CENTRALITY_METRICS}, got '{metric}'."
            )

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

    pre_mat  = (np.random.rand(N, N) > 0.65).astype(np.float32)
    np.fill_diagonal(pre_mat, 1.0)
    post_mat = (np.random.rand(N, N) > 0.55).astype(np.float32)
    np.fill_diagonal(post_mat, 1.0)

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
        store.save_all({"GFC": pre_mat},  tickers, stage="pre_training")
        store.save_all({"GFC": post_mat}, tickers, stage="post_training")

        analyzer = CentralityAnalyzer(store)

        # top_n_by_metric
        top = analyzer.top_n_by_metric("GFC", metric="betweenness", stage="pre_training", top_n=5)
        assert len(top) == 5
        print(f"top_n_by_metric        OK : top ticker = {top.iloc[0]['ticker']}")

        # pre_post_shift_report
        shift = analyzer.pre_post_shift_report("GFC")
        assert "degree_delta" in shift.columns
        assert len(shift) == N
        print(f"pre_post_shift_report  OK : {shift.shape}")

        # cross_shock_summary (only 1 shock available, but test the mechanics)
        store.save_all({"GFC2": pre_mat}, tickers, stage="pre_training")
        cross = analyzer.cross_shock_summary(["GFC", "GFC2"], metric="degree")
        assert "mean_rank" in cross.columns
        print(f"cross_shock_summary    OK : {cross.shape}")

        # sector_centrality
        sec = analyzer.sector_centrality("GFC", stage="pre_training")
        assert "mean_betweenness" in sec.columns
        print(f"sector_centrality      OK : {len(sec)} sectors")

        # sector_centrality_shift
        sec_shift = analyzer.sector_centrality_shift("GFC")
        assert "degree_delta" in sec_shift.columns
        print(f"sector_centrality_shift OK : {len(sec_shift)} sectors")

        # all_target_shocks_report (GFC matches target name in default settings? Use custom)
        reports = analyzer.all_target_shocks_report(target_shock_names=["GFC"])
        assert "GFC" in reports
        print(f"all_target_shocks_report OK : {list(reports.keys())}")

        # Plotting
        import config.settings as cfg_mod
        orig_figs = cfg_mod.FIGURES_DIR
        cfg_mod.FIGURES_DIR = tmp / "figs"
        try:
            fig1, ax1 = analyzer.plot_centrality_shift("GFC", metric="degree", save=True)
            fig2, ax2 = analyzer.plot_sector_centrality("GFC", save=True)
            saved = list((tmp / "figs").glob("*.png"))
            print(f"plot_centrality_shift  OK : {len(saved)} figures saved")
            assert len(saved) == 2
        finally:
            cfg_mod.FIGURES_DIR = orig_figs

        print("\nAll centrality.py tests PASSED.")
    finally:
        adj_mod.PRE_TRAINING_DIR  = orig_pre
        adj_mod.POST_TRAINING_DIR = orig_post
        adj_mod.TICKER_FILE       = orig_tk
        shutil.rmtree(tmp)