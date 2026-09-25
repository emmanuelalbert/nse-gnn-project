"""
comparison/sector_impact.py
-------------------------------
Sector-level vulnerability and resilience ranking across shock periods.

Paper reference: Section IV-D-2 ("Sectoral Impact Patterns") and
Section IV-C-4 ("Implications and Strategic Insights")

    "The impact of shocks across different sectors varied significantly
     across the crises studied:
       - 2008 Financial Crisis: This period marked a systemic shock,
         profoundly affecting the financial sector first... The real
         estate sector followed... The automotive sector... also faced
         substantial challenges.
       - 2016 U.S. Presidential Elections: The most affected sectors...
         were healthcare, energy, and finance.
       - COVID-19 Pandemic: The pandemic brought widespread disruptions,
         notably affecting sectors such as Consumer Discretionary and
         Energy. Conversely, sectors like Information Technology showed
         resilience and even growth... Similarly, the Healthcare sector
         demonstrated resilience, driven by heightened demand for
         healthcare services."

    "Sector-Specific Vulnerabilities: The financial sector was the most
     affected during the 2008 crisis, while the technology sector showed
     resilience during the COVID-19 pandemic."

This module sits at the TOP of the analysis stack — it synthesises outputs
from network_analysis.shock_metrics.ShockMetricsAnalyzer (amplitude/average
change per sector) AND comparison.shock_comparator.ShockComparator (network
structure deltas) into a single VULNERABILITY vs RESILIENCE ranking that
mirrors the paper's qualitative discussion (Section IV-C-4, IV-D-2) with a
quantitative score.

Responsibilities:
    1. vulnerability_score()       — composite score per sector per shock,
                                      combining amplitude, average change
                                      magnitude, and centrality increase.
    2. resilience_ranking()        — sectors ranked from most resilient
                                      (low vulnerability, stable centrality)
                                      to most vulnerable.
    3. cross_shock_vulnerability() — does a sector's vulnerability rank
                                      hold across ALL shocks, or is it
                                      shock-specific? (paper: tech resilient
                                      in COVID but not necessarily in GFC)
    4. sector_response_profile()   — full per-sector profile across all
                                      shocks: which crises hit it hardest,
                                      which it weathered well.
    5. generate_vulnerability_narrative() — auto-generated prose mirroring
                                      the paper's Section IV-D-2 structure.

Usage:
    from comparison.sector_impact import SectorImpactAnalyzer
    analyzer = SectorImpactAnalyzer(shock_metrics_analyzer, shock_comparator)
    ranking  = analyzer.resilience_ranking(shocks)
    profile  = analyzer.sector_response_profile("Information Technology", shocks)
    narrative = analyzer.generate_vulnerability_narrative(shocks)
    analyzer.plot_resilience_ranking(shocks, save=True)
    analyzer.plot_sector_heatmap(shocks, save=True)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as CFG
from config.nifty50_universe import get_sectors, SECTOR_COLOURS
from comparison.shock_comparator import ShockComparator
from network_analysis.shock_metrics import ShockMetricsAnalyzer
from shock_detection.detector import ShockPeriod

logger = logging.getLogger(__name__)

# Weights for the composite vulnerability score. Amplitude and average
# change capture PRICE-level distress (paper Eq. 19-20); centrality
# increase captures STRUCTURAL distress (becoming more causally entangled
# with the rest of the market during the shock — paper Section III-E).
VULNERABILITY_WEIGHTS = {
    "amplitude"           : 0.40,
    "avg_change_magnitude": 0.35,
    "centrality_increase" : 0.25,
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTOR IMPACT ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

class SectorImpactAnalyzer:
    """
    Sector-level vulnerability and resilience ranking, synthesising price-level
    shock metrics (amplitude, average change) with network-structure metrics
    (centrality shift) into the paper's qualitative Section IV-D-2 narrative.

    Args:
        shock_metrics : ShockMetricsAnalyzer instance (prices/returns based).
        comparator    : ShockComparator instance (adjacency-matrix based).
                        Optional — if not provided, vulnerability scoring
                        falls back to price-level metrics only.
    """

    def __init__(
        self,
        shock_metrics : ShockMetricsAnalyzer,
        comparator    : Optional[ShockComparator] = None,
    ):
        self.shock_metrics = shock_metrics
        self.comparator    = comparator
        self.sectors        = get_sectors()

    # ─────────────────────────────────────────
    # 1. VULNERABILITY SCORE  (single shock, all sectors)
    # ─────────────────────────────────────────

    def vulnerability_score(
        self,
        shock : ShockPeriod,
    ) -> pd.DataFrame:
        """
        Compute a composite vulnerability score per sector for one shock
        period, combining:
            - amplitude (price-level fluctuation extent, paper Eq. 20)
            - |average change| (price-level directional impact, paper Eq. 19)
            - centrality increase (structural entanglement increase, if
              comparator is available)

        All three components are min-max normalised to [0, 1] across
        sectors before combining, so the composite score is comparable
        across shocks with very different absolute magnitudes (e.g. COVID's
        higher volatility vs. a milder shock).

        Returns:
            DataFrame: sector, amplitude_norm, avg_change_norm,
                      centrality_norm (NaN if comparator unavailable),
                      vulnerability_score (0=most resilient, 1=most vulnerable),
                      sorted descending by vulnerability_score.
        """
        sector_df = self.shock_metrics.sector_report(shock.name, shock)
        sector_df = sector_df.set_index("sector")

        amp_norm = self._minmax(sector_df["mean_amplitude"])
        chg_norm = self._minmax(sector_df["mean_abs_avg_change"])

        result = pd.DataFrame({
            "sector"          : sector_df.index,
            "amplitude_norm"  : amp_norm.values,
            "avg_change_norm" : chg_norm.values,
        }).set_index("sector")

        # Optional structural component
        cent_norm = pd.Series(np.nan, index=result.index)
        if self.comparator is not None:
            try:
                sector_shift = self.comparator.centrality.sector_centrality_shift(shock.name)
                sector_shift = sector_shift.set_index("sector")
                # Use degree_delta as the structural entanglement signal
                cent_norm = self._minmax(
                    sector_shift["degree_delta"].reindex(result.index).fillna(0)
                )
            except FileNotFoundError:
                logger.debug(
                    f"No post-training matrix for '{shock.name}' — "
                    "vulnerability score uses price-level metrics only."
                )

        result["centrality_norm"] = cent_norm

        w = VULNERABILITY_WEIGHTS
        if cent_norm.isna().all():
            # Re-normalise weights over the two available components
            w_amp = w["amplitude"] / (w["amplitude"] + w["avg_change_magnitude"])
            w_chg = w["avg_change_magnitude"] / (w["amplitude"] + w["avg_change_magnitude"])
            result["vulnerability_score"] = (
                w_amp * result["amplitude_norm"] + w_chg * result["avg_change_norm"]
            )
        else:
            result["vulnerability_score"] = (
                w["amplitude"]            * result["amplitude_norm"] +
                w["avg_change_magnitude"] * result["avg_change_norm"] +
                w["centrality_increase"]  * result["centrality_norm"].fillna(0)
            )

        result = result.round(4).sort_values("vulnerability_score", ascending=False)
        logger.info(
            f"Vulnerability score '{shock.name}': "
            f"most vulnerable = {result.index[0]} ({result.iloc[0]['vulnerability_score']:.3f}), "
            f"most resilient = {result.index[-1]} ({result.iloc[-1]['vulnerability_score']:.3f})"
        )
        return result.reset_index()

    # ─────────────────────────────────────────
    # 2. RESILIENCE RANKING  (across multiple shocks)
    # ─────────────────────────────────────────

    def resilience_ranking(
        self,
        shocks : List[ShockPeriod],
    ) -> pd.DataFrame:
        """
        Aggregate vulnerability_score() across multiple shock periods to
        produce an overall resilience ranking — which sectors are
        consistently resilient vs. consistently vulnerable.

        Mirrors the paper's qualitative cross-period synthesis (Section
        IV-D-2) but as a quantitative composite ranking.

        Returns:
            DataFrame: sector, mean_vulnerability, std_vulnerability,
                      n_shocks, resilience_class
                      sorted ascending by mean_vulnerability (most
                      resilient first), where resilience_class ∈
                          {"resilient", "moderate", "vulnerable"}
        """
        all_scores = []
        for shock in shocks:
            try:
                df = self.vulnerability_score(shock)
                df["shock_name"] = shock.name
                all_scores.append(df)
            except Exception as exc:
                logger.warning(f"Skipping '{shock.name}': {exc}")

        if not all_scores:
            return pd.DataFrame()

        combined = pd.concat(all_scores, ignore_index=True)

        summary = (
            combined.groupby("sector")["vulnerability_score"]
            .agg(mean_vulnerability="mean", std_vulnerability="std", n_shocks="count")
            .fillna(0)
            .round(4)
        )

        # Tercile-based classification (relative to the sector set itself)
        q1, q2 = summary["mean_vulnerability"].quantile([1/3, 2/3])

        def _classify(v: float) -> str:
            if v <= q1:
                return "resilient"
            elif v <= q2:
                return "moderate"
            return "vulnerable"

        summary["resilience_class"] = summary["mean_vulnerability"].apply(_classify)
        summary = summary.sort_values("mean_vulnerability").reset_index()

        logger.info(
            f"Resilience ranking across {len(shocks)} shocks: "
            f"most resilient = {summary.iloc[0]['sector']}, "
            f"most vulnerable = {summary.iloc[-1]['sector']}"
        )
        return summary

    # ─────────────────────────────────────────
    # 3. CROSS-SHOCK VULNERABILITY STABILITY
    # ─────────────────────────────────────────

    def cross_shock_vulnerability(
        self,
        shocks : List[ShockPeriod],
    ) -> pd.DataFrame:
        """
        Wide-format table: sector × shock vulnerability scores, plus a
        rank-stability column. Answers: "is a sector's vulnerability
        consistent across crises, or does it vary by shock TYPE?"

        Paper finding this quantifies: "the financial sector was heavily
        impacted in 2008, whereas the technology sector showed resilience
        during COVID-19" — i.e. vulnerability is shock-TYPE-dependent,
        not a fixed sector property.

        Returns:
            DataFrame: sector, {shock_1}_vuln, {shock_2}_vuln, ...,
                      rank_std (low = consistent across shocks,
                      high = shock-type-dependent vulnerability)
        """
        wide_data = {}
        for shock in shocks:
            try:
                df = self.vulnerability_score(shock).set_index("sector")
                wide_data[shock.name] = df["vulnerability_score"]
            except Exception as exc:
                logger.warning(f"Skipping '{shock.name}': {exc}")

        if not wide_data:
            return pd.DataFrame()

        wide = pd.DataFrame(wide_data).round(4)
        rank_df = wide.rank(ascending=False, method="min")
        wide["rank_std"] = rank_df.std(axis=1, ddof=0).round(2)
        wide.columns = [f"{c}_vuln" if c != "rank_std" else c for c in wide.columns]

        wide = wide.sort_values("rank_std", ascending=False).reset_index()
        wide = wide.rename(columns={"index": "sector"})

        logger.info(
            f"Cross-shock vulnerability stability: "
            f"most shock-type-dependent = {wide.iloc[0]['sector']} "
            f"(rank_std={wide.iloc[0]['rank_std']})"
        )
        return wide

    # ─────────────────────────────────────────
    # 4. SECTOR RESPONSE PROFILE  (single sector, all shocks)
    # ─────────────────────────────────────────

    def sector_response_profile(
        self,
        sector : str,
        shocks : List[ShockPeriod],
    ) -> pd.DataFrame:
        """
        Full response profile for ONE sector across ALL given shock periods.
        Shows how a specific sector (e.g. "Information Technology") behaved
        in each crisis individually.

        Returns:
            DataFrame: shock_name, shock_type, vulnerability_score,
                      amplitude_norm, avg_change_norm, centrality_norm,
                      sorted chronologically by shock start date.
        """
        rows = []
        for shock in shocks:
            try:
                df = self.vulnerability_score(shock)
            except Exception as exc:
                logger.warning(f"Skipping '{shock.name}': {exc}")
                continue

            sector_row = df[df["sector"] == sector]
            if sector_row.empty:
                continue

            row = sector_row.iloc[0].to_dict()
            row["shock_name"] = shock.name
            row["shock_type"] = shock.type
            row["shock_start"] = shock.start
            rows.append(row)

        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values("shock_start").reset_index(drop=True)
            cols = ["shock_name", "shock_type", "shock_start",
                    "vulnerability_score", "amplitude_norm",
                    "avg_change_norm", "centrality_norm"]
            result = result[[c for c in cols if c in result.columns]]

        logger.info(f"Sector response profile for '{sector}': {len(result)} shocks.")
        return result

    # ─────────────────────────────────────────
    # 5. NARRATIVE GENERATION
    # ─────────────────────────────────────────

    def generate_vulnerability_narrative(
        self,
        shocks : List[ShockPeriod],
    ) -> str:
        """
        Auto-generate a paper-style narrative paragraph mirroring Section
        IV-D-2 ("Sectoral Impact Patterns") and Section IV-C-4
        ("Implications and Strategic Insights").

        Returns:
            Multi-paragraph string summarising sector vulnerability findings.
        """
        ranking = self.resilience_ranking(shocks)
        cross   = self.cross_shock_vulnerability(shocks)

        if ranking.empty:
            return "Insufficient data to generate sector vulnerability narrative."

        most_resilient  = ranking.iloc[0]
        most_vulnerable = ranking.iloc[-1]

        lines = []
        lines.append(
            f"Across {len(shocks)} shock periods, sector-level vulnerability "
            f"varied significantly. '{most_vulnerable['sector']}' emerged as "
            f"the most consistently vulnerable sector (mean vulnerability "
            f"score={most_vulnerable['mean_vulnerability']:.3f}), while "
            f"'{most_resilient['sector']}' demonstrated the greatest "
            f"resilience (mean vulnerability score="
            f"{most_resilient['mean_vulnerability']:.3f})."
        )

        n_resilient  = (ranking["resilience_class"] == "resilient").sum()
        n_vulnerable = (ranking["resilience_class"] == "vulnerable").sum()
        lines.append(
            f"\nOf {len(ranking)} sectors analysed, {n_resilient} were "
            f"classified as consistently resilient and {n_vulnerable} as "
            f"consistently vulnerable across the shock periods studied."
        )

        if not cross.empty:
            most_variable = cross.iloc[0]
            lines.append(
                f"\nNotably, '{most_variable['sector']}' showed the highest "
                f"variability in vulnerability rank across different shock "
                f"types (rank std={most_variable['rank_std']:.2f}), "
                f"indicating that its risk exposure is highly dependent on "
                f"the NATURE of the shock (financial, policy, or health-related) "
                f"rather than being a fixed sector characteristic — consistent "
                f"with the paper's finding that 'the financial sector was "
                f"heavily impacted in 2008, whereas the technology sector "
                f"showed resilience during COVID-19.'"
            )

        return "\n".join(lines)

    # ─────────────────────────────────────────
    # PLOTTING
    # ─────────────────────────────────────────

    def plot_resilience_ranking(
        self,
        shocks   : List[ShockPeriod],
        figsize  : Tuple = (10, 7),
        save     : bool  = True,
        filename : str   = "sector_resilience_ranking.png",
    ):
        """Horizontal bar chart of mean vulnerability score per sector, with error bars."""
        import matplotlib.pyplot as plt
        from evaluation._plot_style import apply_style, save_figure

        apply_style()
        df = self.resilience_ranking(shocks)
        if df.empty:
            logger.warning("No data to plot.")
            return None, None

        class_colours = {
            "resilient" : "#3fb950",
            "moderate"  : "#f0883e",
            "vulnerable": "#ff7b72",
        }
        colours = [class_colours.get(c, "#8b949e") for c in df["resilience_class"]]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(
            df["sector"], df["mean_vulnerability"],
            xerr=df["std_vulnerability"], color=colours, alpha=0.88,
            error_kw={"ecolor": "#8b949e", "elinewidth": 1, "capsize": 3},
        )
        ax.invert_yaxis()
        ax.set_xlabel("Mean Vulnerability Score (composite, 0=resilient, 1=vulnerable)")
        ax.set_title(
            f"Sector Resilience Ranking Across {len(shocks)} Shock Periods",
            fontsize=11,
        )
        ax.grid(True, axis="x", alpha=0.4)

        import matplotlib.patches as mpatches
        handles = [mpatches.Patch(color=c, label=k.title()) for k, c in class_colours.items()]
        ax.legend(handles=handles, fontsize=8, loc="lower right")

        fig.tight_layout()

        if save:
            save_figure(fig, filename)
        return fig, ax

    def plot_sector_heatmap(
        self,
        shocks   : List[ShockPeriod],
        figsize  : Tuple = (10, 8),
        save     : bool  = True,
        filename : str   = "sector_vulnerability_heatmap.png",
    ):
        """
        Heatmap of vulnerability score: sector × shock period.
        Visual companion to cross_shock_vulnerability() — directly shows
        which crises hit which sectors hardest (paper Section IV-D-2).
        """
        import matplotlib.pyplot as plt
        from evaluation._plot_style import apply_style, save_figure

        apply_style()
        cross = self.cross_shock_vulnerability(shocks)
        if cross.empty:
            logger.warning("No data to plot.")
            return None, None

        vuln_cols = [c for c in cross.columns if c.endswith("_vuln")]
        matrix    = cross.set_index("sector")[vuln_cols]
        matrix.columns = [c.replace("_vuln", "") for c in matrix.columns]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(matrix.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        cbar.set_label("Vulnerability Score", fontsize=8)

        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index, fontsize=8)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5, color="white" if val > 0.55 else "#1a1a1a")

        ax.set_title("Sector Vulnerability Heatmap Across Shock Periods", fontsize=11, pad=10)
        fig.tight_layout()

        if save:
            save_figure(fig, filename)
        return fig, ax

    # ─────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────

    @staticmethod
    def _minmax(s: pd.Series) -> pd.Series:
        """Min-max normalise a Series to [0, 1]. Returns 0.5 for constant series."""
        rng = s.max() - s.min()
        if rng == 0 or pd.isna(rng):
            return pd.Series(0.5, index=s.index)
        return (s - s.min()) / rng


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, shutil
    logging.basicConfig(level="INFO", format=CFG.LOG_FORMAT)
    np.random.seed(21)

    from config.nifty50_universe import get_tickers
    tickers = get_tickers()
    N       = len(tickers)

    # ── Synthetic prices/returns ──
    dates  = pd.bdate_range("2008-01-01", "2021-01-01")
    prices = pd.DataFrame(
        np.cumprod(1 + np.random.normal(0, 0.015, size=(len(dates), N)), axis=0) * 1000,
        index=dates, columns=tickers,
    )
    log_p   = np.log(prices)
    returns = (log_p - log_p.shift(1)).iloc[1:]

    shocks = [
        ShockPeriod("Global Financial Crisis", "2008-09-15", "2008-10-10",
                    25, -0.025, -0.07, 3.0, "global_financial"),
        ShockPeriod("IL&FS Crisis", "2018-09-21", "2018-10-26",
                    35, -0.020, -0.06, 2.8, "domestic_financial"),
        ShockPeriod("COVID-19 Onset", "2020-02-20", "2020-03-23",
                    32, -0.035, -0.09, 4.5, "health_nonfinancial"),
    ]

    sm = ShockMetricsAnalyzer(prices, returns)

    # ── With comparator (full structural + price-level scoring) ──
    tmp = Path(tempfile.mkdtemp())
    import graph.adjacency as adj_mod
    orig_pre, orig_post, orig_tk = (
        adj_mod.PRE_TRAINING_DIR, adj_mod.POST_TRAINING_DIR, adj_mod.TICKER_FILE
    )
    adj_mod.PRE_TRAINING_DIR  = tmp / "pre_training"
    adj_mod.POST_TRAINING_DIR = tmp / "post_training"
    adj_mod.TICKER_FILE       = tmp / "tickers.json"

    try:
        from graph.adjacency import AdjacencyStore
        store = AdjacencyStore()
        for shock in shocks:
            pre_mat  = (np.random.rand(N, N) > 0.60).astype(np.float32); np.fill_diagonal(pre_mat, 1.0)
            post_mat = (np.random.rand(N, N) > 0.50).astype(np.float32); np.fill_diagonal(post_mat, 1.0)
            store.save_all({shock.name: pre_mat},  tickers, stage="pre_training")
            store.save_all({shock.name: post_mat}, tickers, stage="post_training")

        comparator = ShockComparator(store)
        analyzer   = SectorImpactAnalyzer(sm, comparator)

        # 1. vulnerability_score
        vuln = analyzer.vulnerability_score(shocks[0])
        assert "vulnerability_score" in vuln.columns
        assert (vuln["vulnerability_score"] >= 0).all() and (vuln["vulnerability_score"] <= 1).all()
        print(f"vulnerability_score        OK : most vulnerable={vuln.iloc[0]['sector']}")

        # 2. resilience_ranking
        ranking = analyzer.resilience_ranking(shocks)
        assert "resilience_class" in ranking.columns
        print(f"resilience_ranking         OK : {ranking.shape}")
        print(ranking.to_string(index=False))

        # 3. cross_shock_vulnerability
        cross = analyzer.cross_shock_vulnerability(shocks)
        assert "rank_std" in cross.columns
        print(f"\ncross_shock_vulnerability  OK : {cross.shape}")

        # 4. sector_response_profile
        profile = analyzer.sector_response_profile("Information Technology", shocks)
        assert len(profile) <= len(shocks)
        print(f"sector_response_profile    OK : {profile.shape}")

        # 5. narrative
        narrative = analyzer.generate_vulnerability_narrative(shocks)
        assert len(narrative) > 100
        print(f"\ngenerate_vulnerability_narrative OK :\n{narrative}")

        # Plotting
        import config.settings as cfg_mod
        orig_figs = cfg_mod.FIGURES_DIR
        cfg_mod.FIGURES_DIR = tmp / "figs"
        try:
            fig1, ax1 = analyzer.plot_resilience_ranking(shocks, save=True)
            fig2, ax2 = analyzer.plot_sector_heatmap(shocks, save=True)
            saved = list((tmp / "figs").glob("*.png"))
            print(f"\nplotting                   OK : {len(saved)} figures saved")
            assert len(saved) == 2
        finally:
            cfg_mod.FIGURES_DIR = orig_figs

        # ── Without comparator (price-only fallback) ──
        analyzer_no_comp = SectorImpactAnalyzer(sm, comparator=None)
        vuln2 = analyzer_no_comp.vulnerability_score(shocks[0])
        assert vuln2["centrality_norm"].isna().all()
        print(f"\nfallback (no comparator)   OK : centrality_norm all NaN as expected")

        print("\nAll sector_impact.py tests PASSED.")
    finally:
        adj_mod.PRE_TRAINING_DIR  = orig_pre
        adj_mod.POST_TRAINING_DIR = orig_post
        adj_mod.TICKER_FILE       = orig_tk
        shutil.rmtree(tmp)