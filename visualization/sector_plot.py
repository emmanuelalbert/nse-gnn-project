"""
visualization/sector_plot.py
-------------------------------
Average-change and amplitude bar charts at the sector level, plus
cross-shock sector comparison plots.

Paper reference: Section IV-C-1-d, 2-d, 3-d (Figures 10–11, 16–17, 22–23)

    Figure 10/16/22 — Average Change by stock and sector:
        "This metric represents the average change in returns for each
         stock and sector, revealing the financial impact of the shock."

    Figure 11/17/23 — Amplitude by stock and sector:
        "This measures the amplitude of returns per stock and the average
         amplitude of returns per sector, highlighting the extent of
         variability during the shock period."

    Section IV-D-2 — Cross-period sector comparison:
        "The impact of shocks across different sectors varied significantly
         across the crises studied: 2008 FInancial Crisis: profoundly
         affecting the financial sector first... COVID-19: Consumer
         Discretionary and Energy disrupted; IT showed resilience."

Outputs:
    1. sector_avg_change()     — horizontal bars: avg change per sector,
                                  coloured green (positive=resilient) /
                                  red (negative=vulnerable)
    2. sector_amplitude()      — horizontal bars: amplitude per sector,
                                  ordered most volatile → least volatile
    3. stock_avg_change()      — full per-stock average change with
                                  sector colour coding (paper Fig 10/16/22)
    4. stock_amplitude()       — full per-stock amplitude chart
                                  (paper Fig 11/17/23)
    5. cross_shock_comparison()— grouped bars: sector amplitude compared
                                  across all target shock periods side-by-side
    6. resilience_waterfall()  — sector vulnerability score waterfall from
                                  SectorImpactAnalyzer, visual ranking

Usage:
    from visualization.sector_plot import SectorPlotter
    plotter = SectorPlotter(shock_metrics_analyzer)
    plotter.sector_avg_change(shock, save=True)
    plotter.sector_amplitude(shock, save=True)
    plotter.stock_avg_change(shock, save=True)
    plotter.stock_amplitude(shock, save=True)
    plotter.cross_shock_comparison(shocks, save=True)
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as CFG
from config.nifty50_universe import SECTOR_COLOURS, get_sector_map
from evaluation._plot_style import apply_style, save_figure, COLOR_CYCLE
from network_analysis.shock_metrics import ShockMetricsAnalyzer
from shock_detection.detector import ShockPeriod

logger = logging.getLogger(__name__)


class SectorPlotter:
    """
    Sector-level and stock-level average-change / amplitude visualizations.

    Args:
        shock_metrics : ShockMetricsAnalyzer instance (prices + returns).
    """

    def __init__(self, shock_metrics: ShockMetricsAnalyzer):
        self.sm         = shock_metrics
        self.sector_map = get_sector_map()

    # ─────────────────────────────────────────
    # 1. SECTOR AVERAGE CHANGE  (paper Fig 10/16/22 sector-level)
    # ─────────────────────────────────────────

    def sector_avg_change(
        self,
        shock    : ShockPeriod,
        figsize  : Tuple = (9, 6),
        save     : bool  = True,
        filename : Optional[str] = None,
    ):
        """
        Horizontal bar chart of mean average change per sector.
        Green = positive return (resilient), Red = negative (vulnerable).
        Mirrors the sector-level aggregation in paper Figures 10, 16, 22.
        """
        apply_style()
        df = self.sm.sector_report(shock.name, shock)
        df = df.sort_values("mean_avg_change")

        colours = [
            "#3fb950" if v >= 0 else "#ff7b72"
            for v in df["mean_avg_change"]
        ]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(df["sector"], df["mean_avg_change"], color=colours, alpha=0.88)
        ax.axvline(0, color="#8b949e", linewidth=0.8)
        ax.set_xlabel("Mean Average Change  (Eq. 19)")
        ax.set_title(
            f"Average Change by Sector — {shock.name}\n"
            f"[{shock.start} → {shock.end}]",
            fontsize=11,
        )
        ax.grid(True, axis="x", alpha=0.3)

        handles = [
            mpatches.Patch(color="#3fb950", label="Positive (resilient)"),
            mpatches.Patch(color="#ff7b72", label="Negative (vulnerable)"),
        ]
        ax.legend(handles=handles, fontsize=8)
        fig.tight_layout()

        fname = filename or f"sector_avgchange_{self._slug(shock.name)}.png"
        if save:
            save_figure(fig, fname)
        return fig, ax

    # ─────────────────────────────────────────
    # 2. SECTOR AMPLITUDE  (paper Fig 11/17/23 sector-level)
    # ─────────────────────────────────────────

    def sector_amplitude(
        self,
        shock    : ShockPeriod,
        figsize  : Tuple = (9, 6),
        save     : bool  = True,
        filename : Optional[str] = None,
    ):
        """
        Horizontal bar chart of mean amplitude per sector, coloured by sector.
        Ordered most volatile → least volatile.
        """
        apply_style()
        df = self.sm.sector_report(shock.name, shock)
        df = df.sort_values("mean_amplitude", ascending=True)

        colours = [SECTOR_COLOURS.get(s, "#8b949e") for s in df["sector"]]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(df["sector"], df["mean_amplitude"], color=colours, alpha=0.88)
        ax.set_xlabel("Mean Amplitude  (Eq. 20)  =  Max − Min Daily Return")
        ax.set_title(
            f"Amplitude by Sector — {shock.name}\n"
            f"[{shock.start} → {shock.end}]",
            fontsize=11,
        )
        ax.grid(True, axis="x", alpha=0.3)

        # Annotate with number of tickers per sector
        for _, row in df.iterrows():
            ax.text(
                row["mean_amplitude"] * 1.01,
                df.index[df["sector"] == row["sector"]].tolist()[0]
                if False else list(df["sector"]).index(row["sector"]),
                f"n={row['n_tickers']}",
                va="center", fontsize=7, color="#8b949e",
            )

        fig.tight_layout()
        fname = filename or f"sector_amplitude_{self._slug(shock.name)}.png"
        if save:
            save_figure(fig, fname)
        return fig, ax

    # ─────────────────────────────────────────
    # 3. STOCK AVERAGE CHANGE  (paper Fig 10/16/22 — full stock detail)
    # ─────────────────────────────────────────

    def stock_avg_change(
        self,
        shock    : ShockPeriod,
        top_n    : int   = 50,
        figsize  : Tuple = (10, 14),
        save     : bool  = True,
        filename : Optional[str] = None,
    ):
        """
        Per-stock average change bar chart, coloured by sector.
        Direct equivalent of paper Figures 10, 16, 22.
        """
        apply_style()
        df = self.sm.stock_report(shock.name, shock)
        df = df.sort_values("avg_change").head(top_n)

        colours = [SECTOR_COLOURS.get(s, "#8b949e") for s in df["sector"]]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(df.index, df["avg_change"], color=colours, alpha=0.88)
        ax.axvline(0, color="#8b949e", linewidth=0.7)
        ax.set_xlabel("Average Daily Change  (Eq. 19)")
        ax.set_title(
            f"Average Change by Stock and Sector — {shock.name}\n"
            f"[{shock.start} → {shock.end}]",
            fontsize=11,
        )
        ax.tick_params(axis="y", labelsize=6)

        for lbl, ticker in zip(ax.get_yticklabels(), df.index):
            sector = self.sector_map.get(ticker, "Unknown")
            lbl.set_color(SECTOR_COLOURS.get(sector, "#c9d1d9"))

        self._add_sector_legend(ax)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()

        fname = filename or f"stock_avgchange_{self._slug(shock.name)}.png"
        if save:
            save_figure(fig, fname)
        return fig, ax

    # ─────────────────────────────────────────
    # 4. STOCK AMPLITUDE  (paper Fig 11/17/23 — full stock detail)
    # ─────────────────────────────────────────

    def stock_amplitude(
        self,
        shock    : ShockPeriod,
        top_n    : int   = 50,
        figsize  : Tuple = (10, 14),
        save     : bool  = True,
        filename : Optional[str] = None,
    ):
        """
        Per-stock amplitude bar chart, coloured by sector.
        Direct equivalent of paper Figures 11, 17, 23.
        """
        apply_style()
        df = self.sm.stock_report(shock.name, shock)
        df = df.sort_values("amplitude", ascending=True).head(top_n)

        colours = [SECTOR_COLOURS.get(s, "#8b949e") for s in df["sector"]]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(df.index, df["amplitude"], color=colours, alpha=0.88)
        ax.set_xlabel("Amplitude  (Eq. 20)  =  Max − Min Daily Log Return")
        ax.set_title(
            f"Amplitude by Stock and Sector — {shock.name}\n"
            f"[{shock.start} → {shock.end}]",
            fontsize=11,
        )
        ax.tick_params(axis="y", labelsize=6)

        for lbl, ticker in zip(ax.get_yticklabels(), df.index):
            sector = self.sector_map.get(ticker, "Unknown")
            lbl.set_color(SECTOR_COLOURS.get(sector, "#c9d1d9"))

        self._add_sector_legend(ax)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()

        fname = filename or f"stock_amplitude_{self._slug(shock.name)}.png"
        if save:
            save_figure(fig, fname)
        return fig, ax

    # ─────────────────────────────────────────
    # 5. CROSS-SHOCK SECTOR COMPARISON  (paper Section IV-D-2)
    # ─────────────────────────────────────────

    def cross_shock_comparison(
        self,
        shocks   : List[ShockPeriod],
        metric   : str   = "mean_amplitude",
        figsize  : Tuple = (13, 7),
        save     : bool  = True,
        filename : Optional[str] = None,
    ):
        """
        Grouped horizontal bar chart comparing a sector metric across
        multiple shock periods side by side.

        Reproduces the paper's Section IV-D-2 finding that different
        crises affect different sectors most severely — e.g.:
            GFC     → Financials hardest hit
            COVID   → IT resilient, Consumer Discretionary vulnerable

        Args:
            metric : Column from sector_report() to compare.
                     "mean_amplitude" | "mean_avg_change" | "mean_abs_avg_change"
        """
        apply_style()
        sector_data = {}
        for shock in shocks:
            try:
                df = self.sm.sector_report(shock.name, shock).set_index("sector")
                sector_data[shock.name] = df[metric]
            except Exception as exc:
                logger.warning(f"Skipping '{shock.name}': {exc}")

        if not sector_data:
            logger.warning("No data for cross_shock_comparison.")
            return None, None

        wide = pd.DataFrame(sector_data).fillna(0)
        # Sort sectors by mean across shocks
        wide = wide.loc[wide.mean(axis=1).sort_values(ascending=True).index]

        fig, ax = plt.subplots(figsize=figsize)
        n_shocks = len(sector_data)
        y        = np.arange(len(wide))
        total_h  = 0.75
        bar_h    = total_h / n_shocks

        for i, (shock_name, series) in enumerate(sector_data.items()):
            vals   = series.reindex(wide.index).fillna(0)
            offset = (i - n_shocks / 2 + 0.5) * bar_h
            ax.barh(
                y + offset, vals.values,
                height=bar_h * 0.85,
                color=COLOR_CYCLE[i % len(COLOR_CYCLE)],
                label=shock_name[:40], alpha=0.88,
            )

        ax.set_yticks(y)
        ax.set_yticklabels(
            [f"{s}  " for s in wide.index],
            fontsize=8,
        )
        for lbl, sector in zip(ax.get_yticklabels(), wide.index):
            lbl.set_color(SECTOR_COLOURS.get(sector, "#c9d1d9"))

        label_map = {
            "mean_amplitude"        : "Mean Amplitude  (Eq. 20)",
            "mean_avg_change"       : "Mean Average Change  (Eq. 19)",
            "mean_abs_avg_change"   : "Mean |Average Change|",
        }
        ax.set_xlabel(label_map.get(metric, metric))
        ax.set_title(
            f"Sector {label_map.get(metric, metric)} — Cross-Shock Comparison\n"
            f"({len(shocks)} shock periods)",
            fontsize=11,
        )
        ax.axvline(0, color="#8b949e", linewidth=0.6)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()

        fname = filename or f"sector_cross_{metric}.png"
        if save:
            save_figure(fig, fname)
        return fig, ax

    # ─────────────────────────────────────────
    # 6. RESILIENCE WATERFALL  (sector vulnerability ranking)
    # ─────────────────────────────────────────

    def resilience_waterfall(
        self,
        shocks   : List[ShockPeriod],
        figsize  : Tuple = (11, 7),
        save     : bool  = True,
        filename : str   = "sector_resilience_waterfall.png",
    ):
        """
        Waterfall-style chart showing cumulative vulnerability contribution
        per sector across all shocks. Visually encodes:
          - Bar height = mean amplitude across all shocks
          - Bar colour = resilience class (green/amber/red)
          - Error bar  = std across shocks (consistency of impact)

        Complements SectorImpactAnalyzer.plot_resilience_ranking() with
        amplitude-based (rather than composite-score-based) ordering.
        """
        apply_style()
        sector_rows = []
        for shock in shocks:
            try:
                df = self.sm.sector_report(shock.name, shock)
                for _, row in df.iterrows():
                    sector_rows.append({
                        "sector"    : row["sector"],
                        "amplitude" : row["mean_amplitude"],
                        "avg_change": row["mean_avg_change"],
                    })
            except Exception as exc:
                logger.warning(f"Skipping '{shock.name}': {exc}")

        if not sector_rows:
            logger.warning("No data for resilience_waterfall.")
            return None, None

        agg = (
            pd.DataFrame(sector_rows)
            .groupby("sector")["amplitude"]
            .agg(mean="mean", std="std")
            .fillna(0)
            .sort_values("mean", ascending=False)
            .reset_index()
        )

        # Colour by tercile
        q1, q2 = agg["mean"].quantile([1/3, 2/3])
        def _colour(v):
            if v >= q2:
                return "#ff7b72"   # vulnerable (high amplitude)
            elif v >= q1:
                return "#f0883e"   # moderate
            return "#3fb950"       # resilient (low amplitude)

        colours = [_colour(v) for v in agg["mean"]]

        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(agg))
        ax.bar(
            x, agg["mean"], color=colours, alpha=0.88,
            yerr=agg["std"], error_kw={"ecolor": "#8b949e", "capsize": 4, "elinewidth": 1},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(agg["sector"], rotation=40, ha="right", fontsize=8)
        for lbl, sector in zip(ax.get_xticklabels(), agg["sector"]):
            lbl.set_color(SECTOR_COLOURS.get(sector, "#c9d1d9"))

        ax.set_ylabel("Mean Amplitude Across Shock Periods  (Eq. 20)")
        ax.set_title(
            f"Sector Resilience Waterfall — {len(shocks)} Shock Periods\n"
            f"(red=vulnerable, green=resilient)",
            fontsize=11,
        )
        ax.grid(True, axis="y", alpha=0.3)

        handles = [
            mpatches.Patch(color="#ff7b72", label="Vulnerable  (high amplitude)"),
            mpatches.Patch(color="#f0883e", label="Moderate"),
            mpatches.Patch(color="#3fb950", label="Resilient   (low amplitude)"),
        ]
        ax.legend(handles=handles, fontsize=8)
        fig.tight_layout()

        if save:
            save_figure(fig, filename)
        return fig, ax

    # ─────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────

    def _add_sector_legend(self, ax, loc: str = "lower right") -> None:
        handles = [
            mpatches.Patch(facecolor=colour, label=sector)
            for sector, colour in SECTOR_COLOURS.items()
            if sector != "Index"
        ]
        ax.legend(handles=handles, fontsize=6, loc=loc, ncol=2, framealpha=0.4)

    @staticmethod
    def _slug(name: str) -> str:
        return name.lower().replace(" ", "_").replace("(", "").replace(")", "")[:55]


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, shutil
    logging.basicConfig(level="INFO", format=CFG.LOG_FORMAT)
    np.random.seed(23)

    from config.nifty50_universe import get_tickers
    tickers = get_tickers(); N = len(tickers)

    dates  = pd.bdate_range("2008-01-01", "2021-01-01")
    prices = pd.DataFrame(
        np.cumprod(1 + np.random.normal(0, 0.015, (len(dates), N)), axis=0) * 1000,
        index=dates, columns=tickers,
    )
    log_p   = np.log(prices)
    returns = (log_p - log_p.shift(1)).iloc[1:]

    shocks = [
        ShockPeriod("Global Financial Crisis",  "2008-09-15", "2008-10-10", 18, -0.031, -0.082, 4.2, "global_financial"),
        ShockPeriod("IL&FS Crisis",              "2018-09-21", "2018-10-26", 25, -0.018, -0.045, 2.8, "domestic_financial"),
        ShockPeriod("COVID-19 Pandemic Onset",   "2020-02-20", "2020-03-23", 23, -0.038, -0.135, 5.1, "health_nonfinancial"),
    ]

    sm = ShockMetricsAnalyzer(prices, returns)

    tmp = Path(tempfile.mkdtemp())
    import config.settings as cfg_mod
    orig_figs = cfg_mod.FIGURES_DIR
    cfg_mod.FIGURES_DIR = tmp

    try:
        plotter = SectorPlotter(sm)
        shock   = shocks[0]   # GFC

        fig1, _ = plotter.sector_avg_change(shock, save=True)
        assert fig1 is not None;    print("sector_avg_change      OK")

        fig2, _ = plotter.sector_amplitude(shock, save=True)
        assert fig2 is not None;    print("sector_amplitude       OK")

        fig3, _ = plotter.stock_avg_change(shock, save=True)
        assert fig3 is not None;    print("stock_avg_change       OK")

        fig4, _ = plotter.stock_amplitude(shock, save=True)
        assert fig4 is not None;    print("stock_amplitude        OK")

        fig5, _ = plotter.cross_shock_comparison(shocks, save=True)
        assert fig5 is not None;    print("cross_shock_comparison OK")

        fig6, _ = plotter.resilience_waterfall(shocks, save=True)
        assert fig6 is not None;    print("resilience_waterfall   OK")

        saved = list(tmp.glob("*.png"))
        print(f"Figures saved          : {len(saved)}")
        assert len(saved) == 6

        print("\nAll sector_plot.py tests PASSED.")

    finally:
        cfg_mod.FIGURES_DIR = orig_figs
        shutil.rmtree(tmp)