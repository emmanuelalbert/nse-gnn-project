"""
shock_detection/visualize_shocks.py
------------------------------------
All visualisations for the shock detection stage.

Plots produced:
    1. timeline()            — Full ^NSEI return history with shock windows shaded
    2. rolling_avg_plot()    — Rolling 5-day average vs. ±2σ threshold band
    3. shock_duration_hist() — Distribution of shock durations
    4. sigma_breach_bar()    — Ranked bar chart of σ-breach magnitude per shock
    5. annual_heatmap()      — Heatmap of shock count / severity by year × month
    6. sector_impact_grid()  — Per-sector mean return across shock periods

Design notes:
  - All plots use a consistent dark-background style with sector colours
    from config.nifty50_universe.SECTOR_COLOURS.
  - Figures are saved to CFG.FIGURES_DIR as high-res PNGs (300 dpi).
  - Every function returns the (fig, ax) tuple so callers can further
    customise or embed plots in notebooks.
  - The four TARGET_SHOCK_PERIODS are always annotated with vertical lines
    so their position is clear relative to the full shock history.

Usage:
    from shock_detection.visualize_shocks import ShockVisualizer
    from shock_detection.shock_registry   import ShockRegistry

    registry = ShockRegistry()
    shocks   = registry.load()
    viz      = ShockVisualizer(shocks, returns_df, stats_df)

    viz.timeline(save=True)
    viz.rolling_avg_plot(save=True)
    viz.shock_duration_hist(save=True)
    viz.sigma_breach_bar(save=True)
    viz.annual_heatmap(save=True)
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as CFG
from config.nifty50_universe import SECTOR_COLOURS, get_sector_map
from shock_detection.detector import ShockPeriod

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PLOT STYLE
# ─────────────────────────────────────────────

STYLE = {
    "figure.facecolor" : "#0f1117",
    "axes.facecolor"   : "#161b22",
    "axes.edgecolor"   : "#30363d",
    "axes.labelcolor"  : "#c9d1d9",
    "xtick.color"      : "#8b949e",
    "ytick.color"      : "#8b949e",
    "text.color"       : "#c9d1d9",
    "grid.color"       : "#21262d",
    "grid.linestyle"   : "--",
    "grid.linewidth"   : 0.5,
    "legend.facecolor" : "#161b22",
    "legend.edgecolor" : "#30363d",
    "font.family"      : "DejaVu Sans",
    "font.size"        : 9,
}

SHOCK_FILL_COLOUR  = "#ff7b72"     # red tint for shock windows
TARGET_LINE_COLOUR = "#f0883e"     # orange for TARGET_SHOCK_PERIODS
RETURN_LINE_COLOUR = "#58a6ff"     # blue for return series
ROLLING_COLOUR     = "#ffa657"     # amber for rolling average
THRESHOLD_COLOUR   = "#ff7b72"     # red for threshold bands
SAFE_ZONE_COLOUR   = "#238636"     # green fill for safe zone


class ShockVisualizer:
    """
    Produces all shock detection visualisations.

    Args:
        shocks   : List of ShockPeriod objects from the registry.
        returns  : Full log returns DataFrame (T, N) with DatetimeIndex.
        stats_df : Optional — output of ShockDetector.detect_with_stats().
                   If None, rolling avg plots are skipped.
        save_dir : Directory for saving figures (default: CFG.FIGURES_DIR).
    """

    def __init__(
        self,
        shocks   : List[ShockPeriod],
        returns  : pd.DataFrame,
        stats_df : Optional[pd.DataFrame] = None,
        save_dir : Path = CFG.FIGURES_DIR,
    ):
        self.shocks   = sorted(shocks, key=lambda s: s.start)
        self.returns  = returns
        self.stats_df = stats_df
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.ref_series = (
            returns[CFG.SHOCK_REFERENCE_TICKER]
            if CFG.SHOCK_REFERENCE_TICKER in returns.columns
            else returns.iloc[:, -1]
        )

        plt.rcParams.update(STYLE)

    # ─────────────────────────────────────────
    # 1. FULL TIMELINE
    # ─────────────────────────────────────────

    def timeline(
        self,
        figsize: Tuple = (16, 6),
        save: bool = True,
        filename: str = "shock_timeline.png",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the full ^NSEI daily log return history with:
          - Shock windows shaded in red
          - TARGET_SHOCK_PERIODS annotated with orange vertical lines + labels
          - Zero line for reference
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Daily returns
        ax.plot(
            self.ref_series.index,
            self.ref_series.values,
            color=RETURN_LINE_COLOUR,
            linewidth=0.6,
            alpha=0.85,
            label="^NSEI daily log return",
        )
        ax.axhline(0, color="#8b949e", linewidth=0.6, linestyle="-")

        # Shade all detected shock windows
        shock_patch_added = False
        for shock in self.shocks:
            ax.axvspan(
                pd.Timestamp(shock.start),
                pd.Timestamp(shock.end),
                alpha=0.25,
                color=SHOCK_FILL_COLOUR,
                label="Shock period" if not shock_patch_added else "",
            )
            shock_patch_added = True

        # Annotate the 4 TARGET_SHOCK_PERIODS with vertical lines
        for target in CFG.TARGET_SHOCK_PERIODS:
            mid = pd.Timestamp(target["start"]) + (
                pd.Timestamp(target["end"]) - pd.Timestamp(target["start"])
            ) / 2
            ax.axvline(
                pd.Timestamp(target["start"]),
                color=TARGET_LINE_COLOUR,
                linewidth=1.0,
                linestyle="--",
                alpha=0.9,
            )
            ax.text(
                pd.Timestamp(target["start"]),
                ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 0.04,
                f" {target['name'].split()[0]}",
                color=TARGET_LINE_COLOUR,
                fontsize=7,
                rotation=90,
                va="top",
                ha="left",
                alpha=0.9,
            )

        ax.set_title(
            f"Nifty 50 Index — Daily Log Returns with Detected Shock Periods "
            f"({len(self.shocks)} shocks, {self.ref_series.index.min().year}–"
            f"{self.ref_series.index.max().year})",
            fontsize=11, pad=10,
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Log Return")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, axis="y")
        fig.tight_layout()

        if save:
            self._save(fig, filename)
        return fig, ax

    # ─────────────────────────────────────────
    # 2. ROLLING AVERAGE vs. THRESHOLD BAND
    # ─────────────────────────────────────────

    def rolling_avg_plot(
        self,
        figsize: Tuple = (16, 5),
        save: bool = True,
        filename: str = "rolling_avg_threshold.png",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the 5-day rolling average of ^NSEI returns alongside the ±2σ
        threshold band (paper Eq. 2 visualised).
        """
        if self.stats_df is None:
            logger.warning("stats_df not provided — skipping rolling_avg_plot.")
            return None, None

        fig, ax = plt.subplots(figsize=figsize)
        df = self.stats_df.dropna(subset=["rolling_avg"])

        # Safe zone fill
        ax.fill_between(
            df.index,
            df["threshold_lower"],
            df["threshold_upper"],
            alpha=0.12,
            color=SAFE_ZONE_COLOUR,
            label=f"±{CFG.SHOCK_SIGMA_THRESHOLD}σ safe zone",
        )

        # Threshold lines
        ax.axhline(
            df["threshold_upper"].iloc[0],
            color=THRESHOLD_COLOUR, linewidth=0.8, linestyle="--",
            label=f"+{CFG.SHOCK_SIGMA_THRESHOLD}σ threshold",
        )
        ax.axhline(
            df["threshold_lower"].iloc[0],
            color=THRESHOLD_COLOUR, linewidth=0.8, linestyle="--",
            label=f"−{CFG.SHOCK_SIGMA_THRESHOLD}σ threshold",
        )
        ax.axhline(0, color="#8b949e", linewidth=0.5)

        # Colour the rolling avg: red when shocked, blue when safe
        is_shocked = df["is_shocked"].fillna(False)
        for i in range(1, len(df)):
            colour = SHOCK_FILL_COLOUR if is_shocked.iloc[i] else ROLLING_COLOUR
            ax.plot(
                df.index[i-1:i+1],
                df["rolling_avg"].values[i-1:i+1],
                color=colour,
                linewidth=1.1,
                alpha=0.9,
            )

        # Legend proxies
        safe_proxy   = mpatches.Patch(color=ROLLING_COLOUR,    label="Rolling avg (safe)")
        shock_proxy  = mpatches.Patch(color=SHOCK_FILL_COLOUR, label="Rolling avg (shocked)")
        band_proxy   = mpatches.Patch(color=SAFE_ZONE_COLOUR,  alpha=0.4, label="±2σ band")
        ax.legend(handles=[safe_proxy, shock_proxy, band_proxy], fontsize=8)

        ax.set_title(
            f"5-Day Rolling Average Log Return — ^NSEI vs. ±{CFG.SHOCK_SIGMA_THRESHOLD}σ Threshold (Paper Eq. 2)",
            fontsize=11, pad=10,
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("5-Day Rolling Avg Log Return")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax.grid(True)
        fig.tight_layout()

        if save:
            self._save(fig, filename)
        return fig, ax

    # ─────────────────────────────────────────
    # 3. DURATION HISTOGRAM
    # ─────────────────────────────────────────

    def shock_duration_hist(
        self,
        figsize: Tuple = (8, 4),
        save: bool = True,
        filename: str = "shock_duration_hist.png",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Histogram of shock period durations in trading days.
        """
        durations = [s.duration for s in self.shocks]
        fig, ax   = plt.subplots(figsize=figsize)

        n, bins, patches = ax.hist(
            durations,
            bins=max(10, len(set(durations)) // 2),
            color=RETURN_LINE_COLOUR,
            edgecolor="#0f1117",
            alpha=0.85,
        )

        ax.axvline(np.mean(durations), color=TARGET_LINE_COLOUR,
                   linestyle="--", linewidth=1.2,
                   label=f"Mean = {np.mean(durations):.1f} days")
        ax.axvline(CFG.SHOCK_WINDOW_DAYS, color="#3fb950",
                   linestyle=":", linewidth=1.0,
                   label=f"Min threshold = {CFG.SHOCK_WINDOW_DAYS} days")

        ax.set_title(f"Distribution of Shock Period Durations (n={len(self.shocks)})", fontsize=11)
        ax.set_xlabel("Duration (trading days)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y")
        fig.tight_layout()

        if save:
            self._save(fig, filename)
        return fig, ax

    # ─────────────────────────────────────────
    # 4. SIGMA BREACH BAR CHART
    # ─────────────────────────────────────────

    def sigma_breach_bar(
        self,
        top_n: int = 20,
        figsize: Tuple = (12, 5),
        save: bool = True,
        filename: str = "sigma_breach_bar.png",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Horizontal bar chart of the top_n shock periods ranked by σ-breach magnitude.
        Target periods highlighted in orange.
        """
        sorted_shocks = sorted(self.shocks, key=lambda s: s.sigma_breach, reverse=True)[:top_n]
        target_names  = {t["name"] for t in CFG.TARGET_SHOCK_PERIODS}

        names    = [s.name.split(" (")[0][:30] for s in sorted_shocks]
        breaches = [s.sigma_breach for s in sorted_shocks]
        colours  = [
            TARGET_LINE_COLOUR if s.name in target_names else RETURN_LINE_COLOUR
            for s in sorted_shocks
        ]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(range(len(names)), breaches, color=colours, edgecolor="#0f1117", alpha=0.88)

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.axvline(CFG.SHOCK_SIGMA_THRESHOLD, color="#3fb950",
                   linestyle="--", linewidth=1.0, label=f"{CFG.SHOCK_SIGMA_THRESHOLD}σ threshold")
        ax.set_xlabel("σ-Breach Magnitude")
        ax.set_title(f"Top {top_n} Shock Periods by σ-Breach Magnitude", fontsize=11)
        ax.invert_yaxis()

        # Value labels on bars
        for bar, val in zip(bars, breaches):
            ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}σ", va="center", fontsize=7, color="#c9d1d9")

        target_patch = mpatches.Patch(color=TARGET_LINE_COLOUR, label="Target analysis period")
        other_patch  = mpatches.Patch(color=RETURN_LINE_COLOUR, label="Other shock period")
        ax.legend(handles=[target_patch, other_patch], fontsize=8, loc="lower right")
        ax.grid(True, axis="x")
        fig.tight_layout()

        if save:
            self._save(fig, filename)
        return fig, ax

    # ─────────────────────────────────────────
    # 5. ANNUAL HEATMAP
    # ─────────────────────────────────────────

    def annual_heatmap(
        self,
        metric: str = "count",
        figsize: Tuple = (14, 5),
        save: bool = True,
        filename: str = "annual_shock_heatmap.png",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Heatmap of shock activity by year × month.

        Args:
            metric : "count"         — number of shock days per cell
                     "avg_return"    — mean return of ^NSEI during shock days
                     "sigma_breach"  — max sigma breach among shocks in cell
        """
        # Build a day-level shock series
        shock_days = {}
        for shock in self.shocks:
            dr = pd.date_range(shock.start, shock.end, freq="B")
            for d in dr:
                if metric == "count":
                    shock_days[d] = shock_days.get(d, 0) + 1
                elif metric == "avg_return":
                    shock_days[d] = shock.avg_return
                elif metric == "sigma_breach":
                    shock_days[d] = max(shock_days.get(d, 0), shock.sigma_breach)

        if not shock_days:
            logger.warning("No shock days to plot.")
            return None, None

        series = pd.Series(shock_days).sort_index()
        series.index = pd.to_datetime(series.index)

        # Aggregate by year-month
        monthly = series.resample("MS").sum() if metric == "count" else series.resample("MS").mean()
        pivot   = monthly.to_frame(name=metric)
        pivot["year"]  = pivot.index.year
        pivot["month"] = pivot.index.month
        matrix = pivot.pivot(index="year", columns="month", values=metric).fillna(0)

        month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"]
        matrix.columns = [month_labels[m - 1] for m in matrix.columns]

        fig, ax = plt.subplots(figsize=figsize)
        import matplotlib.colors as mcolors
        cmap = plt.cm.YlOrRd

        im = ax.imshow(matrix.values, cmap=cmap, aspect="auto", interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
        cbar.set_label(metric, fontsize=8)

        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, fontsize=8)
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index, fontsize=8)

        # Annotate cells with values
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix.values[i, j]
                if val > 0:
                    ax.text(j, i, f"{val:.1f}" if metric != "count" else str(int(val)),
                            ha="center", va="center", fontsize=6.5,
                            color="white" if val > matrix.values.max() * 0.6 else "#c9d1d9")

        title_map = {
            "count"       : "Number of Shock Days",
            "avg_return"  : "Mean ^NSEI Return During Shocks",
            "sigma_breach": "Max σ-Breach During Shocks",
        }
        ax.set_title(f"Annual Shock Heatmap — {title_map.get(metric, metric)}", fontsize=11, pad=10)
        fig.tight_layout()

        if save:
            self._save(fig, filename)
        return fig, ax

    # ─────────────────────────────────────────
    # 6. SECTOR IMPACT GRID (across target shocks)
    # ─────────────────────────────────────────

    def sector_impact_grid(
        self,
        target_shocks: Optional[List[ShockPeriod]] = None,
        figsize: Tuple = (14, 8),
        save: bool = True,
        filename: str = "sector_impact_grid.png",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Grid of bar charts: one subplot per target shock period, showing
        mean log return per sector. Mirrors Figures 10, 16, 22 in the paper.
        """
        from config.nifty50_universe import get_stocks_by_sector, get_sectors

        target_shocks = target_shocks or [
            s for s in self.shocks
            if s.name in {t["name"] for t in CFG.TARGET_SHOCK_PERIODS}
        ]

        if not target_shocks:
            logger.warning("No target shocks found for sector_impact_grid.")
            return None, None

        sectors      = get_sectors()
        sector_map   = get_sector_map()
        n_periods    = len(target_shocks)
        fig, axes    = plt.subplots(1, n_periods, figsize=figsize, sharey=False)

        if n_periods == 1:
            axes = [axes]

        for ax, shock in zip(axes, target_shocks):
            mask   = (
                (self.returns.index >= pd.Timestamp(shock.start)) &
                (self.returns.index <= pd.Timestamp(shock.end))
            )
            period = self.returns.loc[mask]

            sector_means = {}
            for ticker in period.columns:
                sec = sector_map.get(ticker, "Unknown")
                if sec == "Index":
                    continue
                sector_means.setdefault(sec, []).append(period[ticker].mean(skipna=True))

            sector_avg = {
                sec: np.nanmean(vals)
                for sec, vals in sector_means.items()
                if vals
            }
            sorted_secs  = sorted(sector_avg, key=lambda s: sector_avg[s])
            sorted_vals  = [sector_avg[s] for s in sorted_secs]
            sorted_cols  = [SECTOR_COLOURS.get(s, "#8b949e") for s in sorted_secs]

            bars = ax.barh(sorted_secs, sorted_vals, color=sorted_cols,
                           edgecolor="#0f1117", alpha=0.88)
            ax.axvline(0, color="#8b949e", linewidth=0.7)
            ax.set_title(f"{shock.name}\n[{shock.start}]", fontsize=8, pad=4)
            ax.set_xlabel("Mean Log Return", fontsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(True, axis="x", alpha=0.4)

        fig.suptitle(
            "Average Sector Returns During Target Shock Periods",
            fontsize=11, y=1.02,
        )
        fig.tight_layout()

        if save:
            self._save(fig, filename)
        return fig, axes

    # ─────────────────────────────────────────
    # SAVE ALL
    # ─────────────────────────────────────────

    def plot_all(self) -> None:
        """Render and save every plot in sequence."""
        logger.info("Generating all shock detection visualisations...")
        self.timeline(save=True)
        self.rolling_avg_plot(save=True)
        self.shock_duration_hist(save=True)
        self.sigma_breach_bar(save=True)
        self.annual_heatmap(save=True)
        self.sector_impact_grid(save=True)
        logger.info(f"All figures saved to {self.save_dir}")

    # ─────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────

    def _save(self, fig: plt.Figure, filename: str) -> None:
        path = self.save_dir / filename
        fig.savefig(path, dpi=300, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info(f"Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, shutil
    logging.basicConfig(level="INFO", format=CFG.LOG_FORMAT)
    np.random.seed(42)

    # Synthetic data
    n      = 4500
    dates  = pd.date_range("2005-01-03", periods=n, freq="B")
    cols   = ["INFY.NS", "TCS.NS", "HDFCBANK.NS", "RELIANCE.NS", "SUNPHARMA.NS", "^NSEI"]
    sector_map_local = {
        "INFY.NS": "Information Technology", "TCS.NS": "Information Technology",
        "HDFCBANK.NS": "Financials", "RELIANCE.NS": "Energy",
        "SUNPHARMA.NS": "Pharma", "^NSEI": "Index",
    }

    data   = np.random.normal(0.0003, 0.010, size=(n, len(cols)))
    # Inject 3 shock windows
    for s, e in [(400, 430), (1500, 1560), (3800, 3840)]:
        data[s:e, -1] = np.random.normal(-0.025, 0.015, size=e - s)
        data[s:e, :-1] += np.random.normal(-0.018, 0.012, size=(e - s, len(cols) - 1))

    returns = pd.DataFrame(data, index=dates, columns=cols)
    returns.index.name = "Date"

    from shock_detection.detector import ShockDetector
    detector       = ShockDetector()
    shocks, stats  = detector.detect_with_stats(returns)

    # Save to temp dir
    tmp = Path(tempfile.mkdtemp())
    viz = ShockVisualizer(shocks, returns, stats_df=stats, save_dir=tmp)

    fig, ax = viz.timeline(save=True)
    fig, ax = viz.rolling_avg_plot(save=True)
    fig, ax = viz.shock_duration_hist(save=True)
    fig, ax = viz.sigma_breach_bar(save=True)
    fig, ax = viz.annual_heatmap(save=True)

    saved = list(tmp.glob("*.png"))
    print(f"\n{len(saved)} figures saved to {tmp}:")
    for p in saved:
        print(f"  {p.name}")

    shutil.rmtree(tmp)
    print("\nAll ShockVisualizer tests PASSED.")