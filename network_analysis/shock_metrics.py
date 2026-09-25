"""
network_analysis/shock_metrics.py
------------------------------------
Average change and amplitude reporting per stock and per sector across
shock periods.

Paper reference: Section III-E (Eq. 19–20) and Section IV-C-1-d / 2-d / 3-d
("Average Change and Amplitude")

    Eq. 19 — Average Change:
        AverageChange = Σ(P_current - P_previous) / N
        "Measures the average change in stock prices from day to day,
         reflecting the typical daily price movement."

    Eq. 20 — Amplitude:
        Amplitude = MaxDailyChange - MinDailyChange
        "Represents the range of stock price movements, measuring the
         extent of fluctuations."

    Paper narrative pattern (Section IV-C-1-d, "Average Change and Amplitude"):
        "Average Change: This metric represents the average change in
         returns for each stock and sector, revealing the financial impact
         of the shock. Sectors such as Consumer Discretionary (F, AMZN),
         Real Estate (SPG), and Financials (JPM, WFC) exhibited higher
         fluctuations in their average returns, indicating their increased
         sensitivity during the shock period."

        "Amplitude: This measures the amplitude of returns per stock and
         the average amplitude of returns per sector, highlighting the
         extent of variability during the shock period."

preprocessing/returns.py already implements the raw per-ticker computation
(ReturnCalculator.amplitude() uses log returns per paper Eq. 20;
ReturnCalculator.average_change() uses raw prices per paper Eq. 19 exactly
as specified). This module is the ANALYTICAL layer: it runs both metrics
across every shock period, aggregates to the sector level, ranks stocks by
sensitivity, and reproduces the paper's narrative structure.

Usage:
    from network_analysis.shock_metrics import ShockMetricsAnalyzer
    analyzer = ShockMetricsAnalyzer(prices_df, returns_df)
    report   = analyzer.stock_report("Global Financial Crisis", shock)
    sector   = analyzer.sector_report("Global Financial Crisis", shock)
    analyzer.plot_average_change("Global Financial Crisis", shock, save=True)
    analyzer.plot_amplitude("Global Financial Crisis", shock, save=True)
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
from preprocessing.returns import ReturnCalculator
from shock_detection.detector import ShockPeriod

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SHOCK METRICS ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

class ShockMetricsAnalyzer:
    """
    Average change (Eq. 19) and amplitude (Eq. 20) analysis at both the
    individual-stock and sector level, reproducing the paper's Section
    IV-C "Average Change and Amplitude" narrative for every shock period.

    Args:
        prices  : Raw adjusted closing prices DataFrame (T, N).
                  Used for average_change (paper Eq. 19 — defined on raw prices).
        returns : Log returns DataFrame (T, N).
                  Used for amplitude (paper Eq. 20 — defined on returns).
    """

    def __init__(
        self,
        prices  : pd.DataFrame,
        returns : pd.DataFrame,
    ):
        self.prices     = prices
        self.returns    = returns
        self.calc       = ReturnCalculator()
        self.sector_map = get_sector_map()

    # ─────────────────────────────────────────
    # STOCK-LEVEL REPORT
    # ─────────────────────────────────────────

    def stock_report(
        self,
        shock_name : str,
        shock      : ShockPeriod,
    ) -> pd.DataFrame:
        """
        Per-stock average change and amplitude for one shock period.

        Reproduces the per-ticker data underlying paper Figures 10-11,
        16-17, 22-23.

        Args:
            shock_name : Display name (used for logging only).
            shock      : ShockPeriod object with start/end dates.

        Returns:
            DataFrame indexed by ticker:
                sector, avg_change, amplitude,
                avg_change_rank, amplitude_rank (1 = most extreme)
        """
        avg_change = self.calc.average_change(self.prices, shock.start, shock.end)
        amplitude  = self.calc.amplitude(self.returns, shock.start, shock.end)

        df = pd.DataFrame({
            "avg_change" : avg_change,
            "amplitude"  : amplitude,
        })
        df["sector"] = [self.sector_map.get(t, "Unknown") for t in df.index]
        df = df[["sector", "avg_change", "amplitude"]]

        # Rank by absolute magnitude (most extreme = rank 1).
        # Use nullable Int64 dtype since some tickers may have NaN avg_change/
        # amplitude if they have no valid price/return data in this period
        # (e.g. period falls outside the available date range).
        df["avg_change_rank"] = (
            df["avg_change"].abs().rank(ascending=False, method="min").astype("Int64")
        )
        df["amplitude_rank"] = (
            df["amplitude"].rank(ascending=False, method="min").astype("Int64")
        )

        df = df.sort_values("amplitude_rank")

        logger.info(
            f"Stock report '{shock_name}' [{shock.start}→{shock.end}] | "
            f"most volatile: {df.index[0]} (amplitude={df.iloc[0]['amplitude']:.4f}) | "
            f"largest avg change: "
            f"{df['avg_change'].abs().idxmax()} "
            f"({df.loc[df['avg_change'].abs().idxmax(), 'avg_change']:.4f})"
        )
        return df

    # ─────────────────────────────────────────
    # SECTOR-LEVEL REPORT
    # ─────────────────────────────────────────

    def sector_report(
        self,
        shock_name : str,
        shock      : ShockPeriod,
    ) -> pd.DataFrame:
        """
        Aggregate average change and amplitude to the sector level.

        Reproduces paper sentences like:
            "Sectors such as Consumer Discretionary (F, AMZN), Real Estate
             (SPG), and Financials (JPM, WFC) exhibited higher fluctuations
             in their average returns, indicating their increased
             sensitivity during the shock period."

        Returns:
            DataFrame: sector, n_tickers,
                      mean_avg_change, mean_abs_avg_change, mean_amplitude,
                      sorted by mean_amplitude descending (most volatile sector first).
        """
        stock_df = self.stock_report(shock_name, shock)
        # Exclude the index itself from sector aggregation (paper analyses
        # company-level sectors, not the benchmark index)
        stock_df = stock_df[stock_df["sector"] != "Index"]

        sector_df = (
            stock_df.groupby("sector")
            .agg(
                n_tickers            = ("avg_change", "size"),
                mean_avg_change      = ("avg_change", "mean"),
                mean_abs_avg_change  = ("avg_change", lambda s: s.abs().mean()),
                mean_amplitude       = ("amplitude", "mean"),
                max_amplitude_ticker = ("amplitude", lambda s: s.idxmax()),
            )
            .round(4)
            .sort_values("mean_amplitude", ascending=False)
            .reset_index()
        )

        logger.info(
            f"Sector report '{shock_name}': "
            f"most volatile sector = {sector_df.iloc[0]['sector']} "
            f"(mean_amplitude={sector_df.iloc[0]['mean_amplitude']:.4f})"
        )
        return sector_df

    # ─────────────────────────────────────────
    # MULTI-SHOCK COMPARISON  (paper's 3-4 target periods)
    # ─────────────────────────────────────────

    def comparative_sector_report(
        self,
        shocks : List[ShockPeriod],
    ) -> pd.DataFrame:
        """
        Compare sector-level amplitude and average change across multiple
        shock periods side by side.

        Mirrors the paper's cross-period comparison (Section IV-D,
        "Sectoral Impact Patterns") which shows that different crises
        affect different sectors most severely:
            2008 GFC          → Financials
            2016 US Elections  → Healthcare, Energy, Finance
            COVID-19          → Energy, IT (resilient); Financials, Utilities (vulnerable)

        Args:
            shocks : List of ShockPeriod objects to compare.

        Returns:
            Wide-format DataFrame: sector × {shock_name}_amplitude columns,
            plus a 'most_volatile_in' column showing which shock period
            hit each sector hardest.
        """
        sector_frames = {}
        for shock in shocks:
            sec_df = self.sector_report(shock.name, shock)
            sector_frames[shock.name] = sec_df.set_index("sector")["mean_amplitude"]

        wide = pd.DataFrame(sector_frames).round(4)
        wide.columns = [f"{c}_amplitude" for c in wide.columns]

        # Identify which shock hit each sector hardest.
        # skipna handled implicitly by idxmax; guard the edge case where a
        # sector has NaN amplitude in every shock (idxmax raises ValueError).
        amp_cols = wide.columns.tolist()
        if wide[amp_cols].notna().any(axis=None):
            wide["most_volatile_in"] = wide[amp_cols].idxmax(axis=1).str.replace("_amplitude", "")
        else:
            wide["most_volatile_in"] = None

        wide = wide.sort_values(amp_cols[0] if amp_cols else wide.columns[0], ascending=False)

        logger.info(
            f"Comparative sector report across {len(shocks)} shocks: "
            f"{len(wide)} sectors compared."
        )
        return wide.reset_index()

    def comparative_target_shocks(self) -> pd.DataFrame:
        """
        Convenience wrapper: comparative_sector_report() restricted to the
        four shock periods defined in CFG.TARGET_SHOCK_PERIODS.

        Reproduces the paper's exact comparative structure (3 chosen periods
        in the paper; 4 in this India-adapted study).
        """
        shocks = [
            ShockPeriod(
                name=t["name"], start=t["start"], end=t["end"],
                duration=0, avg_return=0.0, max_drawdown=0.0,
                sigma_breach=0.0, type=t["type"],
            )
            for t in CFG.TARGET_SHOCK_PERIODS
        ]
        return self.comparative_sector_report(shocks)

    # ─────────────────────────────────────────
    # SENSITIVITY RANKING (top movers across ALL shocks)
    # ─────────────────────────────────────────

    def most_sensitive_stocks(
        self,
        shocks : List[ShockPeriod],
        top_n  : int = 15,
    ) -> pd.DataFrame:
        """
        Identify which stocks are consistently the most volatile (highest
        amplitude) across ALL given shock periods — i.e. structurally
        shock-sensitive stocks rather than period-specific outliers.

        Returns:
            DataFrame: ticker, sector, mean_amplitude_rank, n_shocks_in_top10,
                      sorted by mean_amplitude_rank ascending (most sensitive first).
        """
        rank_records: Dict[str, List[int]] = {}
        top10_count:  Dict[str, int]       = {}

        for shock in shocks:
            df = self.stock_report(shock.name, shock)
            for ticker, row in df.iterrows():
                if pd.isna(row["amplitude_rank"]):
                    continue   # skip tickers with no valid data in this period
                rank_records.setdefault(ticker, []).append(int(row["amplitude_rank"]))
                if row["amplitude_rank"] <= 10:
                    top10_count[ticker] = top10_count.get(ticker, 0) + 1

        rows = []
        for ticker, ranks in rank_records.items():
            rows.append({
                "ticker"               : ticker,
                "sector"               : self.sector_map.get(ticker, "Unknown"),
                "mean_amplitude_rank"  : round(float(np.mean(ranks)), 2),
                "n_shocks_evaluated"   : len(ranks),
                "n_shocks_in_top10"    : top10_count.get(ticker, 0),
            })

        df = (
            pd.DataFrame(rows)
            .sort_values("mean_amplitude_rank")
            .reset_index(drop=True)
            .head(top_n)
        )
        logger.info(
            f"Most sensitive stocks across {len(shocks)} shocks: "
            f"top = {df.iloc[0]['ticker']} (mean rank {df.iloc[0]['mean_amplitude_rank']})"
        )
        return df

    # ─────────────────────────────────────────
    # PLOTTING
    # ─────────────────────────────────────────

    def plot_average_change(
        self,
        shock_name : str,
        shock      : ShockPeriod,
        figsize    : Tuple = (12, 7),
        save       : bool  = True,
        filename   : Optional[str] = None,
    ):
        """
        Bar chart of average change by stock, coloured by sector.
        Reproduces paper Figures 10, 16, 22.
        """
        import matplotlib.pyplot as plt
        from evaluation._plot_style import apply_style, save_figure

        apply_style()
        df = self.stock_report(shock_name, shock).sort_values("avg_change")

        colours = [SECTOR_COLOURS.get(s, "#8b949e") for s in df["sector"]]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(df.index, df["avg_change"], color=colours, alpha=0.88)
        ax.axvline(0, color="#8b949e", linewidth=0.7)
        ax.set_xlabel("Average Daily Price Change (Eq. 19)")
        ax.set_title(f"Average Change by Stock and Sector — {shock_name}", fontsize=11)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.grid(True, axis="x", alpha=0.4)

        self._add_sector_legend(ax, df["sector"].unique())
        fig.tight_layout()

        if save:
            fname = filename or f"avg_change_{self._slug(shock_name)}.png"
            save_figure(fig, fname)
        return fig, ax

    def plot_amplitude(
        self,
        shock_name : str,
        shock      : ShockPeriod,
        figsize    : Tuple = (12, 7),
        save       : bool  = True,
        filename   : Optional[str] = None,
    ):
        """
        Bar chart of amplitude by stock, coloured by sector.
        Reproduces paper Figures 11, 17, 23.
        """
        import matplotlib.pyplot as plt
        from evaluation._plot_style import apply_style, save_figure

        apply_style()
        df = self.stock_report(shock_name, shock).sort_values("amplitude")

        colours = [SECTOR_COLOURS.get(s, "#8b949e") for s in df["sector"]]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(df.index, df["amplitude"], color=colours, alpha=0.88)
        ax.set_xlabel("Amplitude of Returns (Eq. 20)")
        ax.set_title(f"Amplitude by Stock and Sector — {shock_name}", fontsize=11)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.grid(True, axis="x", alpha=0.4)

        self._add_sector_legend(ax, df["sector"].unique())
        fig.tight_layout()

        if save:
            fname = filename or f"amplitude_{self._slug(shock_name)}.png"
            save_figure(fig, fname)
        return fig, ax

    def plot_comparative_sectors(
        self,
        shocks   : List[ShockPeriod],
        figsize  : Tuple = (12, 7),
        save     : bool  = True,
        filename : str   = "sector_amplitude_comparison.png",
    ):
        """
        Grouped bar chart comparing sector amplitude across multiple shock
        periods. Reproduces the paper's Section IV-D cross-period comparison.
        """
        import matplotlib.pyplot as plt
        from evaluation._plot_style import apply_style, save_figure, COLOR_CYCLE

        apply_style()
        wide = self.comparative_sector_report(shocks)
        amp_cols = [c for c in wide.columns if c.endswith("_amplitude")]

        fig, ax = plt.subplots(figsize=figsize)
        n_sectors = len(wide)
        n_shocks  = len(amp_cols)
        x = np.arange(n_sectors)
        w = 0.8 / max(1, n_shocks)

        for i, col in enumerate(amp_cols):
            shock_label = col.replace("_amplitude", "")
            ax.bar(
                x + i * w - 0.4 + w/2,
                wide[col],
                width=w,
                color=COLOR_CYCLE[i % len(COLOR_CYCLE)],
                label=shock_label,
                alpha=0.88,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(wide["sector"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean Sector Amplitude")
        ax.set_title("Sector Amplitude Comparison Across Shock Periods", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.4)
        fig.tight_layout()

        if save:
            save_figure(fig, filename)
        return fig, ax

    # ─────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────

    def _add_sector_legend(self, ax, sectors) -> None:
        """Attach a colour-coded sector legend to a horizontal bar plot."""
        import matplotlib.patches as mpatches
        handles = [
            mpatches.Patch(color=SECTOR_COLOURS.get(s, "#8b949e"), label=s)
            for s in sorted(sectors)
        ]
        ax.legend(handles=handles, fontsize=6.5, loc="lower right", ncol=2)

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

    dates  = pd.date_range("2008-09-01", periods=60, freq="B")
    prices = pd.DataFrame(
        np.cumprod(1 + np.random.normal(0.0, 0.018, size=(60, N)), axis=0) * 1000,
        index=dates, columns=tickers,
    )
    log_prices = np.log(prices)
    returns    = (log_prices - log_prices.shift(1)).iloc[1:]

    shocks = [
        ShockPeriod("Global Financial Crisis", "2008-09-15", "2008-10-10",
                    18, -0.031, -0.082, 4.2, "global_financial"),
        ShockPeriod("Recovery Phase", "2008-11-01", "2008-11-20",
                    14, -0.012, -0.04, 2.1, "auto_detected"),
    ]

    analyzer = ShockMetricsAnalyzer(prices, returns)

    # stock_report
    stock_df = analyzer.stock_report("GFC", shocks[0])
    assert "amplitude" in stock_df.columns and "avg_change" in stock_df.columns
    assert len(stock_df) == N
    print(f"stock_report           OK : {stock_df.shape}")
    print(stock_df.head(3).to_string())

    # sector_report
    sector_df = analyzer.sector_report("GFC", shocks[0])
    assert "mean_amplitude" in sector_df.columns
    print(f"\nsector_report          OK : {sector_df.shape}")
    print(sector_df.head(3).to_string(index=False))

    # comparative_sector_report
    comp_df = analyzer.comparative_sector_report(shocks)
    assert "most_volatile_in" in comp_df.columns
    print(f"\ncomparative_sector_report OK : {comp_df.shape}")

    # most_sensitive_stocks
    sensitive_df = analyzer.most_sensitive_stocks(shocks, top_n=10)
    assert len(sensitive_df) == 10
    print(f"\nmost_sensitive_stocks  OK : top={sensitive_df.iloc[0]['ticker']}")

    # Plotting
    tmp = Path(tempfile.mkdtemp())
    import config.settings as cfg_mod
    orig_figs = cfg_mod.FIGURES_DIR
    cfg_mod.FIGURES_DIR = tmp
    try:
        fig1, ax1 = analyzer.plot_average_change("GFC", shocks[0], save=True)
        fig2, ax2 = analyzer.plot_amplitude("GFC", shocks[0], save=True)
        fig3, ax3 = analyzer.plot_comparative_sectors(shocks, save=True)
        saved = list(tmp.glob("*.png"))
        print(f"\nplotting               OK : {len(saved)} figures saved")
        assert len(saved) == 3
    finally:
        cfg_mod.FIGURES_DIR = orig_figs
        shutil.rmtree(tmp)

    print("\nAll shock_metrics.py tests PASSED.")