"""
visualization/network_plot.py
------------------------------
NetworkX static and Pyvis interactive directed graph visualizations
for Granger causality and TGAT-predicted adjacency networks.

Paper reference: Section III-A-4, Figure 2, Section IV-C
    "To visualize causality, we construct networks for each shock period,
     where nodes represent stocks, and directed edges indicate Granger
     causality. If stock A Granger causes stock B, a directed edge from
     A to B is established."

Four visualization types:
    1. static_graph()         — Spring-layout directed graph, nodes sized by
                                betweenness centrality, coloured by sector.
    2. pre_post_comparison()  — Two side-by-side graphs (pre vs post training)
                                showing how the TGAT changed the network.
    3. interactive_graph()    — Pyvis HTML with hover tooltips (ticker, sector,
                                centrality metrics). Analyst-facing deep-dive.
    4. ego_network()          — Ego sub-graph centred on one ticker showing
                                its direct Granger causes and effects.

Design notes:
  - Node SIZE ∝ betweenness centrality (Eq. 15) — bridge nodes appear larger.
  - Node COLOUR = sector (SECTOR_COLOURS from nifty50_universe.py).
  - ^NSEI index rendered as a diamond (⬦), larger than stock nodes.
  - Edge alpha ∝ edge weight for predicted matrices; fixed 0.5 for binary GC.
  - Spring layout with fixed seed for reproducibility; falls back to
    kamada_kawai_layout for very sparse networks where spring diverges.
  - max_edges cap prevents visual overload on dense 50×50 matrices.

Usage:
    from visualization.network_plot import NetworkPlotter
    plotter = NetworkPlotter(adjacency_store)
    plotter.static_graph("Global Financial Crisis", stage="pre_training", save=True)
    plotter.pre_post_comparison("Global Financial Crisis", save=True)
    plotter.interactive_graph("Global Financial Crisis", stage="post_training")
    plotter.ego_network("HDFCBANK.NS", "Global Financial Crisis", save=True)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as CFG
from config.nifty50_universe import SECTOR_COLOURS, get_sector_map, get_ticker_name_map
from evaluation._plot_style import apply_style, save_figure
from graph.adjacency import AdjacencyStore, MatrixMetrics

logger = logging.getLogger(__name__)

# Rendering constants
INDEX_TICKER   = "^NSEI"
SPRING_SEED    = 42
SPRING_K       = 0.8
MIN_NODE_SIZE  = 80
MAX_NODE_SIZE  = 800
EDGE_ALPHA_MIN = 0.15
EDGE_ALPHA_MAX = 0.80


class NetworkPlotter:
    """
    All network-graph visualizations for the Nifty 50 shock propagation study.

    Args:
        store : AdjacencyStore with saved pre/post adjacency matrices.
    """

    def __init__(self, store: Optional[AdjacencyStore] = None):
        self.store      = store or AdjacencyStore()
        self.sector_map = get_sector_map()
        self.name_map   = get_ticker_name_map()

    # ─────────────────────────────────────────
    # 1. STATIC GRAPH
    # ─────────────────────────────────────────

    def static_graph(
        self,
        shock_name : str,
        stage      : str   = "pre_training",
        threshold  : float = 0.0,
        max_edges  : int   = 300,
        figsize    : Tuple = (14, 10),
        save       : bool  = True,
        filename   : Optional[str] = None,
    ):
        """
        NetworkX spring-layout directed graph.

        Nodes are coloured by sector and sized by betweenness centrality.
        Top-10 betweenness nodes are labelled; all others are unlabelled
        to keep the figure readable at 50 nodes.

        Args:
            threshold : Minimum edge weight to include (binary GC = 0 keeps all
                        edges; use > 0 for predicted matrices to filter weak links).
            max_edges : Cap on rendered edges. Keeps the top-max_edges by weight.
        """
        apply_style()
        matrix, tickers = self.store.load(shock_name, stage=stage)
        G, pos           = self._build_graph(matrix, tickers, threshold, max_edges)
        m                = MatrixMetrics(matrix, tickers)
        betweenness      = m.betweenness()

        fig, ax = plt.subplots(figsize=figsize)

        node_colours, node_sizes = self._node_aesthetics(tickers, betweenness)

        # Edges
        edges   = list(G.edges(data="weight", default=1.0))
        if edges:
            weights    = np.array([w for _, _, w in edges])
            edge_alpha = np.clip(
                EDGE_ALPHA_MIN + (EDGE_ALPHA_MAX - EDGE_ALPHA_MIN) * weights,
                EDGE_ALPHA_MIN, EDGE_ALPHA_MAX,
            ).tolist()
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                edge_color=["#58a6ff"] * len(edges),
                alpha=edge_alpha,
                arrows=True, arrowsize=8,
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.05",
                width=0.6,
            )

        # Stock nodes
        stock_nodes = [i for i, t in enumerate(tickers) if t != INDEX_TICKER]
        index_nodes = [i for i, t in enumerate(tickers) if t == INDEX_TICKER]
        if stock_nodes:
            nx.draw_networkx_nodes(
                G, pos, ax=ax,
                nodelist=stock_nodes,
                node_color=[node_colours[i] for i in stock_nodes],
                node_size=[node_sizes[i]   for i in stock_nodes],
                alpha=0.90,
            )
        if index_nodes:
            nx.draw_networkx_nodes(
                G, pos, ax=ax,
                nodelist=index_nodes,
                node_color="#f8f9fa",
                node_size=MAX_NODE_SIZE,
                node_shape="D",
                alpha=0.95,
            )

        # Labels for top-10 betweenness tickers only
        top10  = betweenness.nlargest(10).index.tolist()
        labels = {i: t for i, t in enumerate(tickers) if t in top10}
        nx.draw_networkx_labels(
            G, pos, labels=labels, ax=ax,
            font_size=6.5, font_color="#c9d1d9",
        )

        density = (matrix.sum() - np.trace(matrix)) / max(1, len(tickers) * (len(tickers) - 1))
        ax.set_title(
            f"Granger Causality Network — {shock_name}\n"
            f"[{stage.replace('_', ' ').title()}]  |  "
            f"{G.number_of_nodes()} nodes · {G.number_of_edges()} edges · density={density:.3f}",
            fontsize=11, pad=8,
        )
        ax.axis("off")
        self._add_sector_legend(ax)
        fig.tight_layout()

        fname = filename or f"network_{self._slug(shock_name)}_{stage}.png"
        if save:
            save_figure(fig, fname)
        return fig, ax

    # ─────────────────────────────────────────
    # 2. PRE/POST COMPARISON
    # ─────────────────────────────────────────

    def pre_post_comparison(
        self,
        shock_name : str,
        max_edges  : int   = 200,
        figsize    : Tuple = (20, 9),
        save       : bool  = True,
        filename   : Optional[str] = None,
    ):
        """
        Side-by-side graphs: pre-training (Granger) vs post-training (TGAT).
        Shows what structural change the model learned.

        Mirrors the paper's narrative (Section IV-C-1-a, 2-a, 3-a):
        "Post-Training: There's a notable increase in connections for many
         stocks, suggesting a more interconnected market."
        """
        apply_style()
        try:
            pre_mat,  tickers = self.store.load(shock_name, stage="pre_training")
            post_mat, _       = self.store.load(shock_name, stage="post_training")
        except FileNotFoundError as exc:
            logger.warning(f"Cannot plot pre/post comparison: {exc}")
            return None, None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Compute a shared spring layout from the pre-training graph for
        # consistent visual positioning across the two panels.
        G_pre, pos = self._build_graph(pre_mat, tickers, 0.0, max_edges)

        for ax, matrix, label in [
            (ax1, pre_mat,  "Pre-Training  (Granger)"),
            (ax2, post_mat, "Post-Training (TGAT Predicted)"),
        ]:
            G, _ = self._build_graph(matrix, tickers, 0.0, max_edges)
            m    = MatrixMetrics(matrix, tickers)
            node_colours, node_sizes = self._node_aesthetics(tickers, m.betweenness())

            edges = list(G.edges())
            if edges:
                nx.draw_networkx_edges(
                    G, pos, ax=ax,
                    edge_color="#58a6ff", alpha=0.40,
                    arrows=True, arrowsize=7,
                    connectionstyle="arc3,rad=0.05", width=0.5,
                )
            stock_nodes = [i for i, t in enumerate(tickers) if t != INDEX_TICKER]
            nx.draw_networkx_nodes(
                G, pos, ax=ax, nodelist=stock_nodes,
                node_color=[node_colours[i] for i in stock_nodes],
                node_size=[node_sizes[i]   for i in stock_nodes],
                alpha=0.88,
            )
            top5   = m.betweenness().nlargest(5).index.tolist()
            labels = {i: t for i, t in enumerate(tickers) if t in top5}
            nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                                    font_size=6, font_color="#c9d1d9")
            ax.set_title(
                f"{label}\n"
                f"{G.number_of_edges()} edges · density={m.density():.3f}",
                fontsize=10,
            )
            ax.axis("off")

        self._add_sector_legend(ax2, loc="lower left")
        fig.suptitle(
            f"Network Structure Comparison — {shock_name}",
            fontsize=12, y=1.01,
        )
        fig.tight_layout()

        fname = filename or f"network_prepost_{self._slug(shock_name)}.png"
        if save:
            save_figure(fig, fname)
        return fig, (ax1, ax2)

    # ─────────────────────────────────────────
    # 3. INTERACTIVE GRAPH  (Pyvis HTML)
    # ─────────────────────────────────────────

    def interactive_graph(
        self,
        shock_name : str,
        stage      : str   = "post_training",
        threshold  : float = 0.3,
        max_edges  : int   = 400,
        filename   : Optional[str] = None,
    ) -> Path:
        """
        Generate a Pyvis HTML interactive network graph.

        Hovering a node reveals: ticker, company name, sector, degree,
        betweenness, closeness. Edge thickness ∝ edge weight.

        Args:
            threshold : Minimum edge weight for display (useful for filtering
                        low-confidence predicted edges from TGAT).

        Returns:
            Path to the saved .html file.
        """
        from pyvis.network import Network

        matrix, tickers = self.store.load(shock_name, stage=stage)
        m               = MatrixMetrics(matrix, tickers)
        betweenness     = m.betweenness()
        degree          = m.degree()
        closeness       = m.closeness()
        N               = len(tickers)

        # Apply threshold and self-loop removal
        display_mat = matrix.copy()
        display_mat[display_mat < threshold] = 0.0
        np.fill_diagonal(display_mat, 0.0)

        # Cap edges
        rows, cols = np.where(display_mat > 0)
        weights    = display_mat[rows, cols]
        if len(weights) > max_edges:
            top_idx = np.argsort(weights)[::-1][:max_edges]
            rows, cols, weights = rows[top_idx], cols[top_idx], weights[top_idx]

        net = Network(
            height="750px", width="100%",
            bgcolor="#ffffff", font_color="#1f2328",
            directed=True,
        )
        net.set_options("""
        {
          "physics": {
            "barnesHut": {"gravitationalConstant": -8000, "springLength": 120}
          },
          "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.6}},
            "smooth": {"type": "curvedCW", "roundness": 0.1}
          },
          "nodes": {"shape": "dot", "font": {"size": 10}}
        }
        """)

        for i, ticker in enumerate(tickers):
            sector    = self.sector_map.get(ticker, "Unknown")
            colour    = SECTOR_COLOURS.get(sector, "#8b949e")
            b_val     = float(betweenness.iloc[i])
            node_size = max(8, int(10 + 30 * min(1.0, b_val * 10)))
            title = (
                f"<b>{ticker}</b><br>"
                f"Name: {self.name_map.get(ticker, ticker)}<br>"
                f"Sector: {sector}<br>"
                f"Degree: {int(degree.iloc[i])}<br>"
                f"Betweenness: {b_val:.4f}<br>"
                f"Closeness: {float(closeness.iloc[i]):.4f}"
            )
            net.add_node(
                i, label=ticker, title=title, color=colour,
                size=node_size,
                shape="diamond" if ticker == INDEX_TICKER else "dot",
            )

        for r, c, w in zip(rows, cols, weights):
            net.add_edge(
                int(r), int(c), value=float(w),
                title=f"GC weight: {w:.3f}",
                color={"opacity": float(np.clip(w, 0.2, 0.9))},
            )

        save_dir = Path(CFG.FIGURES_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)
        fname    = filename or f"network_interactive_{self._slug(shock_name)}_{stage}.html"
        out_path = save_dir / fname
        net.save_graph(str(out_path))

        logger.info(
            f"Interactive graph → {out_path} "
            f"({len(tickers)} nodes, {len(rows)} edges)"
        )
        return out_path

    # ─────────────────────────────────────────
    # 4. EGO NETWORK
    # ─────────────────────────────────────────

    def ego_network(
        self,
        ticker     : str,
        shock_name : str,
        stage      : str  = "post_training",
        radius     : int  = 1,
        figsize    : Tuple = (10, 8),
        save       : bool  = True,
        filename   : Optional[str] = None,
    ):
        """
        Directed ego-network centred on a single ticker.
        Shows all stocks that Granger-cause or are Granger-caused by
        the target ticker within `radius` hops.

        Useful for the paper's stock-level narrative:
        "HDFCBANK showed high betweenness — its ego-network reveals
         direct causal links to stocks across 5 sectors."

        Args:
            ticker : Must be present in the adjacency matrix (e.g. "HDFCBANK.NS").
            radius : 1 = immediate causes/effects; 2 = two-hop neighbourhood.
        """
        apply_style()
        matrix, tickers = self.store.load(shock_name, stage=stage)

        if ticker not in tickers:
            raise ValueError(
                f"'{ticker}' not found in adjacency matrix for '{shock_name}'. "
                f"Available tickers: {tickers[:5]} ..."
            )

        G_full, _  = self._build_graph(matrix, tickers, threshold=0.0, max_edges=9999)
        ego_idx    = tickers.index(ticker)
        m          = MatrixMetrics(matrix, tickers)
        betweenness = m.betweenness()

        # Ego graph in directed sense: include both in-edges and out-edges
        ego_out = nx.ego_graph(G_full, ego_idx, radius=radius, undirected=False)
        ego_und = nx.ego_graph(G_full.to_undirected(), ego_idx, radius=radius)
        for n in ego_und.nodes():
            ego_out.add_node(n)
        for u, v in ego_und.edges():
            if G_full.has_edge(u, v):
                ego_out.add_edge(u, v, **G_full[u][v])
            if G_full.has_edge(v, u):
                ego_out.add_edge(v, u, **G_full[v][u])

        pos = nx.spring_layout(ego_out, seed=SPRING_SEED, k=1.2)

        node_colours, node_sizes = self._node_aesthetics(tickers, betweenness)
        ego_colours = [node_colours[n] for n in ego_out.nodes()]
        ego_sizes   = [node_sizes[n]   for n in ego_out.nodes()]

        # Highlight the ego node in orange
        for idx, n in enumerate(ego_out.nodes()):
            if n == ego_idx:
                ego_colours[idx] = "#f0883e"
                ego_sizes[idx]   = MAX_NODE_SIZE

        fig, ax = plt.subplots(figsize=figsize)

        if ego_out.number_of_edges() > 0:
            nx.draw_networkx_edges(
                ego_out, pos, ax=ax,
                edge_color="#58a6ff", alpha=0.60,
                arrows=True, arrowsize=14,
                connectionstyle="arc3,rad=0.05", width=0.9,
            )
        nx.draw_networkx_nodes(
            ego_out, pos, ax=ax,
            node_color=ego_colours, node_size=ego_sizes, alpha=0.92,
        )
        labels = {n: tickers[n] for n in ego_out.nodes()}
        nx.draw_networkx_labels(
            ego_out, pos, labels=labels, ax=ax,
            font_size=7.5, font_color="#c9d1d9",
        )

        sector = self.sector_map.get(ticker, "Unknown")
        ax.set_title(
            f"Ego Network — {ticker}  [{sector}]\n"
            f"{shock_name} [{stage.replace('_', ' ').title()}]  |  "
            f"radius={radius}  |  "
            f"{ego_out.number_of_nodes()} nodes · {ego_out.number_of_edges()} edges",
            fontsize=11,
        )
        ax.axis("off")
        fig.tight_layout()

        safe_ticker = ticker.replace(".", "_").replace("^", "")
        fname = filename or f"ego_{safe_ticker}_{self._slug(shock_name)}.png"
        if save:
            save_figure(fig, fname)
        return fig, ax

    # ─────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────

    def _build_graph(
        self,
        matrix    : np.ndarray,
        tickers   : List[str],
        threshold : float,
        max_edges : int,
    ) -> Tuple[nx.DiGraph, Dict]:
        """Build a NetworkX DiGraph, cap edges, compute spring layout."""
        N = len(tickers)
        G = nx.DiGraph()
        G.add_nodes_from(range(N))

        rows, cols = np.where((matrix > threshold) & (np.eye(N) == 0))
        weights    = matrix[rows, cols]

        if len(weights) > max_edges:
            top_idx    = np.argsort(weights)[::-1][:max_edges]
            rows, cols = rows[top_idx], cols[top_idx]
            weights    = weights[top_idx]

        for r, c, w in zip(rows, cols, weights):
            G.add_edge(int(r), int(c), weight=float(w))

        try:
            pos = nx.spring_layout(G, seed=SPRING_SEED, k=SPRING_K)
        except Exception:
            pos = nx.kamada_kawai_layout(G)

        return G, pos

    def _node_aesthetics(
        self,
        tickers     : List[str],
        betweenness : pd.Series,
    ) -> Tuple[List[str], List[float]]:
        """Return per-node colour (by sector) and size (by betweenness centrality)."""
        colours = [
            SECTOR_COLOURS.get(self.sector_map.get(t, "Unknown"), "#8b949e")
            for t in tickers
        ]
        b_vals = betweenness.values
        b_norm = (b_vals - b_vals.min()) / max(1e-9, b_vals.max() - b_vals.min())
        sizes  = (MIN_NODE_SIZE + (MAX_NODE_SIZE - MIN_NODE_SIZE) * b_norm).tolist()
        return colours, sizes

    def _add_sector_legend(self, ax, loc: str = "lower right") -> None:
        """Attach colour-coded sector legend."""
        handles = [
            mpatches.Patch(facecolor=colour, label=sector)
            for sector, colour in SECTOR_COLOURS.items()
            if sector != "Index"
        ]
        ax.legend(
            handles=handles, fontsize=6, loc=loc,
            ncol=2, framealpha=0.35, labelcolor="#1f2328",
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
    np.random.seed(7)

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
        store = AdjacencyStore()
        pre_mat  = (np.random.rand(N, N) > 0.65).astype(np.float32)
        np.fill_diagonal(pre_mat, 1.0)
        post_mat = (np.random.rand(N, N) > 0.55).astype(np.float32)
        np.fill_diagonal(post_mat, 1.0)
        store.save_all({"GFC": pre_mat},  tickers, stage="pre_training")
        store.save_all({"GFC": post_mat}, tickers, stage="post_training")

        import config.settings as cfg_mod
        orig_figs = cfg_mod.FIGURES_DIR
        cfg_mod.FIGURES_DIR = tmp / "figs"
        (tmp / "figs").mkdir()

        try:
            plotter = NetworkPlotter(store)

            fig1, ax1 = plotter.static_graph("GFC", stage="pre_training", save=True)
            assert fig1 is not None
            print("static_graph           OK")

            fig2, axes = plotter.pre_post_comparison("GFC", save=True)
            assert fig2 is not None
            print("pre_post_comparison    OK")

            html_path = plotter.interactive_graph("GFC", stage="pre_training")
            assert html_path.exists() and html_path.suffix == ".html"
            print(f"interactive_graph      OK : {html_path.name}")

            fig3, ax3 = plotter.ego_network(tickers[0], "GFC",
                                             stage="pre_training", save=True)
            assert fig3 is not None
            print(f"ego_network            OK : {tickers[0]}")

            pngs  = list((tmp / "figs").glob("*.png"))
            htmls = list((tmp / "figs").glob("*.html"))
            print(f"Figures saved          : {len(pngs)} PNG + {len(htmls)} HTML")
            assert len(pngs) == 3 and len(htmls) == 1

        finally:
            cfg_mod.FIGURES_DIR = orig_figs

        print("\nAll network_plot.py tests PASSED.")

    finally:
        adj_mod.PRE_TRAINING_DIR  = orig_pre
        adj_mod.POST_TRAINING_DIR = orig_post
        adj_mod.TICKER_FILE       = orig_tk
        shutil.rmtree(tmp)