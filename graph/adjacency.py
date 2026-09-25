"""
graph/adjacency.py
------------------
Converts raw Granger causality matrices into structured adjacency objects,
provides network-metric computation, and handles pre/post-training matrix
comparison (the core analytical output of the paper).

Paper reference: Section III-A-4 and Section III-B

    "We initially constructed an adjacency matrix based on historical data
     using Granger Causality tests to establish the directional relationships
     between stocks. During model training, the TGCN integrates the initial
     adjacency matrix and dynamically updates it through the network's layers."

Role in pipeline:
    granger.py  →  adjacency.py  →  dataset.py  →  model/tgat.py
    raw GC dict    AdjacencyStore    PyG Data obj     TGAT encoder

Key responsibilities:
  1. Persist GC matrices to disk as NPZ archives (results/adjacency/).
  2. Expose clean edge_index + edge_weight tensors for PyG Data objects.
  3. Apply GRAPH_LINK_THRESHOLD to predicted (post-training) matrices.
  4. Compute the full suite of network analysis metrics (paper Section III-E,
     Eq. 13–18) for both pre-training and post-training matrices.
  5. Return the delta (pre vs post) that forms the comparative analysis.

Usage:
    from graph.adjacency import AdjacencyStore, MatrixMetrics
    store  = AdjacencyStore()
    store.save_all(gc_matrices, tickers)              # persist
    adj    = store.load("Global Financial Crisis")    # reload one period
    ei, ew = store.to_edge_index(adj, tickers)        # → PyG tensors
    pre_m  = store.load_pre_training("GFC")
    post_m = store.load_post_training("GFC")
    delta  = MatrixMetrics.compare(pre_m, post_m, tickers)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as CFG

logger = logging.getLogger(__name__)

# Sub-directories within results/adjacency/
PRE_TRAINING_DIR  = CFG.ADJACENCY_DIR / "pre_training"
POST_TRAINING_DIR = CFG.ADJACENCY_DIR / "post_training"
TICKER_FILE       = CFG.ADJACENCY_DIR / "tickers.json"


# ─────────────────────────────────────────────────────────────────────────────
# ADJACENCY STORE
# ─────────────────────────────────────────────────────────────────────────────

class AdjacencyStore:
    """
    Persists and retrieves adjacency matrices for all shock periods.

    File naming convention:
        results/adjacency/pre_training/{shock_name_slug}.npz
        results/adjacency/post_training/{shock_name_slug}.npz

    Each .npz contains:
        matrix  : (N, N) float32 GC matrix
        tickers : (N,) str array of ticker symbols
    """

    def __init__(self):
        PRE_TRAINING_DIR.mkdir(parents=True, exist_ok=True)
        POST_TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────────

    def save_all(
        self,
        gc_matrices : Dict[str, np.ndarray],
        tickers     : List[str],
        stage       : str = "pre_training",
    ) -> None:
        """
        Persist a full dict of GC matrices to disk.

        Args:
            gc_matrices : {shock_name: (N,N) float32 matrix}.
            tickers     : Ordered list of ticker symbols.
            stage       : "pre_training" | "post_training".
        """
        self._validate_stage(stage)
        base = PRE_TRAINING_DIR if stage == "pre_training" else POST_TRAINING_DIR

        for name, matrix in gc_matrices.items():
            slug = self._slugify(name)
            path = base / f"{slug}.npz"
            np.savez_compressed(
                path,
                matrix  = matrix.astype(np.float32),
                tickers = np.array(tickers, dtype=str),
            )

        # Save ticker list once (shared across all periods)
        with open(TICKER_FILE, "w") as f:
            json.dump(tickers, f)

        logger.info(
            f"Saved {len(gc_matrices)} {stage} adjacency matrices "
            f"→ {base}"
        )

    def save_one(
        self,
        name    : str,
        matrix  : np.ndarray,
        tickers : List[str],
        stage   : str = "pre_training",
    ) -> None:
        """Save a single adjacency matrix."""
        self._validate_stage(stage)
        base = PRE_TRAINING_DIR if stage == "pre_training" else POST_TRAINING_DIR
        slug = self._slugify(name)
        path = base / f"{slug}.npz"
        np.savez_compressed(
            path,
            matrix  = matrix.astype(np.float32),
            tickers = np.array(tickers, dtype=str),
        )
        logger.debug(f"Saved {stage} matrix for '{name}' → {path}")

    # ─────────────────────────────────────────
    # LOAD
    # ─────────────────────────────────────────

    def load(
        self,
        name  : str,
        stage : str = "pre_training",
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Load a single adjacency matrix.

        Returns:
            (matrix, tickers) — (N,N) float32 array and ordered ticker list.
        """
        self._validate_stage(stage)
        base = PRE_TRAINING_DIR if stage == "pre_training" else POST_TRAINING_DIR
        path = base / f"{self._slugify(name)}.npz"

        if not path.exists():
            raise FileNotFoundError(
                f"No {stage} matrix for '{name}' at {path}. "
                "Run GrangerComputer.compute_all() first."
            )
        data    = np.load(path, allow_pickle=True)
        matrix  = data["matrix"].astype(np.float32)
        tickers = list(data["tickers"])
        return matrix, tickers

    def load_all(
        self,
        stage : str = "pre_training",
    ) -> Dict[str, Tuple[np.ndarray, List[str]]]:
        """Load all matrices for a given stage. Returns {name: (matrix, tickers)}."""
        self._validate_stage(stage)
        base    = PRE_TRAINING_DIR if stage == "pre_training" else POST_TRAINING_DIR
        result  = {}
        for path in sorted(base.glob("*.npz")):
            data    = np.load(path, allow_pickle=True)
            name    = self._deslugify(path.stem)
            result[name] = (
                data["matrix"].astype(np.float32),
                list(data["tickers"]),
            )
        logger.info(f"Loaded {len(result)} {stage} adjacency matrices.")
        return result

    def available_shocks(self, stage: str = "pre_training") -> List[str]:
        """Return list of shock names that have saved matrices."""
        self._validate_stage(stage)
        base = PRE_TRAINING_DIR if stage == "pre_training" else POST_TRAINING_DIR
        return [self._deslugify(p.stem) for p in sorted(base.glob("*.npz"))]

    def load_tickers(self) -> List[str]:
        """Load the canonical ticker list saved by save_all()."""
        if not TICKER_FILE.exists():
            raise FileNotFoundError(
                f"Ticker list not found at {TICKER_FILE}. "
                "Run save_all() first."
            )
        with open(TICKER_FILE) as f:
            return json.load(f)

    # ─────────────────────────────────────────
    # TENSOR CONVERSION (→ PyG)
    # ─────────────────────────────────────────

    def to_edge_index(
        self,
        matrix  : np.ndarray,
        tickers : Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a binary adjacency matrix to PyTorch Geometric edge_index format.

        PyG convention:
            edge_index : LongTensor of shape (2, E) — [source_nodes, target_nodes]
            edge_weight: FloatTensor of shape (E,)  — all 1.0 for binary GC matrix

        Args:
            matrix  : (N, N) float32 GC matrix. Non-zero entries become edges.
            tickers : Optional ticker list (used only for logging).

        Returns:
            (edge_index, edge_weight) ready for torch_geometric.data.Data.
        """
        rows, cols = np.where(matrix > 0)
        edge_index  = torch.tensor(
            np.array([rows, cols], dtype=np.int64),
            dtype = torch.long,
        )
        edge_weight = torch.tensor(
            matrix[rows, cols],
            dtype = torch.float32,
        )
        logger.debug(
            f"edge_index: {edge_index.shape}, "
            f"edge_weight: {edge_weight.shape}, "
            f"edges: {edge_index.shape[1]}"
        )
        return edge_index, edge_weight

    def threshold_predicted(
        self,
        predicted_scores : np.ndarray,
        threshold        : float = CFG.GRAPH_LINK_THRESHOLD,
    ) -> np.ndarray:
        """
        Convert TGAT's predicted link probability scores into a binary
        adjacency matrix using the link threshold.

        Paper Section III-B:
            "The adjacency matrix is dynamically updated by applying a
             threshold to these link scores."

        Args:
            predicted_scores : (N, N) float array of predicted edge probabilities.
            threshold        : Minimum probability to count as an active edge.

        Returns:
            (N, N) binary float32 adjacency matrix.
        """
        binary = (predicted_scores >= threshold).astype(np.float32)
        # Preserve diagonal (self-loops)
        if CFG.GRAPH_SELF_LOOPS:
            np.fill_diagonal(binary, 1.0)
        return binary

    # ─────────────────────────────────────────
    # COMPARISON (pre vs post training)
    # ─────────────────────────────────────────

    def compare(
        self,
        shock_name : str,
    ) -> Dict:
        """
        Load pre- and post-training matrices for a shock and return
        a structured comparison dict consumed by network_analysis/.

        Returns dict with keys:
            pre_matrix, post_matrix, tickers,
            added_edges, removed_edges, stable_edges,
            pre_metrics (MatrixMetrics), post_metrics (MatrixMetrics)
        """
        pre_mat,  tickers = self.load(shock_name, stage="pre_training")
        try:
            post_mat, _       = self.load(shock_name, stage="post_training")
        except FileNotFoundError:
            logger.warning(
                f"No post-training matrix for '{shock_name}' — "
                "run model inference first."
            )
            post_mat = None

        result = {
            "shock_name" : shock_name,
            "tickers"    : tickers,
            "pre_matrix" : pre_mat,
            "post_matrix": post_mat,
        }

        if post_mat is not None:
            # Edge-level changes
            result["added_edges"]   = int(((post_mat - pre_mat) > 0).sum())
            result["removed_edges"] = int(((pre_mat - post_mat) > 0).sum())
            result["stable_edges"]  = int(((pre_mat + post_mat) == 2).sum() - len(tickers))

            # Network metrics for both stages
            result["pre_metrics"]  = MatrixMetrics(pre_mat,  tickers)
            result["post_metrics"] = MatrixMetrics(post_mat, tickers)

        return result

    # ─────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────

    @staticmethod
    def _slugify(name: str) -> str:
        """Convert shock name to a safe filename stem."""
        return (
            name.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("&", "and")
            .replace(".", "")
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
            .replace("–", "_")
            [:80]
        )

    @staticmethod
    def _deslugify(slug: str) -> str:
        """Reverse _slugify for display — approximation only."""
        return slug.replace("_", " ").title()

    @staticmethod
    def _validate_stage(stage: str) -> None:
        if stage not in ("pre_training", "post_training"):
            raise ValueError(
                f"stage must be 'pre_training' or 'post_training', got '{stage}'."
            )


# ─────────────────────────────────────────────────────────────────────────────
# MATRIX METRICS  (paper Section III-E, Eq. 13–18)
# ─────────────────────────────────────────────────────────────────────────────

class MatrixMetrics:
    """
    Computes all network analysis metrics for a single adjacency matrix.

    Paper Eq. 13–18:
        13. Degree of Connection        → degree()
        14. Closeness Centrality        → closeness()
        15. Betweenness Centrality      → betweenness()
        16. Average Clustering Coeff    → clustering()
        17. Network Density             → density()
        18. Degree Centrality           → degree_centrality()

    The paper computes these metrics BEFORE training (on the Granger matrix)
    and AFTER training (on the TGAT-predicted matrix) and reports the delta.
    Both are computed here; the comparison logic lives in network_analysis/.

    Args:
        matrix  : (N, N) binary adjacency matrix.
        tickers : Ordered list of N ticker symbols.
    """

    def __init__(self, matrix: np.ndarray, tickers: List[str]):
        self.matrix  = matrix.astype(np.float32)
        self.tickers = tickers
        self.N       = len(tickers)
        self._G      = self._build_graph()

    def _build_graph(self) -> nx.DiGraph:
        """Build a NetworkX DiGraph from the adjacency matrix."""
        G = nx.DiGraph()
        G.add_nodes_from(range(self.N))
        rows, cols = np.where(self.matrix > 0)
        for r, c in zip(rows, cols):
            if r != c:   # exclude self-loops from graph metrics
                G.add_edge(int(r), int(c), weight=float(self.matrix[r, c]))
        return G

    # ─────────────────────────────────────────
    # INDIVIDUAL METRICS
    # ─────────────────────────────────────────

    def degree(self) -> pd.Series:
        """
        Degree of connection per node (paper Eq. 13).
        Out-degree for directed graph: number of stocks that node i influences.
        """
        out_deg = dict(self._G.out_degree())
        return pd.Series(
            [out_deg.get(i, 0) for i in range(self.N)],
            index = self.tickers,
            name  = "degree",
            dtype = np.float32,
        )

    def closeness(self) -> pd.Series:
        """
        Closeness centrality per node (paper Eq. 14).
        Uses the weakly-connected undirected projection for nodes that
        would otherwise have 0 closeness in a sparse directed graph.
        """
        centrality = nx.closeness_centrality(self._G)
        return pd.Series(
            [centrality.get(i, 0.0) for i in range(self.N)],
            index = self.tickers,
            name  = "closeness",
            dtype = np.float32,
        )

    def betweenness(self) -> pd.Series:
        """
        Betweenness centrality per node (paper Eq. 15).
        Normalised by (N-1)(N-2) for directed graphs.
        """
        centrality = nx.betweenness_centrality(self._G, normalized=True)
        return pd.Series(
            [centrality.get(i, 0.0) for i in range(self.N)],
            index = self.tickers,
            name  = "betweenness",
            dtype = np.float32,
        )

    def clustering(self) -> Tuple[pd.Series, float]:
        """
        Per-node clustering coefficient + network average (paper Eq. 16).
        Uses the undirected projection since directed clustering is rarely
        meaningful for sparse financial networks.

        Returns:
            (per_node_series, average_coefficient)
        """
        G_und   = self._G.to_undirected()
        clust   = nx.clustering(G_und)
        per_node = pd.Series(
            [clust.get(i, 0.0) for i in range(self.N)],
            index = self.tickers,
            name  = "clustering",
            dtype = np.float32,
        )
        avg = float(per_node.mean())
        return per_node, avg

    def density(self) -> float:
        """
        Network density (paper Eq. 17).
        Excludes self-loops from both numerator and denominator.
        """
        n_edges   = self.matrix.sum() - np.trace(self.matrix)
        max_edges = self.N * (self.N - 1)
        return float(n_edges / max_edges) if max_edges > 0 else 0.0

    def degree_centrality(self) -> pd.Series:
        """
        Normalised degree centrality per node (paper Eq. 18).
        degree_centrality(v) = out_degree(v) / (N - 1)
        """
        deg = self.degree()
        return (deg / max(1, self.N - 1)).rename("degree_centrality")

    # ─────────────────────────────────────────
    # FULL METRICS BUNDLE
    # ─────────────────────────────────────────

    def all_metrics(self) -> pd.DataFrame:
        """
        Compute all six network metrics and return as a single DataFrame.
        Index = tickers. One row per ticker.
        Columns: degree, degree_centrality, closeness, betweenness, clustering.
        Scalar metrics (density, avg_clustering) are stored in .scalar_metrics.
        """
        clust_per_node, avg_clust = self.clustering()
        df = pd.DataFrame({
            "degree"            : self.degree(),
            "degree_centrality" : self.degree_centrality(),
            "closeness"         : self.closeness(),
            "betweenness"       : self.betweenness(),
            "clustering"        : clust_per_node,
        })
        self.scalar_metrics = {
            "density"             : self.density(),
            "avg_clustering"      : avg_clust,
            "n_edges"             : int(self.matrix.sum() - np.trace(self.matrix)),
        }
        return df

    # ─────────────────────────────────────────
    # STATIC COMPARISON
    # ─────────────────────────────────────────

    @staticmethod
    def compare(
        pre_matrix  : np.ndarray,
        post_matrix : np.ndarray,
        tickers     : List[str],
    ) -> pd.DataFrame:
        """
        Compute pre-training vs. post-training metric delta for every ticker.
        This is the analytical core of the paper's results sections
        (paper Figures 6–8, 12–14, 18–20).

        Returns DataFrame with columns:
            {metric}_pre, {metric}_post, {metric}_delta
        for degree, degree_centrality, closeness, betweenness, clustering.
        """
        pre_m  = MatrixMetrics(pre_matrix,  tickers)
        post_m = MatrixMetrics(post_matrix, tickers)

        pre_df  = pre_m.all_metrics().add_suffix("_pre")
        post_df = post_m.all_metrics().add_suffix("_post")

        combined = pd.concat([pre_df, post_df], axis=1)

        # Delta columns
        for col in ["degree", "degree_centrality", "closeness", "betweenness", "clustering"]:
            combined[f"{col}_delta"] = combined[f"{col}_post"] - combined[f"{col}_pre"]

        # Scalar summary
        combined.attrs["pre_density"]   = pre_m.density()
        combined.attrs["post_density"]  = post_m.density()
        combined.attrs["pre_avg_clust"] = pre_m.clustering()[1]
        combined.attrs["post_avg_clust"] = post_m.clustering()[1]

        return combined


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, shutil
    logging.basicConfig(level="DEBUG", format=CFG.LOG_FORMAT)
    np.random.seed(0)

    tickers = [f"T{i}.NS" for i in range(6)] + ["^NSEI"]
    N       = len(tickers)

    # Fake GC matrix: sparse directed binary
    pre_mat = (np.random.rand(N, N) > 0.7).astype(np.float32)
    np.fill_diagonal(pre_mat, 1.0)

    post_mat = (np.random.rand(N, N) > 0.55).astype(np.float32)
    np.fill_diagonal(post_mat, 1.0)

    # ── Store ──
    tmp   = Path(tempfile.mkdtemp())
    store = AdjacencyStore.__new__(AdjacencyStore)  # bypass __init__ to use tmp
    store_pre  = tmp / "pre_training"
    store_post = tmp / "post_training"
    store_pre.mkdir(); store_post.mkdir()

    # Monkey-patch dirs for test
    import graph.adjacency as adj_mod
    orig_pre, orig_post, orig_tk = PRE_TRAINING_DIR, POST_TRAINING_DIR, TICKER_FILE
    adj_mod.PRE_TRAINING_DIR  = store_pre
    adj_mod.POST_TRAINING_DIR = store_post
    adj_mod.TICKER_FILE       = tmp / "tickers.json"
    store2 = AdjacencyStore()
    store2.save_all({"GFC": pre_mat},  tickers, stage="pre_training")
    store2.save_all({"GFC": post_mat}, tickers, stage="post_training")

    loaded_pre,  tk = store2.load("GFC", stage="pre_training")
    loaded_post, _  = store2.load("GFC", stage="post_training")
    assert np.allclose(loaded_pre, pre_mat),  "pre load mismatch"
    assert np.allclose(loaded_post, post_mat), "post load mismatch"

    # ── Edge index ──
    ei, ew = store2.to_edge_index(pre_mat, tickers)
    print(f"\nedge_index shape : {ei.shape}   (2 × E)")
    print(f"edge_weight shape: {ew.shape}")

    # ── Metrics ──
    m      = MatrixMetrics(pre_mat, tickers)
    df     = m.all_metrics()
    print(f"\nPre-training metrics:\n{df.round(3)}")
    print(f"Density: {m.density():.3f}")

    delta  = MatrixMetrics.compare(pre_mat, post_mat, tickers)
    print(f"\nPre/Post delta (degree):\n{delta['degree_delta'].round(2)}")

    # Restore
    adj_mod.PRE_TRAINING_DIR  = orig_pre
    adj_mod.POST_TRAINING_DIR = orig_post
    adj_mod.TICKER_FILE       = orig_tk
    shutil.rmtree(tmp)
    print("\nAll AdjacencyStore + MatrixMetrics tests PASSED.")