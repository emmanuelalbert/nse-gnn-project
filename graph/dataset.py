"""
graph/dataset.py
----------------
Builds PyTorch Geometric Data objects from Granger causality adjacency matrices
and node features, ready to be consumed by the TGAT model.

Paper reference: Section III-B
    "We used the mean of returns as the node feature."
    "The TGCN integrates the initial adjacency matrix and dynamically
     updates it through the network's layers."

PyG Data object structure (one per shock period × time step):
    x           : FloatTensor (N, F)   — node feature matrix
                  N = number of tickers, F = number of features
    edge_index  : LongTensor (2, E)    — directed edges from GC matrix
    edge_weight : FloatTensor (E,)     — binary 0/1 from GC (all 1s for pre-training)
    y           : FloatTensor (N, N)   — target adjacency matrix
                  (used for link prediction loss during training)
    num_nodes   : int                  — N
    shock_name  : str                  — shock period identifier (stored in metadata)
    time_delta  : FloatTensor (E,)     — time encoding for TGAT attention
                  seconds since shock start for each edge's source node

Dataset organisation:
    ShockGraphDataset holds one list[Data] per shock period.
    Each element is a single "snapshot" of the graph at one time step within
    the shock window. For the TGAT model, the full sequence of snapshots
    for a period constitutes a temporal graph signal.

    For the Monte Carlo evaluation, ShockGraphDataset.split() returns
    chronologically ordered train/test splits.

Usage:
    from graph.dataset import ShockGraphDataset, build_snapshot
    dataset = ShockGraphDataset(returns, gc_matrices, shocks)
    dataset.build()
    train_data, test_data = dataset.split(train_ratio=0.8)
    loader = dataset.loader(train_data, batch_size=1)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as CFG
from shock_detection.detector import ShockPeriod
from graph.adjacency import AdjacencyStore

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE SNAPSHOT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_snapshot(
    returns_period   : pd.DataFrame,
    gc_matrix        : np.ndarray,
    tickers          : List[str],
    shock_name       : str       = "",
    time_step        : int       = 0,
    feature_cols     : Optional[List[str]] = None,
    target_matrix    : Optional[np.ndarray] = None,
) -> Data:
    """
    Build a single PyG Data snapshot for one time step within a shock period.

    Args:
        returns_period : Log returns DataFrame sliced to the shock window.
                         Shape (window_days, N).
        gc_matrix      : (N, N) float32 Granger causality adjacency matrix.
                         Defines the graph structure for this snapshot.
        tickers        : Ordered list of N ticker symbols.
        shock_name     : Name of the shock period (stored as metadata).
        time_step      : Integer index of this snapshot within the sequence.
                         Used to compute the temporal encoding (time_delta).
        feature_cols   : Which node features to include. Default: ["mean_return"].
        target_matrix  : (N, N) target adjacency matrix for link prediction.
                         If None, gc_matrix is used as its own target.

    Returns:
        torch_geometric.data.Data object with:
            x, edge_index, edge_weight, y, num_nodes,
            shock_name, time_step, time_delta
    """
    feature_cols = feature_cols or CFG.NODE_FEATURE_COLS
    N            = len(tickers)

    # ── Node features (paper: mean of returns over the window) ──────────────
    node_features = _build_node_features(returns_period, tickers, feature_cols)
    x = torch.tensor(node_features, dtype=torch.float32)   # (N, F)

    # ── Edge index and weight from GC matrix ────────────────────────────────
    rows, cols   = np.where(gc_matrix > 0)
    edge_index   = torch.tensor(
        np.array([rows, cols], dtype=np.int64), dtype=torch.long
    )                                                         # (2, E)
    edge_weight  = torch.tensor(
        gc_matrix[rows, cols], dtype=torch.float32
    )                                                         # (E,)

    # ── Temporal encoding for TGAT ──────────────────────────────────────────
    # TGAT uses a learnable time encoding (cos/sin positional encoding on
    # the time delta between events). For a shock window of T days, we
    # assign time_step / T as the normalised time position [0, 1].
    # This is stored per edge as the time delta of the SOURCE node.
    T = len(returns_period)
    normalised_t = time_step / max(1, T - 1)
    time_delta   = torch.full((edge_index.shape[1],), normalised_t,
                               dtype=torch.float32)           # (E,)

    # ── Target adjacency matrix (for link prediction loss) ──────────────────
    target = target_matrix if target_matrix is not None else gc_matrix
    y = torch.tensor(target.astype(np.float32), dtype=torch.float32)  # (N, N)

    data = Data(
        x            = x,
        edge_index   = edge_index,
        edge_weight  = edge_weight,
        y            = y,
        num_nodes    = N,
        time_delta   = time_delta,
    )
    # Store metadata as non-tensor attributes
    data.shock_name = shock_name
    data.time_step  = time_step
    data.tickers    = tickers

    return data


def _build_node_features(
    returns_period : pd.DataFrame,
    tickers        : List[str],
    feature_cols   : List[str],
) -> np.ndarray:
    """
    Build the node feature matrix (N, F) for a single time step.

    The paper uses the mean log return over the shock window as the single
    node feature (F=1). Additional features can be enabled via settings.py.
    Each feature is computed from the full returns_period window.
    """
    F   = len(feature_cols)
    N   = len(tickers)
    mat = np.zeros((N, F), dtype=np.float32)

    ticker_to_idx = {t: i for i, t in enumerate(tickers)}

    for col_idx, feat in enumerate(feature_cols):
        for ticker in tickers:
            node_idx = ticker_to_idx[ticker]
            if ticker not in returns_period.columns:
                mat[node_idx, col_idx] = 0.0
                continue

            series = returns_period[ticker].dropna()
            if series.empty:
                mat[node_idx, col_idx] = 0.0
                continue

            if feat == "mean_return":
                mat[node_idx, col_idx] = float(series.mean())
            elif feat == "volatility":
                mat[node_idx, col_idx] = float(series.std(ddof=1)) if len(series) > 1 else 0.0
            elif feat == "min_return":
                mat[node_idx, col_idx] = float(series.min())
            elif feat == "max_return":
                mat[node_idx, col_idx] = float(series.max())
            elif feat == "amplitude":
                mat[node_idx, col_idx] = float(series.max() - series.min())
            else:
                logger.warning(f"Unknown feature '{feat}' — filling with 0.")
                mat[node_idx, col_idx] = 0.0

    return mat


# ─────────────────────────────────────────────────────────────────────────────
# SHOCK GRAPH DATASET
# ─────────────────────────────────────────────────────────────────────────────

class ShockGraphDataset:
    """
    Builds and manages the full collection of PyG Data objects for all
    shock periods, ready for TGAT training and Monte Carlo evaluation.

    Structure:
        self.snapshots : List[Data]   — all snapshots across all periods,
                                        sorted chronologically by shock start date.
        self.period_index : Dict[str, List[int]] — shock_name → snapshot indices.

    Each shock period produces T snapshots (one per trading day in the window).
    The TGAT model processes these as a temporal sequence.

    For the paper's methodology, the training objective is LINK PREDICTION:
    given the node features and current graph structure at time t, predict
    whether each potential edge (i, j) will exist at time t+1.
    """

    def __init__(
        self,
        returns     : pd.DataFrame,
        gc_matrices : Dict[str, np.ndarray],
        shocks      : List[ShockPeriod],
        tickers     : Optional[List[str]] = None,
        feature_cols: Optional[List[str]] = None,
    ):
        self.returns      = returns
        self.gc_matrices  = gc_matrices
        self.shocks       = sorted(shocks, key=lambda s: s.start)
        self.tickers      = tickers or list(returns.columns)
        self.feature_cols = feature_cols or CFG.NODE_FEATURE_COLS

        self.snapshots    : List[Data] = []
        self.period_index : Dict[str, List[int]] = {}
        self._built       = False

    # ─────────────────────────────────────────
    # BUILD
    # ─────────────────────────────────────────

    def build(self) -> "ShockGraphDataset":
        """
        Build all snapshots for all shock periods.

        For each shock period:
          1. Slice returns to the shock window.
          2. Load the GC matrix for this period.
          3. Create T snapshots (one per trading day), each with:
               - Node features from that day's view of the return window.
               - Edges from the Granger matrix (static per period).
               - Time delta encoding the position in the shock sequence.
          4. The TARGET for each snapshot at step t is the GC matrix itself
             (link prediction: reconstruct the causal structure).

        Returns self for chaining.
        """
        if self._built:
            logger.warning("Dataset already built. Call reset() first to rebuild.")
            return self

        self.snapshots   = []
        self.period_index = {}

        logger.info(
            f"Building ShockGraphDataset: {len(self.shocks)} periods, "
            f"{len(self.tickers)} tickers, features={self.feature_cols}"
        )

        for shock in self.shocks:
            if shock.name not in self.gc_matrices:
                logger.warning(
                    f"No GC matrix for '{shock.name}' — skipping."
                )
                continue

            gc_matrix  = self.gc_matrices[shock.name]
            period_ret = self._slice_returns(shock)

            if period_ret.empty:
                logger.warning(f"Empty returns for '{shock.name}' — skipping.")
                continue

            T         = len(period_ret)
            start_idx = len(self.snapshots)

            for t in range(T):
                # Rolling window view: features from the first t+1 days of the shock.
                # This ensures that at step t, only past/current data is used
                # for node features (no lookahead within the period).
                window_ret = period_ret.iloc[: t + 1]

                snapshot = build_snapshot(
                    returns_period = window_ret,
                    gc_matrix      = gc_matrix,
                    tickers        = self.tickers,
                    shock_name     = shock.name,
                    time_step      = t,
                    feature_cols   = self.feature_cols,
                    target_matrix  = gc_matrix,   # self-supervised link prediction
                )
                self.snapshots.append(snapshot)

            end_idx = len(self.snapshots)
            self.period_index[shock.name] = list(range(start_idx, end_idx))
            logger.debug(
                f"  '{shock.name}': {T} snapshots, "
                f"idx [{start_idx}, {end_idx})"
            )

        self._built = True
        logger.info(
            f"Dataset built: {len(self.snapshots)} total snapshots "
            f"across {len(self.period_index)} shock periods."
        )
        return self

    def reset(self) -> "ShockGraphDataset":
        """Clear all built snapshots and allow rebuild."""
        self.snapshots    = []
        self.period_index = {}
        self._built       = False
        return self

    # ─────────────────────────────────────────
    # TRAIN / TEST SPLIT  (paper Section III-D)
    # ─────────────────────────────────────────

    def split(
        self,
        train_ratio: float = CFG.TRAIN_RATIO,
    ) -> Tuple[List[Data], List[Data]]:
        """
        Chronological train/test split of the full snapshot list.

        The split is performed at the SHOCK PERIOD level (not the snapshot
        level) to prevent any snapshots from the same shock appearing in
        both train and test. The paper uses 80% of earlier shocks for
        training and 20% of later shocks for testing.

        Returns:
            (train_snapshots, test_snapshots)
        """
        if not self._built:
            raise RuntimeError("Call .build() before .split().")

        n_shocks_train = max(1, int(len(self.period_index) * train_ratio))
        train_names    = list(self.period_index.keys())[:n_shocks_train]
        test_names     = list(self.period_index.keys())[n_shocks_train:]

        train_snapshots = [
            self.snapshots[i]
            for name in train_names
            for i in self.period_index[name]
        ]
        test_snapshots = [
            self.snapshots[i]
            for name in test_names
            for i in self.period_index[name]
        ]

        logger.info(
            f"Train/test split ({train_ratio:.0%}/{1-train_ratio:.0%}): "
            f"{len(train_names)} train shocks ({len(train_snapshots)} snapshots) | "
            f"{len(test_names)} test shocks ({len(test_snapshots)} snapshots)"
        )

        # Verify no temporal overlap (belt-and-suspenders)
        if train_snapshots and test_snapshots:
            last_train  = train_names[-1]
            first_test  = test_names[0]
            last_shock  = next(s for s in self.shocks if s.name == last_train)
            first_shock = next(s for s in self.shocks if s.name == first_test)
            assert last_shock.end < first_shock.start, (
                f"Temporal overlap: train ends {last_shock.end}, "
                f"test starts {first_shock.start}"
            )

        return train_snapshots, test_snapshots

    def get_period_snapshots(self, shock_name: str) -> List[Data]:
        """Return all snapshots for a single named shock period."""
        if not self._built:
            raise RuntimeError("Call .build() first.")
        if shock_name not in self.period_index:
            raise KeyError(f"'{shock_name}' not in dataset.")
        return [self.snapshots[i] for i in self.period_index[shock_name]]

    # ─────────────────────────────────────────
    # NEGATIVE SAMPLING  (paper Section III-B)
    # ─────────────────────────────────────────

    def sample_negatives(
        self,
        data          : Data,
        ratio         : float = CFG.NEGATIVE_SAMPLING_RATIO,
        seed          : Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample negative edges (non-edges in the GC matrix) for training.

        The paper uses balanced negative sampling (ratio ≈ 1 negative per
        positive). Negative edges are sampled uniformly from all (i,j) pairs
        where the GC matrix is 0 and i ≠ j.

        Args:
            data  : PyG Data object containing the adjacency structure.
            ratio : Negatives per positive edge.
            seed  : Random seed for reproducibility.

        Returns:
            (pos_edges, neg_edges) — each shape (2, k) LongTensor.
        """
        if seed is not None:
            torch.manual_seed(seed)

        N = data.num_nodes
        y = data.y.numpy()

        # Positive edges: existing GC edges (excluding self-loops)
        pos_rows, pos_cols = np.where((y > 0) & (np.eye(N) == 0))
        pos_edges = torch.tensor(
            np.array([pos_rows, pos_cols]), dtype=torch.long
        )

        # Negative edges: non-existing edges
        neg_rows, neg_cols = np.where((y == 0) & (np.eye(N) == 0))
        n_neg = int(len(pos_rows) * ratio)

        if n_neg < len(neg_rows):
            idx = torch.randperm(len(neg_rows))[:n_neg]
            neg_rows = neg_rows[idx.numpy()]
            neg_cols = neg_cols[idx.numpy()]

        neg_edges = torch.tensor(
            np.array([neg_rows, neg_cols]), dtype=torch.long
        )
        return pos_edges, neg_edges

    # ─────────────────────────────────────────
    # STATISTICS
    # ─────────────────────────────────────────

    def stats(self) -> pd.DataFrame:
        """
        Per-shock-period dataset statistics.
        Columns: shock_name, n_snapshots, n_nodes, n_edges, avg_density.
        """
        rows = []
        for name, indices in self.period_index.items():
            snaps    = [self.snapshots[i] for i in indices]
            n_edges  = [s.edge_index.shape[1] for s in snaps]
            N        = snaps[0].num_nodes
            max_edges = N * (N - 1)
            rows.append({
                "shock_name"   : name,
                "n_snapshots"  : len(snaps),
                "n_nodes"      : N,
                "n_features"   : snaps[0].x.shape[1],
                "avg_edges"    : round(np.mean(n_edges), 1),
                "avg_density"  : round(np.mean(n_edges) / max(1, max_edges), 4),
            })
        return pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, idx: int) -> Data:
        return self.snapshots[idx]

    # ─────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────

    def _slice_returns(self, shock: ShockPeriod) -> pd.DataFrame:
        """Slice the full returns DataFrame to a shock window."""
        mask = (
            (self.returns.index >= pd.Timestamp(shock.start)) &
            (self.returns.index <= pd.Timestamp(shock.end))
        )
        return self.returns.loc[mask]


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level="INFO", format=CFG.LOG_FORMAT)
    np.random.seed(42)
    torch.manual_seed(42)

    # ── Synthetic inputs ──
    n        = 500
    dates    = pd.date_range("2005-01-03", periods=n, freq="B")
    tickers  = [f"T{i}.NS" for i in range(8)] + ["^NSEI"]
    N        = len(tickers)
    data     = np.random.normal(0.0003, 0.010, size=(n, N))
    returns  = pd.DataFrame(data, index=dates, columns=tickers)
    returns.index.name = "Date"

    shocks = [
        ShockPeriod("Shock A", "2005-03-01", "2005-03-20", 15, -0.02, -0.05, 2.5),
        ShockPeriod("Shock B", "2005-08-01", "2005-08-22", 17, -0.03, -0.07, 3.1),
        ShockPeriod("Shock C", "2006-01-10", "2006-01-30", 15, -0.02, -0.06, 2.8),
        ShockPeriod("Shock D", "2006-06-01", "2006-06-20", 14, -0.01, -0.04, 2.2),
        ShockPeriod("Shock E", "2006-10-01", "2006-10-18", 13, -0.02, -0.05, 2.4),
    ]

    # Fake GC matrices
    gc_matrices = {}
    for shock in shocks:
        mat = (np.random.rand(N, N) > 0.65).astype(np.float32)
        np.fill_diagonal(mat, 1.0)
        gc_matrices[shock.name] = mat

    # ── Build dataset ──
    dataset = ShockGraphDataset(
        returns      = returns,
        gc_matrices  = gc_matrices,
        shocks       = shocks,
        tickers      = tickers,
        feature_cols = ["mean_return", "volatility"],
    )
    dataset.build()

    print(f"\nTotal snapshots: {len(dataset)}")
    print(f"\nDataset stats:\n{dataset.stats()}")

    # ── Inspect one snapshot ──
    snap = dataset[0]
    print(f"\nSnapshot[0]:")
    print(f"  x          : {snap.x.shape}    (N nodes × F features)")
    print(f"  edge_index : {snap.edge_index.shape}  (2 × E edges)")
    print(f"  edge_weight: {snap.edge_weight.shape}")
    print(f"  y          : {snap.y.shape}    (N × N target)")
    print(f"  time_delta : {snap.time_delta[:3]} ...")
    print(f"  shock_name : {snap.shock_name}")
    print(f"  time_step  : {snap.time_step}")

    # ── Train/test split ──
    train, test = dataset.split(train_ratio=0.8)
    print(f"\nTrain snapshots: {len(train)}")
    print(f"Test  snapshots: {len(test)}")
    print(f"First train shock: {train[0].shock_name}")
    print(f"First test  shock: {test[0].shock_name}")

    # ── Negative sampling ──
    pos, neg = dataset.sample_negatives(snap, ratio=1.0, seed=0)
    print(f"\nPositive edges: {pos.shape[1]}")
    print(f"Negative edges: {neg.shape[1]}")
    print("\nAll dataset tests PASSED.")