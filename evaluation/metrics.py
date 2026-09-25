"""
evaluation/metrics.py
---------------------
Comprehensive evaluation metrics for the TGAT shock-propagation model.

Paper reference: Section III-C (Eq. 6–10) and Section III-D (Eq. 11–12)

    Three primary metrics used throughout the paper:

        AUC-ROC (Eq. 6):
            AUC = ∫₀¹ TPR(t) dt
            Measures the model's ability to rank positive edges above negative
            ones. An AUC of 0.5 is random; 1.0 is perfect.

        Average Precision / AUPR (Eq. 7):
            AP = Σₙ (Rₙ - Rₙ₋₁) · Pₙ
            Area under the Precision-Recall curve. More informative than
            AUC-ROC when the positive class (GC edges) is in the minority.

        F1 Score (Eq. 8–10):
            F1 = 2·P·R / (P + R)
            where  P = TP / (TP + FP)   [Eq. 9]
                   R = TP / (TP + FN)   [Eq. 10]

    Monte Carlo aggregation (Eq. 11–12):
            μ = (1/N) Σᵢ Metricᵢ
            σ = √( (1/N) Σᵢ (Metricᵢ - μ)² )     [population std]

Paper results (Table 3) to reproduce:
    AUC-ROC : 0.77 ± low σ
    AUPR    : 0.76 ± low σ
    F1      : 0.79 ± low σ
    Low-FPR AUC-ROC (0–0.1) : 0.77

Role in the pipeline:
    model/loss.py    → compute_metrics()           (raw per-snapshot metrics)
    evaluation/metrics.py → EvaluationSuite        (structured multi-level eval)
      ├── per_snapshot_metrics()   evaluate one Data object
      ├── per_shock_metrics()      evaluate all snapshots of one shock period
      ├── full_eval()              evaluate all shocks in a dataset
      ├── mc_report()              aggregate across MC folds (Eq. 11–12)
      └── comparative_report()    compare metric shifts across the 4 target shocks

Usage:
    from evaluation.metrics import EvaluationSuite
    suite = EvaluationSuite(model, dataset, shocks, device="cpu")
    report = suite.full_eval()
    suite.mc_report(fold_results)
    suite.print_summary(report)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as CFG
from config.nifty50_universe import get_sector_map
from graph.dataset import ShockGraphDataset
from model.loss import aggregate_monte_carlo_metrics, compute_metrics
from model.tgat import TGAT
from shock_detection.detector import ShockPeriod
from training.monte_carlo import FoldResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION SUITE
# ─────────────────────────────────────────────────────────────────────────────

class EvaluationSuite:
    """
    Structured evaluation of the TGAT model at three levels of granularity:
        1. Per-snapshot  — one PyG Data object = one time step in one shock period
        2. Per-shock     — aggregate all snapshots in one shock period
        3. Full          — aggregate all shocks in a dataset (the primary report)

    Also provides Monte Carlo aggregation (Eq. 11-12) and a comparative
    analysis across the four target shock periods.

    Args:
        model   : Trained TGAT model (best-checkpoint weights loaded).
        dataset : Built ShockGraphDataset with all shock periods.
        shocks  : List of ShockPeriod objects matching the dataset.
        device  : Compute device.
    """

    def __init__(
        self,
        model   : TGAT,
        dataset : ShockGraphDataset,
        shocks  : List[ShockPeriod],
        device  : str = CFG.DEVICE,
    ):
        self.model   = model.to(device).eval()
        self.dataset = dataset
        self.shocks  = {s.name: s for s in shocks}
        self.device  = device
        self.tickers = dataset.tickers
        self.sector_map = get_sector_map()

    # ─────────────────────────────────────────
    # LEVEL 1: PER-SNAPSHOT
    # ─────────────────────────────────────────

    @torch.no_grad()
    def per_snapshot_metrics(
        self,
        snap        : object,
        x_seq       : torch.Tensor,
        time_deltas : torch.Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate one snapshot.

        Args:
            snap        : PyG Data object with x, edge_index, y, time_delta.
            x_seq       : (T, N, F) temporal sequence up to this snapshot.
            time_deltas : (T,) normalised time positions.

        Returns:
            Metrics dict: auc_roc, aupr, f1, precision, recall,
                          auc_roc_low_fpr, n_pos, n_neg, density.
        """
        snap_dev = snap.to(self.device)
        probs    = self.model.predict_adjacency(
            x_seq, snap_dev.edge_index, time_deltas,
            getattr(snap_dev, "edge_weight", None),
            apply_sigmoid=True,
        )

        N    = probs.shape[0]
        mask = ~torch.eye(N, dtype=torch.bool, device=self.device)

        y_true  = snap_dev.y[mask].cpu().numpy().astype(float)
        y_score = probs[mask].cpu().numpy().astype(float)

        return compute_metrics(y_true, y_score)

    # ─────────────────────────────────────────
    # LEVEL 2: PER-SHOCK-PERIOD
    # ─────────────────────────────────────────

    @torch.no_grad()
    def per_shock_metrics(
        self,
        shock_name : str,
    ) -> Dict[str, float]:
        """
        Evaluate all snapshots belonging to one shock period and aggregate.

        Returns a single metrics dict representing the model's performance
        on that shock period (all time steps pooled).
        """
        if shock_name not in self.dataset.period_index:
            raise KeyError(f"'{shock_name}' not in dataset.")

        snaps = self.dataset.get_period_snapshots(shock_name)

        all_y_true  : List[np.ndarray] = []
        all_y_scores: List[np.ndarray] = []

        for t_idx, snap in enumerate(snaps):
            snap_dev    = snap.to(self.device)
            x_seq, dt   = self._build_sequence(snaps[: t_idx + 1])

            probs = self.model.predict_adjacency(
                x_seq, snap_dev.edge_index, dt,
                getattr(snap_dev, "edge_weight", None),
                apply_sigmoid=True,
            )

            N    = probs.shape[0]
            mask = ~torch.eye(N, dtype=torch.bool, device=self.device)
            all_y_true.append(snap_dev.y[mask].cpu().numpy().astype(float))
            all_y_scores.append(probs[mask].cpu().numpy().astype(float))

        y_true_all   = np.concatenate(all_y_true)
        y_scores_all = np.concatenate(all_y_scores)

        metrics = compute_metrics(y_true_all, y_scores_all)
        logger.debug(
            f"  {shock_name}: AUC={metrics['auc_roc']:.4f} "
            f"AUPR={metrics['aupr']:.4f} F1={metrics['f1']:.4f}"
        )
        return metrics

    # ─────────────────────────────────────────
    # LEVEL 3: FULL EVALUATION (all shocks)
    # ─────────────────────────────────────────

    @torch.no_grad()
    def full_eval(
        self,
        shock_names : Optional[List[str]] = None,
    ) -> Dict:
        """
        Run per-shock evaluation for all (or selected) shocks and return
        a structured report.

        Args:
            shock_names : Optional list of shock names to evaluate.
                          None → evaluate all shocks in the dataset.

        Returns:
            report dict:
            {
                "per_shock"  : {shock_name: metrics_dict},
                "overall"    : aggregated metrics (pooled across all shocks),
                "summary_df" : pd.DataFrame (one row per shock),
            }
        """
        names     = shock_names or list(self.dataset.period_index.keys())
        per_shock : Dict[str, Dict] = {}

        # Pool all predictions for the overall metrics
        all_y_true  : List[np.ndarray] = []
        all_y_scores: List[np.ndarray] = []

        logger.info(f"EvaluationSuite.full_eval() on {len(names)} shock periods")

        for name in names:
            if name not in self.dataset.period_index:
                logger.warning(f"  Skipping '{name}' — not in dataset.")
                continue

            snaps = self.dataset.get_period_snapshots(name)

            for t_idx, snap in enumerate(snaps):
                snap_dev  = snap.to(self.device)
                x_seq, dt = self._build_sequence(snaps[: t_idx + 1])

                probs = self.model.predict_adjacency(
                    x_seq, snap_dev.edge_index, dt,
                    getattr(snap_dev, "edge_weight", None),
                    apply_sigmoid=True,
                )

                N    = probs.shape[0]
                mask = ~torch.eye(N, dtype=torch.bool, device=self.device)
                all_y_true.append(snap_dev.y[mask].cpu().numpy().astype(float))
                all_y_scores.append(probs[mask].cpu().numpy().astype(float))

            per_shock[name] = self.per_shock_metrics(name)

        overall = (
            compute_metrics(
                np.concatenate(all_y_true),
                np.concatenate(all_y_scores),
            )
            if all_y_true else {}
        )

        summary_df = self._build_shock_summary_df(per_shock)

        logger.info(
            f"Overall | AUC-ROC={overall.get('auc_roc','?'):.4f} "
            f"AUPR={overall.get('aupr','?'):.4f} "
            f"F1={overall.get('f1','?'):.4f}"
        )

        return {
            "per_shock"  : per_shock,
            "overall"    : overall,
            "summary_df" : summary_df,
        }

    # ─────────────────────────────────────────
    # MONTE CARLO AGGREGATION  (Eq. 11–12)
    # ─────────────────────────────────────────

    @staticmethod
    def mc_report(
        fold_results    : List[FoldResult],
        save_path       : Optional[Path] = None,
        print_table     : bool = True,
    ) -> Dict:
        """
        Aggregate per-fold metrics across all Monte Carlo folds.

        Implements paper Eq. 11 (μ) and Eq. 12 (σ).

        Paper results to beat:
            AUC-ROC : 0.77 ± σ_small
            AUPR    : 0.76 ± σ_small
            F1      : 0.79 ± σ_small

        Args:
            fold_results : List of FoldResult objects from MonteCarloRunner.run().
            save_path    : If provided, write the report JSON to this path.
            print_table  : If True, print a formatted table to stdout.

        Returns:
            Aggregated metrics dict: {metric: {"mean", "std", "all"}}.
        """
        metrics_list = [
            {
                "auc_roc"         : r.auc_roc,
                "aupr"            : r.aupr,
                "f1"              : r.f1,
                "precision"       : r.precision,
                "recall"          : r.recall,
                "auc_roc_low_fpr" : r.auc_roc_low_fpr,
            }
            for r in fold_results
        ]
        agg = aggregate_monte_carlo_metrics(metrics_list)

        if print_table:
            EvaluationSuite._print_mc_table(agg, n_folds=len(fold_results))

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(agg, f, indent=2)
            logger.info(f"MC report saved → {save_path}")

        return agg

    @staticmethod
    def _print_mc_table(agg: Dict, n_folds: int = 10) -> None:
        """Print the paper-style Table 3 to stdout."""
        LABELS = {
            "auc_roc"         : "AUC-ROC         (Eq. 6)",
            "aupr"            : "Avg Precision   (Eq. 7)",
            "f1"              : "F1 Score        (Eq. 8)",
            "precision"       : "Precision       (Eq. 9)",
            "recall"          : "Recall          (Eq.10)",
            "auc_roc_low_fpr" : "AUC-ROC low FPR [0,0.1]",
        }
        PAPER = {
            "auc_roc": 0.77, "aupr": 0.76, "f1": 0.79,
            "auc_roc_low_fpr": 0.77,
        }

        print("\n" + "═" * 66)
        print(f"  MONTE CARLO EVALUATION REPORT  ({n_folds} folds)")
        print("═" * 66)
        print(f"  {'Metric':<30}  {'μ':>7}  {'σ':>7}  {'Paper':>7}")
        print("─" * 66)
        for key, label in LABELS.items():
            if key not in agg:
                continue
            v = agg[key]
            paper_val = PAPER.get(key, "—")
            paper_str = f"{paper_val:.4f}" if isinstance(paper_val, float) else paper_val
            print(f"  {label:<30}  {v['mean']:>7.4f}  {v['std']:>7.4f}  {paper_str:>7}")
        print("─" * 66)
        print(f"  {'Folds':<30}  {n_folds:>7}")
        print("═" * 66)

    # ─────────────────────────────────────────
    # COMPARATIVE ANALYSIS (target shocks)
    # ─────────────────────────────────────────

    def comparative_report(
        self,
        target_shock_names : Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compare model performance across the four target shock periods.

        Paper Section IV-C: "Analyzing three distinct shock periods in the
        stock market, each characterized by unique global events, offering
        a diverse perspective on market dynamics."

        Returns a DataFrame with one row per target shock, columns:
            shock_name, type, duration, avg_return, sigma_breach,
            auc_roc, aupr, f1, n_pos, n_neg, density
        """
        target_names = target_shock_names or [
            t["name"] for t in CFG.TARGET_SHOCK_PERIODS
            if t["name"] in self.dataset.period_index
        ]

        if not target_names:
            logger.warning(
                "No target shock periods found in dataset. "
                "Run ShockDetector.match_target_periods() before building the dataset."
            )
            return pd.DataFrame()

        rows = []
        for name in target_names:
            if name not in self.dataset.period_index:
                logger.warning(f"  '{name}' not in dataset — skipping.")
                continue

            metrics = self.per_shock_metrics(name)
            shock   = self.shocks.get(name)

            row = {"shock_name": name}
            if shock:
                row.update({
                    "type"         : shock.type,
                    "start"        : shock.start,
                    "end"          : shock.end,
                    "duration"     : shock.duration,
                    "avg_return"   : shock.avg_return,
                    "sigma_breach" : shock.sigma_breach,
                })
            row.update({
                "auc_roc"    : metrics.get("auc_roc", 0),
                "aupr"       : metrics.get("aupr", 0),
                "f1"         : metrics.get("f1", 0),
                "n_pos"      : metrics.get("n_pos", 0),
                "n_neg"      : metrics.get("n_neg", 0),
                "density"    : metrics.get("density", 0),
            })
            rows.append(row)

        df = pd.DataFrame(rows)
        logger.info(
            f"Comparative report: {len(df)} target shocks evaluated."
        )
        return df

    # ─────────────────────────────────────────
    # SECTOR-LEVEL METRICS
    # ─────────────────────────────────────────

    @torch.no_grad()
    def sector_metrics(
        self,
        shock_name : str,
    ) -> pd.DataFrame:
        """
        Compute per-sector edge-prediction accuracy for one shock period.

        For each sector, evaluates how well the model predicts the GC edges
        ORIGINATING from that sector's stocks (out-edges from sector nodes).

        Returns DataFrame with columns:
            sector, n_tickers, auc_roc, aupr, f1, n_edges_predicted,
            n_edges_true, edge_precision
        """
        if shock_name not in self.dataset.period_index:
            raise KeyError(f"'{shock_name}' not in dataset.")

        snaps   = self.dataset.get_period_snapshots(shock_name)
        tickers = self.tickers
        N       = len(tickers)

        # Accumulate per-ticker prediction arrays
        ticker_y_true  = {t: [] for t in tickers}
        ticker_y_score = {t: [] for t in tickers}

        for t_idx, snap in enumerate(snaps):
            snap_dev  = snap.to(self.device)
            x_seq, dt = self._build_sequence(snaps[: t_idx + 1])

            probs = self.model.predict_adjacency(
                x_seq, snap_dev.edge_index, dt,
                getattr(snap_dev, "edge_weight", None),
                apply_sigmoid=True,
            )   # (N, N)

            y = snap_dev.y.cpu().numpy()   # (N, N)
            p = probs.cpu().numpy()

            for i, ticker in enumerate(tickers):
                # Out-edges from ticker i to all j ≠ i
                row_mask = np.arange(N) != i
                ticker_y_true[ticker].extend(y[i, row_mask].tolist())
                ticker_y_score[ticker].extend(p[i, row_mask].tolist())

        # Aggregate to sector level
        sector_rows = []
        sectors     = sorted(set(self.sector_map.get(t, "Unknown") for t in tickers
                                 if t != "^NSEI"))

        for sector in sectors:
            sector_tickers = [
                t for t in tickers
                if self.sector_map.get(t) == sector
            ]
            if not sector_tickers:
                continue

            y_true_sector  = np.concatenate([ticker_y_true[t]  for t in sector_tickers])
            y_score_sector = np.concatenate([ticker_y_score[t] for t in sector_tickers])

            m = compute_metrics(y_true_sector, y_score_sector)
            sector_rows.append({
                "sector"             : sector,
                "n_tickers"          : len(sector_tickers),
                "auc_roc"            : m["auc_roc"],
                "aupr"               : m["aupr"],
                "f1"                 : m["f1"],
                "n_edges_true"       : m["n_pos"],
                "n_edges_predicted"  : int((y_score_sector >= 0.5).sum()),
                "edge_precision"     : m["precision"],
            })

        if not sector_rows:
            logger.warning(
                f"sector_metrics('{shock_name}'): no sector produced valid metrics "
                "(likely too few tickers per sector in this dataset)."
            )
            return pd.DataFrame(columns=[
                "sector", "n_tickers", "auc_roc", "aupr", "f1",
                "n_edges_true", "n_edges_predicted", "edge_precision",
            ])

        return (
            pd.DataFrame(sector_rows)
            .sort_values("auc_roc", ascending=False)
            .reset_index(drop=True)
        )

    # ─────────────────────────────────────────
    # THRESHOLD SENSITIVITY ANALYSIS
    # ─────────────────────────────────────────

    @torch.no_grad()
    def threshold_sweep(
        self,
        shock_name  : str,
        thresholds  : Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Compute F1, Precision, Recall at multiple decision thresholds.
        Useful for identifying the optimal operating threshold beyond 0.5.

        Args:
            shock_name : Shock period name.
            thresholds : Array of threshold values ∈ (0, 1).
                         Defaults to 50 linearly-spaced values.

        Returns:
            DataFrame with columns: threshold, f1, precision, recall, n_pos_predicted.
        """
        if thresholds is None:
            thresholds = np.linspace(0.05, 0.95, 50)

        snaps = self.dataset.get_period_snapshots(shock_name)
        all_y_true   = []
        all_y_scores = []

        for t_idx, snap in enumerate(snaps):
            snap_dev  = snap.to(self.device)
            x_seq, dt = self._build_sequence(snaps[: t_idx + 1])
            probs = self.model.predict_adjacency(
                x_seq, snap_dev.edge_index, dt,
                getattr(snap_dev, "edge_weight", None), apply_sigmoid=True,
            )
            N    = probs.shape[0]
            mask = ~torch.eye(N, dtype=torch.bool, device=self.device)
            all_y_true.append(snap_dev.y[mask].cpu().numpy().astype(float))
            all_y_scores.append(probs[mask].cpu().numpy().astype(float))

        y_true   = np.concatenate(all_y_true)
        y_scores = np.concatenate(all_y_scores)

        rows = []
        for thr in thresholds:
            m = compute_metrics(y_true, y_scores, threshold=float(thr))
            rows.append({
                "threshold"        : round(float(thr), 3),
                "f1"               : m["f1"],
                "precision"        : m["precision"],
                "recall"           : m["recall"],
                "n_pos_predicted"  : int((y_scores >= thr).sum()),
            })

        df = pd.DataFrame(rows)
        best_row = df.loc[df["f1"].idxmax()]
        logger.info(
            f"Threshold sweep '{shock_name}': "
            f"best F1={best_row['f1']:.4f} at threshold={best_row['threshold']:.3f}"
        )
        return df

    # ─────────────────────────────────────────
    # REPORTING
    # ─────────────────────────────────────────

    def print_summary(self, report: Dict) -> None:
        """Print the full_eval() report in a readable format."""
        overall = report.get("overall", {})
        df      = report.get("summary_df", pd.DataFrame())

        print("\n" + "═" * 60)
        print("  TGAT EVALUATION SUMMARY")
        print("═" * 60)
        print("  Overall (all shocks pooled):")
        for key in ["auc_roc", "aupr", "f1", "precision", "recall", "auc_roc_low_fpr"]:
            val = overall.get(key)
            if val is not None:
                print(f"    {key:<22} {val:.4f}")
        print()
        if not df.empty:
            print("  Per-shock breakdown:")
            cols = [c for c in ["shock_name", "auc_roc", "aupr", "f1", "n_pos", "density"]
                    if c in df.columns]
            print("  " + df[cols].to_string(index=False).replace("\n", "\n  "))
        print("═" * 60)

    def save_report(self, report: Dict, path: Path) -> None:
        """Save the full_eval() report to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "overall"   : report.get("overall", {}),
            "per_shock" : report.get("per_shock", {}),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info(f"Evaluation report saved → {path}")

    # ─────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────

    def _build_sequence(
        self,
        snaps: List,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build (x_seq, time_deltas) from a list of snapshots."""
        x_list  = [s.x.to(self.device) for s in snaps]
        x_seq   = torch.stack(x_list, dim=0)
        dt_list = [s.time_delta[:1].to(self.device) for s in snaps]
        t_delta = torch.cat(dt_list, dim=0)
        return x_seq, t_delta

    def _build_shock_summary_df(
        self,
        per_shock: Dict[str, Dict],
    ) -> pd.DataFrame:
        """Build a summary DataFrame from per-shock metrics."""
        rows = []
        for name, m in per_shock.items():
            row = {"shock_name": name}
            shock = self.shocks.get(name)
            if shock:
                row.update({
                    "duration"     : shock.duration,
                    "avg_return"   : shock.avg_return,
                    "sigma_breach" : shock.sigma_breach,
                    "type"         : shock.type,
                })
            row.update(m)
            rows.append(row)
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE METRIC HELPERS  (extend model/loss.py compute_metrics)
# ─────────────────────────────────────────────────────────────────────────────

def metrics_from_adjacency(
    adj_pred  : np.ndarray,
    adj_true  : np.ndarray,
    threshold : float = 0.5,
) -> Dict[str, float]:
    """
    Compute all metrics directly from two N×N adjacency matrices.
    Diagonal (self-loops) is excluded automatically.

    Args:
        adj_pred  : (N, N) predicted probabilities ∈ [0, 1].
        adj_true  : (N, N) binary ground-truth GC matrix.
        threshold : Decision boundary for F1.

    Returns:
        Full metrics dict from compute_metrics().
    """
    N    = adj_pred.shape[0]
    mask = ~np.eye(N, dtype=bool)
    return compute_metrics(
        adj_true[mask].flatten(),
        adj_pred[mask].flatten(),
        threshold=threshold,
    )


def per_ticker_auc(
    adj_pred  : np.ndarray,
    adj_true  : np.ndarray,
    tickers   : List[str],
    direction : str = "out",
) -> pd.Series:
    """
    Compute per-ticker AUC-ROC for edge prediction.

    Args:
        adj_pred  : (N, N) predicted probabilities.
        adj_true  : (N, N) binary ground truth.
        tickers   : Ordered list of N ticker symbols.
        direction : "out" → edges emanating FROM each ticker (row-wise).
                    "in"  → edges arriving AT each ticker (column-wise).

    Returns:
        pd.Series indexed by ticker, values = AUC-ROC.
    """
    from sklearn.metrics import roc_auc_score

    N      = len(tickers)
    scores = {}

    for i, ticker in enumerate(tickers):
        if direction == "out":
            row_mask = np.arange(N) != i
            y_t = adj_true[i, row_mask].astype(float)
            y_s = adj_pred[i, row_mask].astype(float)
        else:
            col_mask = np.arange(N) != i
            y_t = adj_true[col_mask, i].astype(float)
            y_s = adj_pred[col_mask, i].astype(float)

        if len(np.unique(y_t)) < 2:
            scores[ticker] = float("nan")
            continue
        scores[ticker] = float(roc_auc_score(y_t, y_s))

    return pd.Series(scores, name=f"auc_roc_{direction}").sort_values(ascending=False)


def delta_metrics(
    metrics_pre  : Dict[str, float],
    metrics_post : Dict[str, float],
) -> pd.DataFrame:
    """
    Compute the delta between pre-training and post-training metrics.
    Used in the comparative analysis to quantify what the TGAT learned
    beyond the raw Granger causality structure.

    Returns:
        DataFrame with columns: metric, pre, post, delta, delta_pct.
    """
    common = set(metrics_pre) & set(metrics_post)
    rows   = []
    for key in sorted(common):
        v_pre  = metrics_pre[key]
        v_post = metrics_post[key]
        if not (isinstance(v_pre, (int, float)) and isinstance(v_post, (int, float))):
            continue
        delta     = v_post - v_pre
        delta_pct = (delta / abs(v_pre) * 100) if v_pre != 0 else float("nan")
        rows.append({
            "metric"    : key,
            "pre"       : round(v_pre,  4),
            "post"      : round(v_post, 4),
            "delta"     : round(delta,  4),
            "delta_pct" : round(delta_pct, 2),
        })
    return pd.DataFrame(rows).set_index("metric")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level="INFO", format=CFG.LOG_FORMAT)
    import pandas as pd
    import tempfile, shutil

    torch.manual_seed(42)
    np.random.seed(42)

    # Synthetic 10-ticker, 5-shock dataset
    N   = 10
    dates   = pd.date_range("2008-01-02", periods=200, freq="B")
    tickers = [f"T{i}" for i in range(N - 1)] + ["^NSEI"]
    ret     = np.random.normal(0, 0.012, (200, N))
    returns = pd.DataFrame(ret, index=dates, columns=tickers)

    from shock_detection.detector import ShockPeriod
    from graph.dataset import ShockGraphDataset
    from model import build_tgat

    shocks = [
        ShockPeriod(t["name"], t["start"], t["end"],
                    10, -0.02, -0.05, 3.0, t["type"])
        for t in CFG.TARGET_SHOCK_PERIODS
        if pd.Timestamp(t["start"]) >= dates[0] and pd.Timestamp(t["end"]) <= dates[-1]
    ]

    # Fall back to synthetic shocks if no target periods fall in range
    if not shocks:
        shocks = [
            ShockPeriod(f"Shock_{i+1:02d}",
                        dates[i*30].strftime("%Y-%m-%d"),
                        dates[i*30+9].strftime("%Y-%m-%d"),
                        8, -0.02 - i*0.005, -0.05, 2.5 + i*0.3)
            for i in range(5)
        ]

    gc_mats = {}
    for s in shocks:
        m = (np.random.rand(N, N) > 0.60).astype("float32")
        np.fill_diagonal(m, 1.0)
        gc_mats[s.name] = m

    dataset = ShockGraphDataset(returns, gc_mats, shocks, tickers, ["mean_return"])
    dataset.build()

    model = build_tgat(in_features=1)

    suite = EvaluationSuite(model, dataset, shocks, device="cpu")

    # ── per_shock_metrics ──
    first_shock = list(dataset.period_index.keys())[0]
    m = suite.per_shock_metrics(first_shock)
    assert "auc_roc" in m and "f1" in m
    print(f"per_shock_metrics  OK : AUC={m['auc_roc']:.4f}  F1={m['f1']:.4f}")

    # ── full_eval ──
    report = suite.full_eval()
    assert "overall" in report and "per_shock" in report
    assert "summary_df" in report
    suite.print_summary(report)

    # ── mc_report ──
    fake_folds = [
        FoldResult(i, [], [], 50, 20,
                   auc_roc=0.77+np.random.normal(0,.01),
                   aupr=0.76+np.random.normal(0,.01),
                   f1=0.79+np.random.normal(0,.01),
                   auc_roc_low_fpr=0.77+np.random.normal(0,.01))
        for i in range(1, 11)
    ]
    agg = EvaluationSuite.mc_report(fake_folds, print_table=True)
    assert abs(agg["auc_roc"]["mean"] - 0.77) < 0.05
    print(f"mc_report          OK : AUC μ={agg['auc_roc']['mean']:.4f}")

    # ── threshold_sweep ──
    ts = suite.threshold_sweep(first_shock)
    assert len(ts) == 50 and "f1" in ts.columns
    best = ts.loc[ts["f1"].idxmax()]
    print(f"threshold_sweep    OK : best F1={best['f1']:.4f} at thr={best['threshold']:.3f}")

    # ── metrics_from_adjacency ──
    adj_p = np.random.rand(N, N)
    adj_t = (np.random.rand(N, N) > 0.6).astype(float)
    m2 = metrics_from_adjacency(adj_p, adj_t)
    assert "auc_roc" in m2
    print(f"metrics_from_adj   OK : AUC={m2['auc_roc']:.4f}")

    # ── per_ticker_auc ──
    pt = per_ticker_auc(adj_p, adj_t, tickers, direction="out")
    assert len(pt) == N
    print(f"per_ticker_auc     OK : top={pt.index[0]} AUC={pt.iloc[0]:.4f}")

    # ── delta_metrics ──
    dm = delta_metrics({"auc_roc": 0.70, "f1": 0.65}, {"auc_roc": 0.77, "f1": 0.79})
    assert abs(dm.loc["auc_roc", "delta"] - 0.07) < 1e-3
    print(f"delta_metrics      OK : AUC delta={dm.loc['auc_roc','delta']:.4f}")

    print("\nAll evaluation/metrics.py tests PASSED.")