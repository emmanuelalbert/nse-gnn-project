"""
run_pipeline.py
================
End-to-end orchestrator for the Nifty 50 TGAT Shock Propagation pipeline.

[PATCHED] _select_best_fold() now respects CFG.CHECKPOINT_MONITOR / MODE
---------------------------------------------------------------------------
Previously this always picked the fold with the LOWEST best_val_loss:

    best = min(results, key=lambda r: r.best_val_loss)
    best_row = numeric.loc[numeric["best_val_loss"].astype(float).idxmin()]

This is the same val_loss-vs-AUC-ROC divergence bug documented in
settings.py's CHECKPOINT_MONITOR comment and fixed in
training/checkpointing.py: on this task, the lowest-val_loss checkpoint is
often the LEAST useful one (essentially pre-training), while
settings.py explicitly sets CHECKPOINT_MONITOR="auc_roc" / MODE="max" as
the correct selection criterion, with real numbers to back it up
(epoch 1: val_loss=0.58 but auc_roc=0.39; epoch 3: val_loss=0.72 but
auc_roc=0.75). Left unpatched, this function would silently pick the
WORST fold's checkpoint to generate the final post-training adjacency
matrices in stage_postadj() — undermining the checkpointing.py fix at the
pipeline-orchestration layer, one level up from where it was actually
fixed.

_select_best_fold() now derives its comparison direction and metric name
directly from CFG.CHECKPOINT_MONITOR / CFG.CHECKPOINT_MODE, matching
CheckpointManager's own selection logic, with a safe fallback to
best_val_loss if the monitored metric isn't present in a given summary
(e.g. an older cached FoldSummary written before this fix).
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import settings as CFG

logger = logging.getLogger("run_pipeline")

ALL_STAGES = [
    "data", "preprocess", "shocks", "granger", "dataset",
    "train", "postadj", "analysis", "comparison", "visualize",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nifty 50 TGAT Shock Propagation — end-to-end pipeline runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage", nargs="+", default=["all"], choices=ALL_STAGES + ["all"],
        help="Which stage(s) to run, in order. 'all' runs the full pipeline.",
    )
    parser.add_argument(
        "--shock", type=str, default=None,
        help="Restrict granger/postadj/analysis/comparison/visualize stages "
             "to a single named shock period.",
    )
    parser.add_argument(
        "--folds", type=int, default=None,
        help="Override CFG.MONTE_CARLO_N_SPLITS for the train stage.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override CFG.TRAIN_EPOCHS for the train stage.",
    )
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Ignore cached data/results and recompute every stage from scratch.",
    )
    parser.add_argument(
        "--resume-from-fold", type=int, default=None,
        help="Resume Monte Carlo training from this fold number (crash recovery).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use small synthetic data instead of real Yahoo Finance downloads, "
             "and run 2 epochs / 2 folds.",
    )
    parser.add_argument(
        "--log-level", type=str, default=CFG.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def resolve_stages(requested: List[str]) -> List[str]:
    if "all" in requested:
        return list(ALL_STAGES)
    return [s for s in ALL_STAGES if s in requested]


class PipelineContext:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.prices = None
        self.returns = None
        self.shocks = None
        self.gc_matrices: Dict = {}
        self.tickers: Optional[List[str]] = None
        self.dataset = None
        self.mc_report: Optional[Dict] = None


def stage_data(ctx: PipelineContext) -> None:
    logger.info("STAGE [data] — fetching raw closing prices")

    if ctx.args.dry_run:
        ctx.prices = _make_synthetic_prices()
        logger.info(f"[dry-run] synthetic prices: {ctx.prices.shape}")
        return

    from data.collector import DataCollector
    collector = DataCollector(force_refresh=ctx.args.force_refresh)
    ctx.prices = collector.fetch_all()
    logger.info(f"Prices ready: {ctx.prices.shape[0]} days × {ctx.prices.shape[1]} tickers")


def stage_preprocess(ctx: PipelineContext) -> None:
    logger.info("STAGE [preprocess] — log returns + KNN imputation")

    if ctx.prices is None:
        ctx.prices = _load_prices_or_synthetic(ctx)

    from preprocessing.pipeline import PreprocessingPipeline
    pipeline = PreprocessingPipeline(force_refresh=ctx.args.force_refresh)
    ctx.returns = pipeline.run(ctx.prices, save=not ctx.args.dry_run)
    ctx.tickers = list(ctx.returns.columns)
    logger.info(f"Returns ready: {ctx.returns.shape}")


def stage_shocks(ctx: PipelineContext) -> None:
    logger.info("STAGE [shocks] — detecting shock periods (Eq. 2)")

    if ctx.returns is None:
        ctx.returns = _load_returns_or_synthetic(ctx)

    from shock_detection.detector import ShockDetector
    from shock_detection.shock_registry import ShockRegistry

    detector = ShockDetector()
    shocks = detector.detect(ctx.returns)
    shocks = detector.match_target_periods(shocks)

    registry = ShockRegistry()
    registry.save(
        shocks,
        detector_params={
            "window_days": detector.window_days,
            "sigma_threshold": detector.sigma_threshold,
            "merge_gap_days": detector.merge_gap_days,
        },
        data_range={
            "start": str(ctx.returns.index.min().date()),
            "end": str(ctx.returns.index.max().date()),
        },
        overwrite=True,
    )

    ctx.shocks = _filter_shocks(shocks, ctx.args.shock)
    logger.info(
        f"Detected {len(shocks)} shocks total; "
        f"{len(ctx.shocks)} selected for downstream stages."
    )


def stage_granger(ctx: PipelineContext) -> None:
    logger.info("STAGE [granger] — computing Granger causality matrices (Eq. 3-4)")

    if ctx.returns is None:
        ctx.returns = _load_returns_or_synthetic(ctx)
    if ctx.shocks is None:
        ctx.shocks = _load_shocks_or_raise(ctx)

    from graph.granger import GrangerComputer
    from graph.adjacency import AdjacencyStore

    gc = GrangerComputer()
    ctx.gc_matrices = gc.compute_all(ctx.returns, ctx.shocks)

    density_report = gc.density_report(ctx.gc_matrices, list(ctx.returns.columns))
    logger.info(f"Granger density report:\n{density_report}")

    store = AdjacencyStore()
    store.save_all(ctx.gc_matrices, list(ctx.returns.columns), stage="pre_training")
    logger.info(f"Saved {len(ctx.gc_matrices)} pre-training adjacency matrices.")


def stage_dataset(ctx: PipelineContext) -> None:
    logger.info("STAGE [dataset] — building PyG temporal snapshots")

    if ctx.returns is None:
        ctx.returns = _load_returns_or_synthetic(ctx)
    if ctx.shocks is None:
        ctx.shocks = _load_shocks_or_raise(ctx)
    if not ctx.gc_matrices:
        ctx.gc_matrices = _load_gc_matrices_or_raise(ctx)

    from graph.dataset import ShockGraphDataset

    ctx.dataset = ShockGraphDataset(
        returns=ctx.returns,
        gc_matrices=ctx.gc_matrices,
        shocks=ctx.shocks,
        tickers=list(ctx.returns.columns),
        feature_cols=CFG.NODE_FEATURE_COLS,
    )
    ctx.dataset.build()
    logger.info(f"Dataset stats:\n{ctx.dataset.stats()}")


def stage_train(ctx: PipelineContext) -> None:
    logger.info("STAGE [train] — Monte Carlo TGAT training (Eq. 11-12)")

    if ctx.dataset is None:
        stage_dataset(ctx)

    from training.monte_carlo import MonteCarloRunner

    n_folds = ctx.args.folds or (2 if ctx.args.dry_run else CFG.MONTE_CARLO_N_SPLITS)
    epochs = ctx.args.epochs or (2 if ctx.args.dry_run else CFG.TRAIN_EPOCHS)

    runner = MonteCarloRunner(
        dataset=ctx.dataset,
        returns=ctx.returns,
        shocks=ctx.shocks,
        n_folds=n_folds,
        epochs=epochs,
        verbose=1 if ctx.args.dry_run else 10,
    )
    ctx.mc_report = runner.run(resume_from_fold=ctx.args.resume_from_fold)
    runner.print_report()


def stage_postadj(ctx: PipelineContext) -> None:
    """
    Generate TGAT-predicted adjacency matrices for every shock period using
    the best-performing Monte Carlo fold's checkpoint, and persist them as
    the "post_training" stage in AdjacencyStore.
    """
    logger.info("STAGE [postadj] — generating post-training predicted adjacency")

    import torch
    import numpy as np
    from graph.adjacency import AdjacencyStore
    from model.tgat import build_tgat
    from training.checkpointing import CheckpointManager, FoldSummary

    if ctx.returns is None:
        ctx.returns = _load_returns_or_synthetic(ctx)
    if ctx.shocks is None:
        ctx.shocks = _load_shocks_or_raise(ctx)
    if ctx.dataset is None:
        stage_dataset(ctx)

    tickers = list(ctx.returns.columns)

    best_fold = _select_best_fold(ctx)
    logger.info(f"Using fold {best_fold} checkpoint for post-training predictions.")

    model = build_tgat(in_features=len(CFG.NODE_FEATURE_COLS))
    ckpt = CheckpointManager(
        fold=best_fold,
        monitor=CFG.CHECKPOINT_MONITOR,
        mode=CFG.CHECKPOINT_MODE,
    )
    model = ckpt.load_best(model, device=CFG.DEVICE)
    model.eval()

    # FIX: all tensors built here from dataset snapshots default to CPU.
    # The model was just loaded onto CFG.DEVICE (cuda), so every input
    # tensor must be explicitly moved there too, or torch raises
    # "Expected all tensors to be on the same device" deep inside the
    # model's forward pass. MonteCarloRunner/Trainer handle this
    # internally during training; this stage builds tensors directly and
    # was missing the equivalent .to(device) calls.
    device = torch.device(CFG.DEVICE)

    predicted: Dict = {}
    with torch.no_grad():
        for shock in ctx.shocks:
            try:
                snaps = ctx.dataset.get_period_snapshots(shock.name)
            except KeyError:
                logger.warning(f"No snapshots for '{shock.name}' — skipping.")
                continue
            if not snaps:
                continue

            x_seq = torch.stack([s.x for s in snaps], dim=0).to(device)            # (T, N, F)
            time_deltas = torch.cat([s.time_delta[:1] for s in snaps]).to(device)  # (T,)
            edge_index = snaps[-1].edge_index.to(device)
            edge_weight = getattr(snaps[-1], "edge_weight", None)
            if edge_weight is not None:
                edge_weight = edge_weight.to(device)

            probs = model.predict_adjacency(
                x_seq, edge_index, time_deltas, edge_weight, apply_sigmoid=True
            )

            # FIX: a fixed absolute threshold (GRAPH_LINK_THRESHOLD=0.5)
            # produces a degenerate fully-connected matrix if the decoder's
            # raw output isn't well-calibrated across the full (0,1) range
            # — observed empirically: every single predicted probability
            # landed in [0.512, 0.529], so literally everything trivially
            # cleared 0.5. Using the model's RELATIVE ranking (top N% most
            # confident pairs) instead is far more defensible given limited
            # training, since that's what AUC-ROC/AUPR actually measure and
            # what this checkpoint was selected on.
            probs_np = probs.cpu().numpy()
            off_diag_mask = ~np.eye(probs_np.shape[0], dtype=bool)

            off_diag_probs = probs_np[off_diag_mask]
            logger.info(
                f"[{shock.name}] predicted-prob stats | "
                f"min={off_diag_probs.min():.4f} max={off_diag_probs.max():.4f} "
                f"mean={off_diag_probs.mean():.4f} std={off_diag_probs.std():.4f} | "
                f"frac>=0.5={np.mean(off_diag_probs >= 0.5):.4f}"
            )

            if CFG.GRAPH_LINK_METHOD == "percentile":
                cutoff = np.percentile(off_diag_probs, 100 - CFG.GRAPH_LINK_TOP_PCT)
                adj = (probs_np >= cutoff).astype(np.float32)
                np.fill_diagonal(adj, 1.0)  # self-loops always present, paper convention
            else:  # "absolute"
                adj = (probs_np >= CFG.GRAPH_LINK_THRESHOLD).astype(np.float32)

            predicted[shock.name] = adj

    store = AdjacencyStore()
    store.save_all(predicted, tickers, stage="post_training")
    logger.info(f"Saved {len(predicted)} post-training adjacency matrices.")


def _select_best_fold(ctx: PipelineContext) -> int:
    """
    Pick the fold with the best score across all completed folds.

    [PATCHED] Previously always used `min(..., key=best_val_loss)`, which
    silently ignored CFG.CHECKPOINT_MONITOR/MODE and would tend to select
    the fold whose checkpoint had merely converged to a low loss early
    (often the LEAST useful checkpoint on this task — see settings.py's
    CHECKPOINT_MONITOR comment and training/checkpointing.py's docstring
    for the documented val_loss-vs-AUC-ROC divergence). Now mirrors
    CheckpointManager's own selection logic: compares whichever metric
    CFG.CHECKPOINT_MONITOR names, in the direction CFG.CHECKPOINT_MODE
    specifies ("max" or "min"), with a safe fallback to best_val_loss if
    that metric isn't present (e.g. an older cached FoldSummary/mc_report
    written before this fix, or a fold with an empty metrics dict).
    """
    metric = CFG.CHECKPOINT_MONITOR   # e.g. "auc_roc"
    mode   = CFG.CHECKPOINT_MODE      # "max" or "min"
    better = max if mode == "max" else min

    if ctx.mc_report is not None:
        results = ctx.mc_report["fold_results"]

        def _score(r):
            # FoldResult exposes auc_roc/aupr/f1/... as top-level attrs
            # (see monte_carlo.py's FoldResult dataclass), so getattr
            # covers the common case; fall back to best_val_loss (and
            # flip comparison sense, since lower loss = better) if the
            # named metric isn't a recognised attribute.
            if hasattr(r, metric):
                return getattr(r, metric)
            logger.warning(
                f"CFG.CHECKPOINT_MONITOR='{metric}' not found on FoldResult "
                f"— falling back to best_val_loss (inverted) for fold selection."
            )
            return -r.best_val_loss   # so `better=max` still ends up picking lowest loss

        best = better(results, key=_score)
        return best.fold

    from training.checkpointing import FoldSummary
    try:
        summary = FoldSummary.load()
        numeric = summary[summary["fold"].apply(lambda x: str(x).isdigit())]

        if metric in numeric.columns:
            col   = metric
            idx   = numeric[col].astype(float).idxmax() if mode == "max" else numeric[col].astype(float).idxmin()
        else:
            logger.warning(
                f"CFG.CHECKPOINT_MONITOR='{metric}' not found in cached FoldSummary "
                f"columns ({list(numeric.columns)}) — falling back to best_val_loss."
            )
            idx = numeric["best_val_loss"].astype(float).idxmin()   # lower loss = better, always

        best_row = numeric.loc[idx]
        return int(best_row["fold"])
    except (FileNotFoundError, KeyError, ValueError):
        logger.warning("No fold summary found — defaulting to fold 1.")
        return 1


def stage_analysis(ctx: PipelineContext) -> None:
    logger.info("STAGE [analysis] — centrality / density / shock-metric reports")

    if ctx.prices is None:
        ctx.prices = _load_prices_or_synthetic(ctx)
    if ctx.returns is None:
        ctx.returns = _load_returns_or_synthetic(ctx)
    if ctx.shocks is None:
        ctx.shocks = _load_shocks_or_raise(ctx)

    from network_analysis.centrality import CentralityAnalyzer
    from network_analysis.density import DensityAnalyzer
    from network_analysis.shock_metrics import ShockMetricsAnalyzer
    from graph.adjacency import AdjacencyStore

    store = AdjacencyStore()
    centrality = CentralityAnalyzer(store)
    density = DensityAnalyzer(store)
    shock_metrics = ShockMetricsAnalyzer(ctx.prices, ctx.returns)

    for shock in ctx.shocks:
        try:
            cent_report = centrality.pre_post_shift_report(shock.name)
            dens_report = density.network_indicators_report(shock.name)
            stock_report = shock_metrics.stock_report(shock.name, shock)
            logger.info(
                f"[{shock.name}] centrality rows={len(cent_report)} | "
                f"density={dens_report.get('density_pre')}→{dens_report.get('density_post')} | "
                f"stock report rows={len(stock_report)}"
            )
        except FileNotFoundError as exc:
            logger.warning(f"Skipping analysis for '{shock.name}': {exc}")


def stage_comparison(ctx: PipelineContext) -> None:
    logger.info("STAGE [comparison] — cross-shock synthesis + sector vulnerability")

    if ctx.prices is None:
        ctx.prices = _load_prices_or_synthetic(ctx)
    if ctx.returns is None:
        ctx.returns = _load_returns_or_synthetic(ctx)
    if ctx.shocks is None:
        ctx.shocks = _load_shocks_or_raise(ctx)

    from comparison.shock_comparator import ShockComparator
    from comparison.sector_impact import SectorImpactAnalyzer
    from network_analysis.shock_metrics import ShockMetricsAnalyzer

    comparator = ShockComparator()
    shock_metrics = ShockMetricsAnalyzer(ctx.prices, ctx.returns)
    impact = SectorImpactAnalyzer(shock_metrics, comparator)

    for shock in ctx.shocks:
        try:
            scores = impact.vulnerability_score(shock)
            logger.info(f"[{shock.name}] sector vulnerability:\n{scores}")
        except FileNotFoundError as exc:
            logger.warning(f"Skipping comparison for '{shock.name}': {exc}")


def stage_visualize(ctx: PipelineContext) -> None:
    logger.info("STAGE [visualize] — generating figures")

    if ctx.shocks is None:
        ctx.shocks = _load_shocks_or_raise(ctx)
    if ctx.prices is None:
        ctx.prices = _load_prices_or_synthetic(ctx)
    if ctx.returns is None:
        ctx.returns = _load_returns_or_synthetic(ctx)

    from visualization import NetworkPlotter, CentralityPlotter, HeatmapPlotter, SectorPlotter
    from network_analysis.shock_metrics import ShockMetricsAnalyzer

    net = NetworkPlotter()
    cent = CentralityPlotter()
    heat = HeatmapPlotter()
    sector = SectorPlotter(ShockMetricsAnalyzer(ctx.prices, ctx.returns))

    for shock in ctx.shocks:
        name = shock.name
        try:
            net.pre_post_comparison(name, save=True)
        except Exception as exc:
            logger.warning(f"[{name}] network_plot failed: {exc}")
        try:
            heat.pre_post_diff_heatmap(name, save=True)
        except Exception as exc:
            logger.warning(f"[{name}] heatmap failed: {exc}")
        try:
            cent.paired_bars(name, metric="degree", save=True)
        except Exception as exc:
            logger.warning(f"[{name}] centrality_plot failed: {exc}")
        try:
            sector.sector_avg_change(shock, save=True)
        except Exception as exc:
            logger.warning(f"[{name}] sector_plot failed: {exc}")

    logger.info(f"Figures saved → {CFG.FIGURES_DIR}")


def _load_prices_or_synthetic(ctx: PipelineContext):
    if ctx.args.dry_run:
        return _make_synthetic_prices()
    from data.storage import load_prices
    try:
        return load_prices()
    except FileNotFoundError:
        logger.error("No cached prices found. Run the 'data' stage first.")
        raise


def _load_returns_or_synthetic(ctx: PipelineContext):
    if ctx.args.dry_run:
        prices = ctx.prices if ctx.prices is not None else _make_synthetic_prices()
        from preprocessing.returns import ReturnCalculator
        return ReturnCalculator().compute(prices)
    from data.storage import load_returns
    try:
        return load_returns()
    except FileNotFoundError:
        logger.error("No cached returns found. Run the 'preprocess' stage first.")
        raise


def _load_shocks_or_raise(ctx: PipelineContext):
    from shock_detection.shock_registry import ShockRegistry
    try:
        shocks = ShockRegistry().load()
    except FileNotFoundError:
        logger.error("No shock registry found. Run the 'shocks' stage first.")
        raise
    return _filter_shocks(shocks, ctx.args.shock)


def _load_gc_matrices_or_raise(ctx: PipelineContext) -> Dict:
    from graph.adjacency import AdjacencyStore
    store = AdjacencyStore()
    shocks = ctx.shocks if ctx.shocks is not None else _load_shocks_or_raise(ctx)

    matrices = {}
    for shock in shocks:
        try:
            matrix, _tickers = store.load(shock.name, stage="pre_training")
            matrices[shock.name] = matrix
        except FileNotFoundError:
            logger.warning(f"No pre-training matrix for '{shock.name}' — skipping.")

    if not matrices:
        logger.error("No pre-training adjacency matrices found. Run the 'granger' stage first.")
        raise FileNotFoundError("No pre-training adjacency matrices.")
    return matrices


def _filter_shocks(shocks: List, shock_name: Optional[str]) -> List:
    if shock_name is None:
        target_names = {t["name"] for t in CFG.TARGET_SHOCK_PERIODS}
        scoped = [s for s in shocks if s.name in target_names]
        return scoped if scoped else shocks
    matched = [s for s in shocks if s.name == shock_name]
    if not matched:
        raise ValueError(f"No shock named '{shock_name}' found in registry.")
    return matched


def _make_synthetic_prices():
    import numpy as np
    import pandas as pd
    from config.nifty50_universe import get_tickers

    rng = np.random.default_rng(0)
    n_days = 600
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = get_tickers()
    N = len(tickers)

    log_rets = rng.normal(0.0003, 0.012, size=(n_days, N))
    log_rets[100:113] = rng.normal(-0.032, 0.01, size=(13, N))
    log_rets[300:313] = rng.normal(0.030, 0.01, size=(13, N))
    log_rets[480:495] = rng.normal(-0.028, 0.01, size=(15, N))

    prices = 100 * np.exp(np.cumsum(log_rets, axis=0))
    df = pd.DataFrame(prices, index=dates, columns=tickers)
    df.index.name = "Date"
    return df


STAGE_FUNCS = {
    "data": stage_data,
    "preprocess": stage_preprocess,
    "shocks": stage_shocks,
    "granger": stage_granger,
    "dataset": stage_dataset,
    "train": stage_train,
    "postadj": stage_postadj,
    "analysis": stage_analysis,
    "comparison": stage_comparison,
    "visualize": stage_visualize,
}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format=CFG.LOG_FORMAT)

    if args.dry_run:
        logger.warning(
            "DRY RUN — using synthetic data, reduced epochs/folds. "
            "Do not treat these results as meaningful."
        )

    stages = resolve_stages(args.stage)
    logger.info(f"Running stages: {stages}")

    ctx = PipelineContext(args)
    start = time.time()

    for stage_name in stages:
        stage_start = time.time()
        logger.info(f"\n{'='*70}\n{stage_name.upper()}\n{'='*70}")
        STAGE_FUNCS[stage_name](ctx)
        logger.info(f"Stage '{stage_name}' done in {time.time() - stage_start:.1f}s")

    logger.info(f"\nPipeline complete in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()