"""
evaluation/__init__.py
-----------------------
Public API for the evaluation package.

Usage:
    from evaluation import EvaluationSuite, metrics_from_adjacency
    from evaluation import per_ticker_auc, delta_metrics
    from evaluation import ROCAnalyzer, compare_roc_curves
    from evaluation import PRAnalyzer, compare_pr_curves
"""

from evaluation.metrics       import (                          # noqa: F401
    EvaluationSuite,
    metrics_from_adjacency,
    per_ticker_auc,
    delta_metrics,
)
from evaluation.roc_analysis  import ROCAnalyzer, compare_roc_curves   # noqa: F401
from evaluation.pr_analysis   import PRAnalyzer, compare_pr_curves     # noqa: F401

__all__ = [
    "EvaluationSuite",
    "metrics_from_adjacency",
    "per_ticker_auc",
    "delta_metrics",
    "ROCAnalyzer",
    "compare_roc_curves",
    "PRAnalyzer",
    "compare_pr_curves",
]