"""
network_analysis/__init__.py
------------------------------
Public API for the network analysis package.

This package is the ANALYTICAL layer on top of graph.adjacency.MatrixMetrics
and preprocessing.returns.ReturnCalculator. It runs paper Eq. 13-20 across
every shock period, ties results to sectors, and reproduces the paper's
Section IV-C / IV-D narrative structure (centrality shifts, network
indicators, average change & amplitude) for the Nifty 50 comparative study.

Usage:
    from network_analysis import CentralityAnalyzer, DensityAnalyzer, ShockMetricsAnalyzer

    centrality = CentralityAnalyzer(adjacency_store)
    density    = DensityAnalyzer(adjacency_store)
    shocks     = ShockMetricsAnalyzer(prices_df, returns_df)
"""

from network_analysis.centrality    import CentralityAnalyzer, CENTRALITY_METRICS  # noqa: F401
from network_analysis.density       import DensityAnalyzer                          # noqa: F401
from network_analysis.shock_metrics import ShockMetricsAnalyzer                     # noqa: F401

__all__ = [
    "CentralityAnalyzer",
    "CENTRALITY_METRICS",
    "DensityAnalyzer",
    "ShockMetricsAnalyzer",
]