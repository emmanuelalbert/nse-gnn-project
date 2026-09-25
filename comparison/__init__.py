"""
comparison/__init__.py
-------------------------
Public API for the comparison package.

This is the TOP-LEVEL synthesis layer of the analysis stack:

    graph/adjacency.py          (raw matrix metrics, Eq. 13-18)
        ↓
    network_analysis/           (per-shock-period reporting layer)
        ↓
    comparison/                 (cross-shock-period synthesis — THIS PACKAGE)
        ↓
    visualization/ + notebooks  (final paper-style figures and narrative)

ShockComparator synthesises pre/post-training network structure changes
across MULTIPLE shock periods (paper Section IV-D, "Comparative Analysis
Across Shock Periods").

SectorImpactAnalyzer combines price-level shock metrics (amplitude,
average change) with network-structure metrics (centrality shift) into
a single vulnerability/resilience ranking per sector (paper Section
IV-D-2, "Sectoral Impact Patterns").

Usage:
    from comparison import ShockComparator, SectorImpactAnalyzer
    from graph.adjacency import AdjacencyStore
    from network_analysis import ShockMetricsAnalyzer

    comparator = ShockComparator(AdjacencyStore())
    sm         = ShockMetricsAnalyzer(prices_df, returns_df)
    impact     = SectorImpactAnalyzer(sm, comparator)
"""

from comparison.shock_comparator import (        # noqa: F401
    ShockComparator,
    PROPAGATION_MECHANISMS,
)
from comparison.sector_impact import (            # noqa: F401
    SectorImpactAnalyzer,
    VULNERABILITY_WEIGHTS,
)

__all__ = [
    "ShockComparator",
    "PROPAGATION_MECHANISMS",
    "SectorImpactAnalyzer",
    "VULNERABILITY_WEIGHTS",
]