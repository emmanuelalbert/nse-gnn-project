"""
model/__init__.py
-----------------
Public API for the TGAT model package.

Usage:
    from model import TGAT, build_tgat, TGATLoss, compute_metrics
    from model import TGATCombinedLoss, FocalLoss, aggregate_monte_carlo_metrics
"""

from model.tgat              import TGAT, build_tgat                       # noqa: F401
from model.loss              import (                                        # noqa: F401
    TGATLoss,
    TGATCombinedLoss,
    FocalLoss,
    ContrastiveLoss,
    compute_metrics,
    aggregate_monte_carlo_metrics,
)
from model.layers            import (                                        # noqa: F401
    GCNLayer,
    ResidualBlock,
    InputProjection,
    TGATEncoderLayer,
    LinkDecoder,
)
from model.temporal_attention import TemporalEncoding, TemporalAttention    # noqa: F401

__all__ = [
    "TGAT", "build_tgat",
    "TGATLoss", "TGATCombinedLoss", "FocalLoss", "ContrastiveLoss",
    "compute_metrics", "aggregate_monte_carlo_metrics",
    "GCNLayer", "ResidualBlock", "InputProjection",
    "TGATEncoderLayer", "LinkDecoder",
    "TemporalEncoding", "TemporalAttention",
]