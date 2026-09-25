"""
preprocessing/__init__.py
-------------------------
Public API for the preprocessing package.

Usage:
    from preprocessing import PreprocessingPipeline, ReturnCalculator, ReturnImputer, FeatureNormalizer

    pipeline = PreprocessingPipeline()
    returns  = pipeline.run(prices_df)
"""

from preprocessing.returns    import ReturnCalculator      # noqa: F401
from preprocessing.imputer    import ReturnImputer         # noqa: F401
from preprocessing.normalizer import FeatureNormalizer     # noqa: F401
from preprocessing.pipeline   import PreprocessingPipeline # noqa: F401

__all__ = [
    "ReturnCalculator",
    "ReturnImputer",
    "FeatureNormalizer",
    "PreprocessingPipeline",
]