"""AlphaZero-like candidate search prototype for Orbit Wars.

This package is intentionally separate from ``training2``.  It keeps the same
practical candidate-action constraint, but trains on search-improved policy
targets instead of one-hot rulebase labels.
"""

from alphaZeroLike.model import AlphaZeroLikeNet

__all__ = ["AlphaZeroLikeNet"]
