"""
Benchmarks package.

Future home for baseline strategies (buy & hold, trend rules, 60/40) and evaluation wrappers.
Currently a scaffold for future refactoring.
"""

from .static import weights_equal_weight, weights_buy_hold_spx, weights_6040
from .trend import weights_simple_trend

__all__ = [
    "weights_equal_weight",
    "weights_buy_hold_spx",
    "weights_6040",
    "weights_simple_trend",
]
