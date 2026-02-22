"""Backtest-engine adapters over the frozen unified core."""

from primary_model_unified import (
    annualized_return,
    annualized_vol,
    backtest_from_weights,
    max_drawdown,
    sharpe_ratio,
)

__all__ = [
    "annualized_return",
    "annualized_vol",
    "backtest_from_weights",
    "max_drawdown",
    "sharpe_ratio",
]
