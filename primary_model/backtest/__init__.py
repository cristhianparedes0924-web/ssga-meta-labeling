"""Backtest and performance-reporting adapters."""

from .engine import (
    annualized_return,
    annualized_vol,
    backtest_from_weights,
    max_drawdown,
    sharpe_ratio,
)
from .reporting import perf_table, plot_drawdowns, plot_equity_curves, plot_rolling_sharpe, run_benchmarks

__all__ = [
    "annualized_return",
    "annualized_vol",
    "backtest_from_weights",
    "max_drawdown",
    "perf_table",
    "plot_drawdowns",
    "plot_equity_curves",
    "plot_rolling_sharpe",
    "run_benchmarks",
    "sharpe_ratio",
]
