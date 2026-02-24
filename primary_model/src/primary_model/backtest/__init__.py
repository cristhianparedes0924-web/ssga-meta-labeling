"""Backtest engine and reporting API."""

from .engine import backtest_from_weights
from .reporting import (
    plot_drawdowns,
    plot_equity_curves,
    plot_rolling_sharpe,
    run_primary_variant1,
)

__all__ = [
    "plot_drawdowns",
    "plot_equity_curves",
    "plot_rolling_sharpe",
    "run_primary_variant1",
]
