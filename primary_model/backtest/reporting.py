"""Benchmark/reporting adapters over the frozen unified core."""

from __future__ import annotations

from pathlib import Path

from primary_model_unified import (
    perf_table,
    plot_drawdowns,
    plot_equity_curves,
    plot_rolling_sharpe,
    run_benchmarks as _run_benchmarks,
)


def run_benchmarks(root: Path) -> None:
    """Run canonical benchmark generation and reporting."""
    _run_benchmarks(root)


__all__ = [
    "perf_table",
    "plot_drawdowns",
    "plot_equity_curves",
    "plot_rolling_sharpe",
    "run_benchmarks",
]
