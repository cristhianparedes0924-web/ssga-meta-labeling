"""QC-report adapters over the frozen unified core."""

from __future__ import annotations

from pathlib import Path

from primary_model_unified import (
    annualized_stats,
    build_asset_summary,
    maybe_yield_warning,
    run_data_qc as _run_data_qc,
)


def run_data_qc(root: Path) -> None:
    """Run canonical data-quality checks and HTML reporting."""
    _run_data_qc(root)


__all__ = [
    "annualized_stats",
    "build_asset_summary",
    "maybe_yield_warning",
    "run_data_qc",
]
