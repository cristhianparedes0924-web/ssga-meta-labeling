"""Clean-data loading and treasury-adjustment adapters."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from primary_model_unified import (
    DEFAULT_ASSETS,
    apply_treasury_total_return,
    compute_bond_total_return_from_yield,
    load_clean_asset_csv,
    load_universe,
    universe_returns_matrix,
)


def load_default_universe(root: Path) -> dict[str, pd.DataFrame]:
    """Load the default four-asset universe from ``root/data/clean``."""
    clean_dir = root / "data" / "clean"
    return load_universe(clean_dir, list(DEFAULT_ASSETS))


__all__ = [
    "DEFAULT_ASSETS",
    "apply_treasury_total_return",
    "compute_bond_total_return_from_yield",
    "load_clean_asset_csv",
    "load_default_universe",
    "load_universe",
    "universe_returns_matrix",
]
