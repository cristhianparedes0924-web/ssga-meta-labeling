"""Data ingestion and cleaning adapters."""

from .cleaner import clean_asset_file, prepare_data
from .loader import (
    apply_treasury_total_return,
    compute_bond_total_return_from_yield,
    load_clean_asset_csv,
    load_default_universe,
    load_universe,
    universe_returns_matrix,
)

__all__ = [
    "apply_treasury_total_return",
    "clean_asset_file",
    "compute_bond_total_return_from_yield",
    "load_clean_asset_csv",
    "load_default_universe",
    "load_universe",
    "prepare_data",
    "universe_returns_matrix",
]
