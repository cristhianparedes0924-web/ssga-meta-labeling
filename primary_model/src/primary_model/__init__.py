"""Primary model package."""

from .data_contract import ASSET_FILE_MAP, CANONICAL_COLUMNS, RAW_REQUIRED_COLUMNS
from .data import (
    apply_treasury_total_return,
    load_clean_asset_csv,
    load_universe,
    universe_returns_matrix,
)
from .signals import (
    build_primary_signal_variant1,
    build_variant1_indicators,
    composite_score,
    expanding_zscore,
    score_to_signal,
)
from .treasury import compute_bond_total_return_from_yield

__all__ = [
    "ASSET_FILE_MAP",
    "CANONICAL_COLUMNS",
    "RAW_REQUIRED_COLUMNS",
    "compute_bond_total_return_from_yield",
    "apply_treasury_total_return",
    "load_clean_asset_csv",
    "load_universe",
    "universe_returns_matrix",
    "build_variant1_indicators",
    "expanding_zscore",
    "composite_score",
    "score_to_signal",
    "build_primary_signal_variant1",
]
