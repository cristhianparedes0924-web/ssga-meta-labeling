"""Raw-data cleaning adapters over the frozen unified core."""

from __future__ import annotations

from pathlib import Path

from primary_model_unified import (
    ASSET_FILE_MAP,
    CANONICAL_COLUMNS,
    RAW_REQUIRED_COLUMNS,
    clean_asset_file,
    find_column,
    find_header_row,
    resolve_required_columns,
    run_prepare_data,
    validate_canonical_dataframe,
)


def prepare_data(root: Path) -> None:
    """Run the canonical raw->clean data pipeline."""
    run_prepare_data(root)


__all__ = [
    "ASSET_FILE_MAP",
    "CANONICAL_COLUMNS",
    "RAW_REQUIRED_COLUMNS",
    "clean_asset_file",
    "find_column",
    "find_header_row",
    "prepare_data",
    "resolve_required_columns",
    "validate_canonical_dataframe",
]
