"""Data contracts and validation helpers for canonical clean datasets."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

CANONICAL_COLUMNS = ["Date", "Price", "Return"]
RAW_REQUIRED_COLUMNS = ["Date", "PX_LAST", "CHG_PCT_1D"]

ASSET_FILE_MAP = {
    "spx": "spx.xlsx",
    "bcom": "bcom.xlsx",
    "treasury_10y": "treasury_10y.xlsx",
    "corp_bonds": "corp_bonds.xlsx",
}


def _normalize_column_name(value: object) -> str:
    return str(value).strip().upper()


def find_column(columns: Iterable[object], target: str) -> str | None:
    """Return the original column name matching target (case/whitespace-insensitive)."""
    target_norm = _normalize_column_name(target)
    for col in columns:
        if _normalize_column_name(col) == target_norm:
            return str(col)
    return None


def resolve_required_columns(columns: Iterable[object]) -> dict[str, str]:
    """Map required raw columns to actual file column names."""
    mapping: dict[str, str] = {}
    missing: list[str] = []

    for required in RAW_REQUIRED_COLUMNS:
        match = find_column(columns, required)
        if match is None:
            missing.append(required)
        else:
            mapping[required] = match

    if missing:
        raise ValueError(f"Missing required raw columns: {missing}")
    return mapping


def validate_canonical_dataframe(df: pd.DataFrame) -> None:
    """Validate canonical dataframe structure and core dtypes."""
    if list(df.columns) != CANONICAL_COLUMNS:
        raise ValueError(
            f"Canonical columns must be exactly {CANONICAL_COLUMNS}, got {list(df.columns)}"
        )

    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        raise ValueError("Column 'Date' must be datetime64.")

    if not pd.api.types.is_numeric_dtype(df["Price"]):
        raise ValueError("Column 'Price' must be numeric.")

    if not pd.api.types.is_numeric_dtype(df["Return"]):
        raise ValueError("Column 'Return' must be numeric.")
