"""Raw data cleaning pipeline for canonical asset CSVs."""

from __future__ import annotations

from pathlib import Path
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

DEFAULT_ASSETS = ["spx", "bcom", "treasury_10y", "corp_bonds"]


def _normalize_column_name(value: object) -> str:
    return str(value).strip().upper()


def find_column(columns: Iterable[object], target: str) -> str | None:
    """Return original column matching target (case/whitespace-insensitive)."""
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
    """Validate canonical dataframe structure and dtypes."""
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


def find_header_row(df: pd.DataFrame) -> int:
    """Find first row containing a cell with value 'Date'."""
    for idx, row in df.iterrows():
        for value in row.tolist():
            if isinstance(value, str) and value.strip().lower() == "date":
                return int(idx)
    raise ValueError("Could not find a header row containing 'Date'.")


def clean_asset_file(asset_name: str, file_name: str, raw_dir: Path, clean_dir: Path) -> Path:
    """Clean one raw Bloomberg-style Excel file into canonical CSV."""
    input_path = raw_dir / file_name
    output_path = clean_dir / f"{asset_name}.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing required input file: {input_path}")

    raw = pd.read_excel(input_path, header=None, dtype=object, engine="openpyxl")
    header_row = find_header_row(raw)

    header = raw.iloc[header_row].tolist()
    data = raw.iloc[header_row + 1 :].copy()
    data.columns = header
    data = data.loc[:, [col for col in data.columns if pd.notna(col)]]

    column_map = resolve_required_columns(data.columns)

    clean = data[
        [column_map["Date"], column_map["PX_LAST"], column_map["CHG_PCT_1D"]]
    ].copy()
    clean.columns = CANONICAL_COLUMNS

    clean["Date"] = pd.to_datetime(clean["Date"], errors="coerce")
    clean["Price"] = pd.to_numeric(clean["Price"], errors="coerce")

    clean["Return"] = (
        clean["Return"].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False)
    )
    clean["Return"] = pd.to_numeric(clean["Return"], errors="coerce") / 100.0

    clean = clean.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)
    validate_canonical_dataframe(clean)

    clean_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False, date_format="%Y-%m-%d")

    if len(clean) == 0:
        print(f"[{asset_name}] rows=0")
    else:
        min_date = clean["Date"].min().date()
        max_date = clean["Date"].max().date()
        print(f"[{asset_name}] rows={len(clean)} min_date={min_date} max_date={max_date}")
        print(clean.head(3).to_string(index=False))
    print()

    return output_path


def prepare_data(root: Path) -> None:
    """Run the raw Excel to clean CSV pipeline for default assets."""
    raw_dir = root / "data" / "raw"
    clean_dir = root / "data" / "clean"

    print("Preparing clean data files...")
    for asset, file_name in ASSET_FILE_MAP.items():
        clean_asset_file(asset, file_name, raw_dir=raw_dir, clean_dir=clean_dir)
    print(f"Done. Clean CSVs are in: {clean_dir}")


__all__ = [
    "ASSET_FILE_MAP",
    "CANONICAL_COLUMNS",
    "DEFAULT_ASSETS",
    "RAW_REQUIRED_COLUMNS",
    "clean_asset_file",
    "find_column",
    "find_header_row",
    "prepare_data",
    "resolve_required_columns",
    "validate_canonical_dataframe",
]
