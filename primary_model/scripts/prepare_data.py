#!/usr/bin/env python3
"""Prepare canonical CSVs from Bloomberg-style Excel exports."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from primary_model.data_contract import (  # noqa: E402
    ASSET_FILE_MAP,
    CANONICAL_COLUMNS,
    resolve_required_columns,
    validate_canonical_dataframe,
)

RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"


def find_header_row(df: pd.DataFrame) -> int:
    """Find first row containing a cell with value 'Date'."""
    for idx, row in df.iterrows():
        for value in row.tolist():
            if isinstance(value, str) and value.strip().lower() == "date":
                return int(idx)
    raise ValueError("Could not find a header row containing 'Date'.")


def clean_asset_file(asset_name: str, file_name: str) -> Path:
    input_path = RAW_DIR / file_name
    output_path = CLEAN_DIR / f"{asset_name}.csv"

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
        clean["Return"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    clean["Return"] = pd.to_numeric(clean["Return"], errors="coerce") / 100.0

    clean = clean.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)
    validate_canonical_dataframe(clean)

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
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


def main() -> None:
    print("Preparing clean data files...")
    for asset, file_name in ASSET_FILE_MAP.items():
        clean_asset_file(asset, file_name)
    print(f"Done. Clean CSVs are in: {CLEAN_DIR}")


if __name__ == "__main__":
    main()
