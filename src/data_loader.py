from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_FILES: dict[str, tuple[str, int]] = {
    "BCOM": ("bcom.xlsx", 6),
    "SPX": ("spx.xlsx", 6),
    "Treasury10Y": ("treasury_10y.xlsx", 5),
    "IG_Corp": ("corp_bonds.xlsx", 5),
}
REQUIRED_SOURCE_COLUMNS = {"Date", "PX_LAST"}
EXPECTED_OUTPUT_COLUMNS = ["Date", "BCOM_Price", "SPX_Price", "Treasury10Y_Price", "IG_Corp_Price"]


def load_data(data_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Load and merge the four required monthly Bloomberg series.

    The loader is strict by design: schema or data-quality issues raise errors
    immediately to prevent silent downstream leakage or misalignment.
    """
    resolved_data_dir = Path(data_dir) if data_dir is not None else _default_data_dir()
    if not resolved_data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {resolved_data_dir}")
    if not resolved_data_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory for data_dir, got: {resolved_data_dir}")

    merged_df: pd.DataFrame | None = None
    for series_name, (filename, skiprows) in DATA_FILES.items():
        single_series = _load_single_series(
            data_dir=resolved_data_dir,
            series_name=series_name,
            filename=filename,
            skiprows=skiprows,
        )
        if merged_df is None:
            merged_df = single_series
        else:
            merged_df = merged_df.merge(single_series, on="Date", how="outer")

    if merged_df is None or merged_df.empty:
        raise ValueError("No data loaded; check input files and schema.")

    merged_df = merged_df.sort_values("Date").reset_index(drop=True)

    if merged_df["Date"].duplicated().any():
        duplicate_count = int(merged_df["Date"].duplicated().sum())
        raise ValueError(f"Merged data contains {duplicate_count} duplicate dates.")

    missing_by_column = merged_df[EXPECTED_OUTPUT_COLUMNS].isna().sum()
    if int(missing_by_column.sum()) > 0:
        missing_summary = {
            column: int(count)
            for column, count in missing_by_column.items()
            if int(count) > 0
        }
        raise ValueError(f"Merged data contains missing values: {missing_summary}")

    return merged_df[EXPECTED_OUTPUT_COLUMNS]


def _load_single_series(
    data_dir: Path,
    series_name: str,
    filename: str,
    skiprows: int,
) -> pd.DataFrame:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    try:
        raw_df = pd.read_excel(path, skiprows=skiprows)
    except Exception as exc:
        raise RuntimeError(f"Failed reading {path}: {exc}") from exc

    raw_df.columns = [str(column).strip() for column in raw_df.columns]
    missing_columns = REQUIRED_SOURCE_COLUMNS.difference(raw_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"{path.name} is missing required columns: {missing}. "
            f"Found columns: {list(raw_df.columns)}"
        )

    series_df = raw_df[["Date", "PX_LAST"]].copy()
    series_df["Date"] = pd.to_datetime(series_df["Date"], errors="coerce")
    series_df["PX_LAST"] = pd.to_numeric(series_df["PX_LAST"], errors="coerce")

    invalid_date_rows = int(series_df["Date"].isna().sum())
    if invalid_date_rows > 0:
        raise ValueError(f"{path.name} has {invalid_date_rows} rows with invalid Date values.")

    invalid_price_rows = int(series_df["PX_LAST"].isna().sum())
    if invalid_price_rows > 0:
        raise ValueError(f"{path.name} has {invalid_price_rows} rows with invalid PX_LAST values.")

    duplicate_dates = int(series_df["Date"].duplicated().sum())
    if duplicate_dates > 0:
        raise ValueError(f"{path.name} contains {duplicate_dates} duplicate dates.")

    series_df = series_df.sort_values("Date").reset_index(drop=True)
    series_df.rename(columns={"PX_LAST": f"{series_name}_Price"}, inplace=True)
    return series_df


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


if __name__ == "__main__":
    load_data()
