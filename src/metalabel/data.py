"""Data cleaning, loading, and treasury-return adjustment utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from metalabel import PROJECT_ROOT


LOCAL_RAW_DIR = PROJECT_ROOT / "data" / "raw"

CANONICAL_COLUMNS = ["Date", "Price", "Return"]
RAW_REQUIRED_COLUMNS = ["Date", "PX_LAST", "CHG_PCT_1D"]
ASSET_FILE_MAP = {
    "spx": "spx.xlsx",
    "bcom": "bcom.xlsx",
    "treasury_10y": "treasury_10y.xlsx",
    "corp_bonds": "corp_bonds.xlsx",
}
DEFAULT_ASSETS = ["spx", "bcom", "treasury_10y", "corp_bonds"]
_REQUIRED_ASSETS = ("spx", "bcom", "treasury_10y", "corp_bonds")


def _normalize_column_name(value: object) -> str:
    return str(value).strip().upper()


def _resolve_local_raw_dir(source_raw: Path | None = None) -> Path:
    """Return the only supported raw-data directory for this project."""
    expected = LOCAL_RAW_DIR.resolve()
    if source_raw is None:
        resolved = expected
    else:
        base = PROJECT_ROOT if not source_raw.is_absolute() else Path("/")
        resolved = (base / source_raw).resolve()
    if resolved != expected:
        raise ValueError(
            f"Raw data must be read from: {expected}. Received: {resolved}"
        )
    if not resolved.exists():
        raise FileNotFoundError(f"Raw data directory not found: {resolved}")
    return resolved


def _resolve_within_project(path: Path, label: str) -> Path:
    """Resolve a path and ensure it stays inside this project folder."""
    resolved = (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    project = PROJECT_ROOT.resolve()
    if resolved != project and project not in resolved.parents:
        raise ValueError(f"{label} must be inside project folder: {project}. Received: {resolved}")
    return resolved


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
    raw_dir = _resolve_local_raw_dir()
    clean_dir = root / "data" / "clean"

    print("Preparing clean data files...")
    for asset, file_name in ASSET_FILE_MAP.items():
        clean_asset_file(asset, file_name, raw_dir=raw_dir, clean_dir=clean_dir)
    print(f"Done. Clean CSVs are in: {clean_dir}")


def _coerce_decimal_return(series: pd.Series) -> pd.Series:
    """Coerce return-like values to numeric decimals."""
    as_text = series.astype(str).str.replace(",", "", regex=False).str.strip()
    has_percent = as_text.str.contains("%", regex=False).any()
    cleaned = as_text.str.replace("%", "", regex=False)
    out = pd.to_numeric(cleaned, errors="coerce")
    if has_percent:
        out = out / 100.0
    return out


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce numeric-like values to float."""
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def load_clean_asset_csv(path: Path) -> pd.DataFrame:
    """Load a clean asset CSV with Date index and columns: Price, Return."""
    raw = pd.read_csv(path, dtype=object)

    date_col = find_column(raw.columns, "Date")
    price_col = find_column(raw.columns, "Price")
    return_col = find_column(raw.columns, "Return")

    missing: list[str] = []
    if date_col is None:
        missing.append("Date")
    if price_col is None:
        missing.append("Price")
    if missing:
        raise ValueError(f"{path}: missing required column(s): {', '.join(missing)}")

    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(raw[date_col], errors="coerce"),
            "Price": _coerce_numeric(raw[price_col]),
        }
    )
    if return_col is not None:
        frame["Return"] = _coerce_decimal_return(raw[return_col])

    frame = frame.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)

    if return_col is None:
        frame["Return"] = frame["Price"].pct_change()

    frame["Return"] = pd.to_numeric(frame["Return"], errors="coerce")

    validate_canonical_dataframe(frame[CANONICAL_COLUMNS])
    return frame.set_index("Date")[["Price", "Return"]]


def load_universe(clean_dir: Path, assets: list[str]) -> dict[str, pd.DataFrame]:
    """Load clean asset CSVs into a dictionary keyed by asset name."""
    universe: dict[str, pd.DataFrame] = {}

    for asset in assets:
        path = clean_dir / f"{asset}.csv"
        df = load_clean_asset_csv(path)

        canonical = df.reset_index()[CANONICAL_COLUMNS]
        validate_canonical_dataframe(canonical)
        universe[asset] = df

    return universe


def universe_returns_matrix(universe_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build an aligned return matrix, one column per asset."""
    if not universe_dict:
        return pd.DataFrame()

    matrix = pd.concat(
        [df["Return"].rename(asset) for asset, df in universe_dict.items()],
        axis=1,
        join="inner",
    ).sort_index()

    return matrix.dropna(how="any")


def compute_bond_total_return_from_yield(
    yield_level_percent: pd.Series,
    duration: float = 8.5,
    periods_per_year: int = 12,
    include_carry: bool = True,
) -> pd.Series:
    """Approximate bond total return from a yield-level series."""
    y = pd.to_numeric(yield_level_percent, errors="coerce") / 100.0
    dy = y.diff()
    price_return = -duration * dy

    if include_carry:
        carry = y.shift(1) / periods_per_year
        total_return = price_return + carry
    else:
        total_return = price_return

    return total_return


def apply_treasury_total_return(
    universe: dict[str, pd.DataFrame], duration: float = 8.5
) -> dict[str, pd.DataFrame]:
    """Return copy of universe with treasury returns replaced by bond proxy."""
    adjusted = {asset: df.copy(deep=True) for asset, df in universe.items()}

    treasury_key = "treasury_10y"
    if treasury_key not in adjusted:
        return adjusted

    treasury_df = adjusted[treasury_key].copy(deep=True)
    treasury_df["Return"] = compute_bond_total_return_from_yield(
        treasury_df["Price"], duration=duration
    )

    validate_canonical_dataframe(treasury_df.reset_index()[CANONICAL_COLUMNS])
    adjusted[treasury_key] = treasury_df
    return adjusted


def load_default_universe(root: Path) -> dict[str, pd.DataFrame]:
    """Load the default four-asset universe from ``root/data/clean``."""
    clean_dir = root / "data" / "clean"
    return load_universe(clean_dir, list(DEFAULT_ASSETS))
