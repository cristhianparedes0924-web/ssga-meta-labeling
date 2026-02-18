"""Data loading utilities for canonical clean asset CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_contract import CANONICAL_COLUMNS, find_column, validate_canonical_dataframe
from .treasury import compute_bond_total_return_from_yield


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
    """Coerce numeric-like values to float, handling comma separators."""
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def load_clean_asset_csv(path: Path) -> pd.DataFrame:
    """Load a clean asset CSV with Date index and columns: Price, Return.

    Accepted input schemas:
    - Date, Price, Return
    - Date, Price (Return will be derived as Price.pct_change())
    """
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

    # Ensure stable numeric return dtype after either source/derived path.
    frame["Return"] = pd.to_numeric(frame["Return"], errors="coerce")

    validate_canonical_dataframe(frame[CANONICAL_COLUMNS])

    return frame.set_index("Date")[["Price", "Return"]]


def load_universe(clean_dir: Path, assets: list[str]) -> dict[str, pd.DataFrame]:
    """Load a set of clean asset CSVs into a dictionary keyed by asset name."""
    universe: dict[str, pd.DataFrame] = {}

    for asset in assets:
        path = clean_dir / f"{asset}.csv"
        df = load_clean_asset_csv(path)

        canonical = df.reset_index()[CANONICAL_COLUMNS]
        validate_canonical_dataframe(canonical)
        universe[asset] = df

    return universe


def universe_returns_matrix(universe_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build an aligned matrix of returns, one column per asset key."""
    if not universe_dict:
        return pd.DataFrame()

    matrix = pd.concat(
        [df["Return"].rename(asset) for asset, df in universe_dict.items()],
        axis=1,
        join="inner",
    ).sort_index()

    return matrix.dropna(how="any")


def apply_treasury_total_return(
    universe: dict[str, pd.DataFrame], duration: float = 8.5
) -> dict[str, pd.DataFrame]:
    """Return a copy of universe with treasury_10y returns replaced by bond proxy."""
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
