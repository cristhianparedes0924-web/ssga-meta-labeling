#!/usr/bin/env python3
"""Single-file implementation of the primary_model project workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent

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


# ==============================
# Data cleaning and loading
# ==============================


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


# ==============================
# Signal construction
# ==============================


def expanding_zscore(
    series: pd.Series,
    min_periods: int = 12,
    ddof: int = 1,
) -> pd.Series:
    """Compute expanding z-score using only prior observations."""
    if min_periods < 1:
        raise ValueError("min_periods must be >= 1.")

    x = pd.to_numeric(series, errors="coerce")
    history = x.shift(1)
    hist_mean = history.expanding(min_periods=min_periods).mean()
    hist_std = history.expanding(min_periods=min_periods).std(ddof=ddof)
    z = (x - hist_mean) / hist_std
    return z.replace([np.inf, -np.inf], np.nan)


def build_variant1_indicators(
    universe: dict[str, pd.DataFrame],
    trend_window: int = 12,
    relative_window: int = 3,
) -> pd.DataFrame:
    """Build Variant 1 raw indicators from universe data."""
    if trend_window < 1 or relative_window < 1:
        raise ValueError("trend_window and relative_window must be >= 1.")

    missing = [asset for asset in _REQUIRED_ASSETS if asset not in universe]
    if missing:
        raise ValueError(f"Universe is missing required asset(s): {missing}")

    spx_price = pd.to_numeric(universe["spx"]["Price"], errors="coerce")
    bcom_price = pd.to_numeric(universe["bcom"]["Price"], errors="coerce")

    spx_ret = pd.to_numeric(universe["spx"]["Return"], errors="coerce")
    bcom_ret = pd.to_numeric(universe["bcom"]["Return"], errors="coerce")
    corp_ret = pd.to_numeric(universe["corp_bonds"]["Return"], errors="coerce")
    ust_ret = pd.to_numeric(universe["treasury_10y"]["Return"], errors="coerce")

    spx_trend = spx_price / spx_price.rolling(trend_window, min_periods=trend_window).mean() - 1.0
    bcom_trend = (
        bcom_price / bcom_price.rolling(trend_window, min_periods=trend_window).mean() - 1.0
    )
    credit_vs_rates = (corp_ret - ust_ret).rolling(
        relative_window, min_periods=relative_window
    ).mean()
    risk_breadth = (pd.concat([spx_ret, bcom_ret, corp_ret], axis=1).mean(axis=1) - ust_ret).rolling(
        relative_window, min_periods=relative_window
    ).mean()

    indicators = pd.DataFrame(
        {
            "spx_trend": spx_trend,
            "bcom_trend": bcom_trend,
            "credit_vs_rates": credit_vs_rates,
            "risk_breadth": risk_breadth,
        }
    ).sort_index()
    return indicators


def _weight_series(columns: pd.Index, weights: Mapping[str, float] | None) -> pd.Series:
    if len(columns) == 0:
        raise ValueError("No indicator columns were provided.")

    if weights is None:
        return pd.Series(1.0, index=columns, dtype=float)

    provided = pd.Series(weights, dtype=float)
    missing = [col for col in columns if col not in provided.index]
    extra = [col for col in provided.index if col not in columns]
    if missing:
        raise ValueError(f"Missing indicator weight(s): {missing}")
    if extra:
        raise ValueError(f"Unknown indicator weight(s): {extra}")

    if np.isclose(provided.abs().sum(), 0.0):
        raise ValueError("At least one indicator weight must be non-zero.")
    return provided.reindex(columns)


def composite_score(
    zscores: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
) -> pd.Series:
    """Compute a weighted composite score from indicator z-scores."""
    w = _weight_series(zscores.columns, weights)
    weighted_sum = zscores.mul(w, axis=1).sum(axis=1, min_count=1)
    active_weight = zscores.notna().mul(w.abs(), axis=1).sum(axis=1)
    score = weighted_sum / active_weight.replace(0.0, np.nan)
    return score.replace([np.inf, -np.inf], np.nan)


def score_to_signal(
    score: pd.Series,
    buy_threshold: float = 0.0001,
    sell_threshold: float = -0.0001,
) -> pd.Series:
    """Convert a continuous score to discrete BUY/HOLD/SELL labels."""
    if buy_threshold <= sell_threshold:
        raise ValueError("buy_threshold must be greater than sell_threshold.")

    out = pd.Series("HOLD", index=score.index, dtype=object)
    clean_score = pd.to_numeric(score, errors="coerce")
    out.loc[clean_score > buy_threshold] = "BUY"
    out.loc[clean_score < sell_threshold] = "SELL"
    out.loc[clean_score.isna()] = np.nan
    return out


def _dynamic_composite_score(zscores: pd.DataFrame, target_ret: pd.Series) -> pd.Series:
    """Compute expanding IC/correlation-adjusted factor score."""
    ic_data = zscores.shift(1).copy()
    ic_data["target_ret"] = pd.to_numeric(target_ret, errors="coerce")

    weights = pd.DataFrame(index=zscores.index, columns=zscores.columns, dtype=float)

    for i in range(len(zscores)):
        window = zscores.iloc[: i + 1].dropna(how="all")

        ic_mask = pd.Series(1.0, index=zscores.columns)
        if i >= 36:
            ic_window = ic_data.iloc[i - 36 : i + 1].dropna()
            if len(ic_window) >= 12:
                corrs = ic_window.corr(method="spearman")["target_ret"].drop("target_ret")
                ic_mask = (corrs > 0.0).astype(float)

        if len(window) < 12:
            base_w = pd.Series(1.0 / zscores.shape[1], index=zscores.columns)
        else:
            corr_mat = window.corr()
            if corr_mat.isna().any().any():
                base_w = pd.Series(1.0 / zscores.shape[1], index=zscores.columns)
            else:
                sum_corr = corr_mat.sum(axis=0).clip(lower=1.0)
                inv_corr = 1.0 / sum_corr
                base_w = inv_corr / inv_corr.sum()

        final_w = base_w * ic_mask
        if final_w.sum() > 0:
            final_w = final_w / final_w.sum()
        else:
            final_w = pd.Series(1.0 / zscores.shape[1], index=zscores.columns)

        weights.iloc[i] = final_w.values

    return (zscores * weights).sum(axis=1).rename("composite_score")


def build_primary_signal_variant1(
    universe: dict[str, pd.DataFrame],
    trend_window: int = 12,
    relative_window: int = 3,
    zscore_min_periods: int = 12,
    indicator_weights: Mapping[str, float] | None = None,
    buy_threshold: float = 0.0001,
    sell_threshold: float = -0.0001,
) -> pd.DataFrame:
    """Build Variant 1 indicators, z-scores, composite score, and signal."""
    indicators = build_variant1_indicators(
        universe=universe,
        trend_window=trend_window,
        relative_window=relative_window,
    )

    zscores = indicators.apply(
        expanding_zscore,
        axis=0,
        min_periods=zscore_min_periods,
    )
    zscores.columns = [f"{col}_z" for col in zscores.columns]

    if indicator_weights is None:
        spx_price = pd.to_numeric(universe["spx"]["Price"], errors="coerce")
        target_ret = spx_price.pct_change()
        score = _dynamic_composite_score(zscores=zscores, target_ret=target_ret)
    else:
        score = composite_score(zscores=zscores, weights=indicator_weights).rename("composite_score")

    signal = score_to_signal(
        score=score,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    ).rename("signal")

    return pd.concat([indicators, zscores, score, signal], axis=1)


# ==============================
# Portfolio and backtest
# ==============================


def _empty_weights(index: pd.Index, columns: Iterable[str]) -> pd.DataFrame:
    cols = list(columns)
    return pd.DataFrame(0.0, index=index, columns=cols, dtype=float)


def _allocation_for_assets(columns: list[str], assets: tuple[str, ...]) -> pd.Series:
    """Allocate 100% equally across provided assets present in columns."""
    out = pd.Series(0.0, index=columns, dtype=float)
    active = [asset for asset in assets if asset in columns]
    if active:
        out.loc[active] = 1.0 / len(active)
    return out


def weights_from_primary_signal(
    signal: pd.Series,
    returns_columns: list[str],
    risk_on: tuple[str, ...] = ("spx", "bcom", "corp_bonds"),
    risk_off: tuple[str, ...] = ("treasury_10y",),
    pre_signal_mode: str = "equal_weight",
    hold_mode: str = "carry",
) -> pd.DataFrame:
    """Convert BUY/HOLD/SELL labels to long-only portfolio weights."""
    if len(returns_columns) == 0:
        raise ValueError("returns_columns must include at least one asset.")
    if len(set(returns_columns)) != len(returns_columns):
        raise ValueError("returns_columns must not contain duplicates.")
    if pre_signal_mode not in {"equal_weight", "risk_off"}:
        raise ValueError("pre_signal_mode must be one of: {'equal_weight', 'risk_off'}.")
    if hold_mode != "carry":
        raise ValueError("hold_mode must be 'carry'.")

    columns = list(returns_columns)
    weights = pd.DataFrame(0.0, index=signal.index, columns=columns, dtype=float)

    equal_weight = pd.Series(1.0 / len(columns), index=columns, dtype=float)
    buy_weight = _allocation_for_assets(columns, risk_on)
    sell_weight = _allocation_for_assets(columns, risk_off)
    pre_weight = equal_weight if pre_signal_mode == "equal_weight" else sell_weight

    seen_valid_signal = False
    previous_weight = pre_weight.copy()

    for ts, raw_signal in signal.items():
        if pd.isna(raw_signal):
            current = previous_weight.copy() if seen_valid_signal else pre_weight.copy()
        else:
            label = str(raw_signal).strip().upper()
            seen_valid_signal = True
            if label == "BUY":
                current = buy_weight.copy()
            elif label == "SELL":
                current = sell_weight.copy()
            elif label == "HOLD":
                current = previous_weight.copy()
            else:
                raise ValueError(f"Unsupported signal label: {raw_signal!r}")

        current = current.clip(lower=0.0)
        row_sum = float(current.sum())
        if row_sum > 0.0:
            current = current / row_sum

        weights.loc[ts, :] = current.values
        previous_weight = current

    return weights


def primary_signal_to_weights(signal: pd.Series, returns_columns: list[str]) -> pd.DataFrame:
    """Alias for converting BUY/HOLD/SELL signals into long-only weights."""
    return weights_from_primary_signal(signal=signal, returns_columns=returns_columns)


def weights_equal_weight(returns: pd.DataFrame) -> pd.DataFrame:
    if returns.shape[1] == 0:
        raise ValueError("returns must contain at least one asset column.")

    w = _empty_weights(returns.index, returns.columns)
    w.loc[:, :] = 1.0 / returns.shape[1]
    return w


def weights_buy_hold_spx(returns: pd.DataFrame, spx_col: str = "spx") -> pd.DataFrame:
    if spx_col not in returns.columns:
        raise ValueError(f"Column '{spx_col}' not found in returns.")

    w = _empty_weights(returns.index, returns.columns)
    w[spx_col] = 1.0
    return w


def weights_6040(
    returns: pd.DataFrame, spx_col: str = "spx", ust_col: str = "treasury_10y"
) -> pd.DataFrame:
    missing = [col for col in (spx_col, ust_col) if col not in returns.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in returns: {missing}")

    w = _empty_weights(returns.index, returns.columns)
    w[spx_col] = 0.6
    w[ust_col] = 0.4
    return w


def weights_simple_trend(
    prices: pd.Series,
    returns_cols: list[str],
    risk_on: tuple[str, ...] = ("spx",),
    risk_off: tuple[str, ...] = ("treasury_10y",),
    fast: int = 10,
    slow: int = 12,
) -> pd.DataFrame:
    if len(returns_cols) == 0:
        raise ValueError("returns_cols must include at least one asset.")
    if fast <= 0 or slow <= 0:
        raise ValueError("fast and slow must be positive integers.")

    cols = list(returns_cols)
    missing = [asset for asset in (*risk_on, *risk_off) if asset not in cols]
    if missing:
        raise ValueError(f"Assets not present in returns_cols: {missing}")
    if set(risk_on) & set(risk_off):
        raise ValueError("risk_on and risk_off must be disjoint.")
    if len(risk_on) == 0 or len(risk_off) == 0:
        raise ValueError("risk_on and risk_off must be non-empty.")

    prices_numeric = pd.to_numeric(prices, errors="coerce")
    sma_fast = prices_numeric.rolling(window=fast, min_periods=fast).mean()
    sma_slow = prices_numeric.rolling(window=slow, min_periods=slow).mean()
    signal_risk_on = sma_fast > sma_slow

    w = _empty_weights(prices.index, cols)
    on_weight = 1.0 / len(risk_on)
    off_weight = 1.0 / len(risk_off)

    on_mask = signal_risk_on.fillna(False).to_numpy()
    off_mask = ~on_mask

    for asset in risk_on:
        w.loc[on_mask, asset] = on_weight
    for asset in risk_off:
        w.loc[off_mask, asset] = off_weight

    return w


def _normalize_long_only_weights(weights: pd.DataFrame) -> pd.DataFrame:
    long_only = weights.clip(lower=0.0)
    row_sums = long_only.sum(axis=1)
    normalized = long_only.div(row_sums.replace(0.0, np.nan), axis=0)
    return normalized


def backtest_from_weights(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    tcost_bps: float = 0.0,
) -> pd.DataFrame:
    """Backtest by applying weights at t to next period returns (t+1)."""
    common_index = returns.index.intersection(weights.index)
    common_cols = returns.columns.intersection(weights.columns)
    if common_index.empty:
        raise ValueError("No overlapping dates between returns and weights.")
    if common_cols.empty:
        raise ValueError("No overlapping asset columns between returns and weights.")

    rets = returns.loc[common_index, common_cols].sort_index()
    w_raw = weights.loc[common_index, common_cols].sort_index()
    w = _normalize_long_only_weights(w_raw)

    next_rets = rets.shift(-1)
    gross_return = (w * next_rets).sum(axis=1, min_count=1)

    turnover = 0.5 * w.diff().abs().sum(axis=1, min_count=1).fillna(0.0)
    cost = turnover * (tcost_bps / 10000.0)
    net_return = gross_return - cost

    result = pd.DataFrame(
        {
            "gross_return": gross_return,
            "net_return": net_return,
            "turnover": turnover,
        }
    )

    result = result.iloc[:-1].copy()
    result["equity_gross"] = (1.0 + result["gross_return"]).cumprod()
    result["equity_net"] = (1.0 + result["net_return"]).cumprod()

    return result


def annualized_return(r: pd.Series, periods_per_year: int = 12) -> float:
    clean = pd.to_numeric(r, errors="coerce").dropna()
    n = len(clean)
    if n == 0 or periods_per_year <= 0:
        return float(np.nan)

    gross = float((1.0 + clean).prod())
    if gross <= 0.0:
        return float(np.nan)

    return float(gross ** (periods_per_year / n) - 1.0)


def annualized_vol(r: pd.Series, periods_per_year: int = 12) -> float:
    clean = pd.to_numeric(r, errors="coerce").dropna()
    if len(clean) < 2 or periods_per_year <= 0:
        return float(np.nan)

    return float(clean.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    r: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 12
) -> float:
    clean = pd.to_numeric(r, errors="coerce").dropna()
    if len(clean) < 2 or periods_per_year <= 0 or rf_annual <= -1.0:
        return float(np.nan)

    rf_period = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess = clean - rf_period
    std_excess = float(excess.std(ddof=1))
    if std_excess == 0.0 or np.isnan(std_excess):
        return float(np.nan)

    return float(excess.mean() / std_excess * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    clean = pd.to_numeric(equity, errors="coerce").dropna()
    if len(clean) == 0:
        return float(np.nan)

    running_max = clean.cummax()
    drawdown = clean / running_max - 1.0
    return float(drawdown.min())


# ==============================
# Reporting and QC
# ==============================


def perf_table(
    backtests: dict[str, pd.DataFrame], periods_per_year: int = 12
) -> pd.DataFrame:
    """Build standardized performance summary table."""
    columns = [
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "calmar",
        "avg_turnover",
    ]
    if not backtests:
        return pd.DataFrame(columns=columns)

    rows: dict[str, dict[str, float]] = {}
    for name, df in backtests.items():
        required = {"net_return", "equity_net"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{name}: missing required columns: {sorted(missing)}")

        net = df["net_return"]
        equity = df["equity_net"]

        ann_ret = annualized_return(net, periods_per_year=periods_per_year)
        ann_vol = annualized_vol(net, periods_per_year=periods_per_year)
        shp = sharpe_ratio(net, rf_annual=0.0, periods_per_year=periods_per_year)
        mdd = max_drawdown(equity)

        calmar = float(np.nan)
        if not np.isnan(ann_ret) and not np.isnan(mdd) and mdd < 0.0:
            calmar = float(ann_ret / abs(mdd))

        avg_turnover = float(np.nan)
        if "turnover" in df.columns:
            avg_turnover = float(pd.to_numeric(df["turnover"], errors="coerce").mean())

        rows[name] = {
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": shp,
            "max_drawdown": mdd,
            "calmar": calmar,
            "avg_turnover": avg_turnover,
        }

    return pd.DataFrame.from_dict(rows, orient="index")[columns]


def plot_equity_curves(backtests: dict[str, pd.DataFrame], out_path: Path) -> Path:
    """Plot net equity curves for all strategies."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, df in backtests.items():
        if "equity_net" not in df.columns:
            raise ValueError(f"{name}: missing required column 'equity_net'.")
        equity = pd.to_numeric(df["equity_net"], errors="coerce")
        ax.plot(equity.index, equity.values, label=name, linewidth=1.8)

    ax.set_title("Equity Curves (Net)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_drawdowns(backtests: dict[str, pd.DataFrame], out_path: Path) -> Path:
    """Plot drawdown series for all strategies."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, df in backtests.items():
        if "equity_net" not in df.columns:
            raise ValueError(f"{name}: missing required column 'equity_net'.")
        equity = pd.to_numeric(df["equity_net"], errors="coerce")
        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        ax.plot(drawdown.index, drawdown.values, label=name, linewidth=1.8)

    ax.set_title("Drawdowns (Net Equity)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_rolling_sharpe(
    backtests: dict[str, pd.DataFrame], out_path: Path, window: int = 12
) -> Path:
    """Plot rolling Sharpe for each strategy using net returns."""
    if window <= 1:
        raise ValueError("window must be > 1 for rolling Sharpe.")

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, df in backtests.items():
        if "net_return" not in df.columns:
            raise ValueError(f"{name}: missing required column 'net_return'.")
        net = pd.to_numeric(df["net_return"], errors="coerce")
        rolling_mean = net.rolling(window=window, min_periods=window).mean()
        rolling_std = net.rolling(window=window, min_periods=window).std(ddof=1)
        rolling_sharpe = rolling_mean / rolling_std * np.sqrt(12.0)
        rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan)
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, label=name, linewidth=1.8)

    ax.set_title(f"Rolling Sharpe (Net, {window}-month)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _strategy_return_table(backtests: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame(
        {name: bt["net_return"] for name, bt in backtests.items()}
    ).sort_index()


def _excess_vs_equal_weight(strategy_returns: pd.DataFrame, ew_col: str) -> pd.DataFrame:
    ew = strategy_returns[ew_col]
    rows = []
    for col in strategy_returns.columns:
        diff = strategy_returns[col] - ew
        rows.append(
            {
                "strategy": col,
                "mean_excess_annual": float(diff.mean() * 12.0),
                "months_won": int((diff > 0).sum()),
                "months_lost": int((diff < 0).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


def _corr_vs_equal_weight(strategy_returns: pd.DataFrame, ew_col: str) -> pd.DataFrame:
    ew = strategy_returns[ew_col]
    corr = strategy_returns.corrwith(ew)
    return corr.to_frame(name="corr_with_equal_weight")


def _asset_sanity_table(returns: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame(index=returns.columns)
    table["ann_mean"] = returns.mean() * 12.0
    table["ann_vol"] = returns.std(ddof=1) * np.sqrt(12.0)
    table["sharpe"] = [
        sharpe_ratio(returns[col], rf_annual=0.0, periods_per_year=12)
        for col in returns.columns
    ]
    return table


def run_primary_variant1(root: Path) -> None:
    """Run primary strategy pipeline and save backtest summary outputs."""
    clean_dir = root / "data" / "clean"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    universe = load_universe(clean_dir, DEFAULT_ASSETS)
    adj_universe = apply_treasury_total_return(universe, duration=8.5)
    signals = build_primary_signal_variant1(adj_universe)
    returns = universe_returns_matrix(adj_universe)

    weights_raw = weights_from_primary_signal(
        signal=signals["signal"],
        returns_columns=list(returns.columns),
    )
    weights = weights_raw.reindex(returns.index).ffill()
    equal_weight_row = pd.Series(1.0 / len(returns.columns), index=returns.columns, dtype=float)
    weights = weights.fillna(equal_weight_row)

    backtest = backtest_from_weights(returns=returns, weights=weights, tcost_bps=0.0)
    summary = perf_table({"PrimaryV1": backtest}, periods_per_year=12)

    signal_counts = (
        signals["signal"]
        .value_counts(dropna=True)
        .reindex(["BUY", "HOLD", "SELL"], fill_value=0)
        .astype(int)
    )
    avg_turnover = float(backtest["turnover"].mean())

    print("Performance table (net):")
    print(summary.to_string())
    print()

    print("Signal counts:")
    print(signal_counts.to_string())
    print()

    print(f"Average turnover: {avg_turnover:.6f}")

    backtest_path = reports_dir / "primary_v1_backtest.csv"
    summary_path = reports_dir / "primary_v1_summary.csv"
    backtest.to_csv(backtest_path, index=True)
    summary.to_csv(summary_path, index=True)

    print(f"Saved CSV: {backtest_path}")
    print(f"Saved CSV: {summary_path}")


def run_benchmarks(root: Path) -> None:
    """Run benchmark comparisons and export summary reports/plots."""
    clean_dir = root / "data" / "clean"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = reports_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    universe = load_universe(clean_dir, DEFAULT_ASSETS)
    adj_universe = apply_treasury_total_return(universe, duration=8.5)
    returns = universe_returns_matrix(adj_universe)
    primary_signals = build_primary_signal_variant1(adj_universe)

    weights_by_name = {
        "EqualWeight25": weights_equal_weight(returns),
        "BuyHoldSPX": weights_buy_hold_spx(returns),
        "60/40": weights_6040(returns),
        "SimpleTrend": weights_simple_trend(
            prices=adj_universe["spx"]["Price"],
            returns_cols=list(returns.columns),
        ),
    }

    primary_weights = weights_from_primary_signal(
        signal=primary_signals["signal"],
        returns_columns=list(returns.columns),
    )
    primary_weights = primary_weights.reindex(returns.index).ffill()
    equal_weight_row = pd.Series(1.0 / len(returns.columns), index=returns.columns, dtype=float)
    primary_weights = primary_weights.fillna(equal_weight_row)
    weights_by_name["PrimaryV1"] = primary_weights

    backtests = {
        name: backtest_from_weights(returns=returns, weights=w, tcost_bps=0.0)
        for name, w in weights_by_name.items()
    }

    summary = perf_table(backtests, periods_per_year=12)
    strategy_returns = _strategy_return_table(backtests)
    excess = _excess_vs_equal_weight(strategy_returns, ew_col="EqualWeight25")
    corr = _corr_vs_equal_weight(strategy_returns, ew_col="EqualWeight25")
    asset_stats = _asset_sanity_table(returns)
    asset_corr = returns.corr()

    print("Benchmark performance table (net):")
    print(summary.to_string())
    print()

    primary_signal_counts = (
        primary_signals["signal"]
        .value_counts(dropna=True)
        .reindex(["BUY", "HOLD", "SELL"], fill_value=0)
        .astype(int)
    )
    primary_avg_turnover = float(backtests["PrimaryV1"]["turnover"].mean())
    print("PrimaryV1 signal counts:")
    print(primary_signal_counts.to_string())
    print()
    print(f"PrimaryV1 average turnover: {primary_avg_turnover:.6f}")
    print()

    print("Excess vs EqualWeight25:")
    print(excess.to_string())
    print()

    print("Correlation with EqualWeight25 net returns:")
    print(corr.to_string())
    print()

    print("Asset-level sanity stats (adjusted returns):")
    print(asset_stats.to_string())
    print()

    print("Asset returns correlation matrix (adjusted returns):")
    print(asset_corr.to_string())
    print()

    equity_plot_path = plot_equity_curves(backtests, assets_dir / "equity_curves.png")
    drawdowns_plot_path = plot_drawdowns(backtests, assets_dir / "drawdowns.png")
    rolling_sharpe_plot_path = plot_rolling_sharpe(
        backtests, assets_dir / "rolling_sharpe.png", window=12
    )

    csv_path = reports_dir / "benchmarks_summary.csv"
    html_path = reports_dir / "benchmarks_summary.html"
    primary_signal_path = reports_dir / "primary_v1_signal.csv"
    primary_weights_path = reports_dir / "primary_v1_weights.csv"

    summary.to_csv(csv_path, index=True)
    primary_signals.to_csv(primary_signal_path, index=True)
    primary_weights.to_csv(primary_weights_path, index=True)

    html_parts = [
        (
            "<html><head><meta charset='utf-8'><title>Benchmark Summary</title>"
            "<style>"
            "body{font-family:Arial,sans-serif;margin:20px;color:#111;}"
            "h1,h2,h3{margin-top:20px;}"
            "table{border-collapse:collapse;margin-bottom:16px;}"
            "th,td{border:1px solid #d0d0d0;padding:6px 10px;text-align:right;}"
            "th:first-child,td:first-child{text-align:left;}"
            "img{max-width:100%;height:auto;border:1px solid #d0d0d0;margin-bottom:16px;}"
            "</style></head><body>"
        ),
        "<h1>Benchmark Summary</h1>",
        "<h2>Performance Table (Net)</h2>",
        summary.to_html(),
        "<h2>Excess vs EqualWeight25</h2>",
        excess.to_html(),
        "<h2>Correlation with EqualWeight25 Net Returns</h2>",
        corr.to_html(),
        "<h2>Benchmark Sanity Check</h2>",
        "<h3>Asset-Level Annualized Stats (Adjusted Returns)</h3>",
        asset_stats.to_html(),
        "<h3>Asset Return Correlation Matrix (Adjusted Returns)</h3>",
        asset_corr.to_html(),
        "<ul>",
        "<li>EqualWeight25 controls for exposure to this fixed four-asset universe by allocating 25% to each asset every month.</li>",
        "<li>It does not control for volatility targeting, risk parity weighting, or dynamic leverage constraints.</li>",
        "<li>It also does not control for timing skill; any excess over EqualWeight25 reflects allocation and timing differences.</li>",
        "</ul>",
        "<h2>Plots</h2>",
        "<h3>Equity Curves (Net)</h3>",
        f"<img src='assets/{equity_plot_path.name}' alt='Equity curves'>",
        "<h3>Drawdowns</h3>",
        f"<img src='assets/{drawdowns_plot_path.name}' alt='Drawdowns'>",
        "<h3>Rolling Sharpe (12M)</h3>",
        f"<img src='assets/{rolling_sharpe_plot_path.name}' alt='Rolling Sharpe'>",
        "</body></html>",
    ]
    html_path.write_text("\n".join(html_parts), encoding="utf-8")

    print(f"Saved CSV: {csv_path}")
    print(f"Saved CSV: {primary_signal_path}")
    print(f"Saved CSV: {primary_weights_path}")
    print(f"Saved HTML: {html_path}")
    print(f"Saved plot: {equity_plot_path}")
    print(f"Saved plot: {drawdowns_plot_path}")
    print(f"Saved plot: {rolling_sharpe_plot_path}")


def build_asset_summary(universe: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create row-level quality summary for each asset."""
    rows: list[dict[str, object]] = []

    for asset, df in universe.items():
        rows.append(
            {
                "asset": asset,
                "rows": int(len(df)),
                "min_date": df.index.min().date().isoformat() if len(df) else None,
                "max_date": df.index.max().date().isoformat() if len(df) else None,
                "pct_missing_return": float(df["Return"].isna().mean() * 100.0),
                "price_min": float(df["Price"].min()) if len(df) else np.nan,
                "price_max": float(df["Price"].max()) if len(df) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def maybe_yield_warning(universe: dict[str, pd.DataFrame]) -> str | None:
    """Warn if treasury prices appear to be yield-level inputs."""
    treasury = universe.get("treasury_10y")
    if treasury is None or treasury.empty:
        return None

    price_min = float(treasury["Price"].min())
    price_max = float(treasury["Price"].max())
    if 0.0 <= price_min <= 20.0 and 0.0 <= price_max <= 20.0:
        return (
            "WARNING: treasury_10y Price appears yield-like "
            f"(range {price_min:.4f} to {price_max:.4f}). "
            "Treat this as a yield level series, not a bond total-return index."
        )
    return None


def annualized_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute annualized mean and vol under monthly frequency assumption."""
    return pd.DataFrame(
        {
            "ann_mean": returns.mean() * 12.0,
            "ann_vol": returns.std(ddof=1) * np.sqrt(12.0),
        }
    )


def run_data_qc(root: Path) -> None:
    """Run data quality checks and write a compact HTML QC report."""
    clean_dir = root / "data" / "clean"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    html_path = reports_dir / "data_qc.html"

    universe = load_universe(clean_dir, DEFAULT_ASSETS)
    asset_summary = build_asset_summary(universe)
    raw_returns_matrix = universe_returns_matrix(universe)
    adj_universe = apply_treasury_total_return(universe, duration=8.5)
    adj_returns_matrix = universe_returns_matrix(adj_universe)

    overlap_summary = pd.DataFrame(
        [
            {
                "raw_overlap_rows": int(len(raw_returns_matrix)),
                "adjusted_overlap_rows": int(len(adj_returns_matrix)),
                "overlap_min_date": (
                    raw_returns_matrix.index.min().date().isoformat()
                    if len(raw_returns_matrix)
                    else None
                ),
                "overlap_max_date": (
                    raw_returns_matrix.index.max().date().isoformat()
                    if len(raw_returns_matrix)
                    else None
                ),
            }
        ]
    )

    corr = raw_returns_matrix.corr()
    raw_ann_stats = annualized_stats(raw_returns_matrix)
    adj_ann_stats = annualized_stats(adj_returns_matrix)

    warning_line = maybe_yield_warning(universe)

    print("Asset summary:")
    print(asset_summary.to_string(index=False))
    print()

    if warning_line:
        print(warning_line)
        print()

    print("Overlap summary:")
    print(overlap_summary.to_string(index=False))
    print()

    print("Returns correlation matrix:")
    print(corr.to_string())
    print()

    print("Raw annualized mean/vol (monthly assumption):")
    print(raw_ann_stats.to_string())
    print()

    print("Adjusted annualized mean/vol (monthly assumption, treasury duration=8.5):")
    print(adj_ann_stats.to_string())
    print()

    html_parts = [
        "<html><head><meta charset='utf-8'><title>Data QC Report</title></head><body>",
        "<h1>Data QC Report</h1>",
        "<h2>Asset Summary</h2>",
        asset_summary.to_html(index=False),
        "<h2>Overlap Summary</h2>",
        overlap_summary.to_html(index=False),
        "<h2>Returns Correlation Matrix</h2>",
        corr.to_html(),
        "<h2>Treasury Return Handling</h2>",
        (
            "<p>Raw treasury returns reflect percent changes in yield levels. "
            "Adjusted treasury returns use a duration-based approximation to "
            "proxy bond total returns, while keeping the treasury Price as the yield level.</p>"
        ),
        "<h3>Raw Annualized Stats (Monthly Assumption)</h3>",
        raw_ann_stats.to_html(),
        "<h3>Adjusted Annualized Stats (Duration-Based Treasury, Monthly Assumption)</h3>",
        adj_ann_stats.to_html(),
    ]
    if warning_line:
        html_parts.extend(["<h2>Warning</h2>", f"<p>{warning_line}</p>"])
    html_parts.append("</body></html>")

    html_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"HTML report written to: {html_path}")


# ==============================
# Script-equivalent workflows
# ==============================


def run_all(root: Path) -> None:
    prepare_data(root)
    run_data_qc(root)
    run_primary_variant1(root)
    run_benchmarks(root)


def setup_test_root(
    target_root: Path = Path("test"),
    source_raw: Path = Path("project_results/data/raw"),
    clean: bool = False,
) -> None:
    """Bootstrap an isolated test root with bundled raw input files."""
    target_root = (PROJECT_ROOT / target_root).resolve()
    source_raw = (PROJECT_ROOT / source_raw).resolve()

    if not source_raw.exists():
        raise FileNotFoundError(f"Source raw directory not found: {source_raw}")

    target_raw = target_root / "data" / "raw"
    target_clean = target_root / "data" / "clean"
    target_reports = target_root / "reports"

    if clean:
        if target_clean.exists():
            shutil.rmtree(target_clean)
        if target_reports.exists():
            shutil.rmtree(target_reports)

    target_raw.mkdir(parents=True, exist_ok=True)
    target_clean.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for path in sorted(source_raw.iterdir()):
        if path.is_file():
            dst = target_raw / path.name
            shutil.copy2(path, dst)
            copied.append(dst)

    print(f"Target root: {target_root}")
    print(f"Copied {len(copied)} raw files to: {target_raw}")
    for path in copied:
        print(f"- {path}")


def _parse_float_list(value: str) -> list[float]:
    out: list[float] = []
    for item in value.split(","):
        token = item.strip()
        if token:
            out.append(float(token))
    if not out:
        raise ValueError("At least one numeric value is required.")
    return out


def run_robustness(
    root: Path = Path("test"),
    out_dir: Path | None = None,
    tcost_grid_bps: str = "0,5,10,25,50",
    buy_grid: str = "0.0001,0.25,0.50",
    sell_grid: str = "-0.0001,-0.25,-0.50",
    duration_grid: str = "6.0,8.5,10.0",
) -> None:
    """Run robustness grid for costs, thresholds, and treasury duration."""
    root = root.resolve()
    out_dir = (out_dir or (root / "reports" / "robustness")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tcost_grid = _parse_float_list(tcost_grid_bps)
    buy_grid_values = _parse_float_list(buy_grid)
    sell_grid_values = _parse_float_list(sell_grid)
    duration_grid_values = _parse_float_list(duration_grid)

    clean_dir = root / "data" / "clean"
    universe = load_universe(clean_dir, list(DEFAULT_ASSETS))

    rows: list[dict[str, float | int | str]] = []
    scenario_counter = 0

    for duration in duration_grid_values:
        adjusted = apply_treasury_total_return(universe, duration=duration)
        returns = universe_returns_matrix(adjusted)
        equal_weight_row = pd.Series(
            1.0 / len(returns.columns), index=returns.columns, dtype=float
        )

        for buy_threshold in buy_grid_values:
            for sell_threshold in sell_grid_values:
                if buy_threshold <= sell_threshold:
                    continue

                signals = build_primary_signal_variant1(
                    adjusted,
                    buy_threshold=buy_threshold,
                    sell_threshold=sell_threshold,
                )
                signal_series = signals["signal"]
                signal_counts = (
                    signal_series.value_counts(dropna=False)
                    .reindex(["BUY", "HOLD", "SELL"], fill_value=0)
                    .astype(int)
                )

                weights = weights_from_primary_signal(
                    signal=signal_series,
                    returns_columns=list(returns.columns),
                )
                weights = weights.reindex(returns.index).ffill().fillna(equal_weight_row)

                scenario_counter += 1
                scenario_name = f"S{scenario_counter:03d}"

                for tcost_bps in tcost_grid:
                    backtest = backtest_from_weights(
                        returns=returns,
                        weights=weights,
                        tcost_bps=tcost_bps,
                    )
                    summary = perf_table({"PrimaryV1": backtest}).iloc[0]

                    rows.append(
                        {
                            "scenario": scenario_name,
                            "duration": duration,
                            "buy_threshold": buy_threshold,
                            "sell_threshold": sell_threshold,
                            "tcost_bps": tcost_bps,
                            "ann_return": float(summary["ann_return"]),
                            "ann_vol": float(summary["ann_vol"]),
                            "sharpe": float(summary["sharpe"]),
                            "max_drawdown": float(summary["max_drawdown"]),
                            "calmar": float(summary["calmar"]),
                            "avg_turnover": float(summary["avg_turnover"]),
                            "ending_equity_net": float(backtest["equity_net"].iloc[-1]),
                            "buy_count": int(signal_counts["BUY"]),
                            "hold_count": int(signal_counts["HOLD"]),
                            "sell_count": int(signal_counts["SELL"]),
                            "periods": int(len(backtest)),
                        }
                    )

    grid = pd.DataFrame(rows).sort_values(
        by=["sharpe", "ann_return"], ascending=[False, False]
    )
    grid_path = out_dir / "robustness_grid_results.csv"
    grid.to_csv(grid_path, index=False)

    baseline = grid[
        (grid["duration"].round(6) == 8.5)
        & (grid["buy_threshold"].round(6) == 0.0001)
        & (grid["sell_threshold"].round(6) == -0.0001)
    ].sort_values("tcost_bps")
    baseline_path = out_dir / "baseline_cost_sensitivity.csv"
    baseline.to_csv(baseline_path, index=False)

    top15_path = out_dir / "top15_by_sharpe.csv"
    grid.head(15).to_csv(top15_path, index=False)

    md_lines = [
        "# Robustness Summary",
        "",
        f"- Scenarios evaluated: `{len(grid)}`",
        f"- Duration grid: `{duration_grid_values}`",
        f"- Buy thresholds: `{buy_grid_values}`",
        f"- Sell thresholds: `{sell_grid_values}`",
        f"- Transaction-cost grid (bps): `{tcost_grid}`",
        "",
        "## Best Sharpe Scenario",
    ]
    best = grid.iloc[0]
    md_lines.extend(
        [
            f"- Scenario: `{best['scenario']}`",
            f"- Duration: `{best['duration']}`",
            f"- Thresholds: buy `{best['buy_threshold']}`, sell `{best['sell_threshold']}`",
            f"- Cost (bps): `{best['tcost_bps']}`",
            f"- Sharpe: `{best['sharpe']:.4f}`",
            f"- Ann. return: `{best['ann_return']:.4%}`",
            f"- Max drawdown: `{best['max_drawdown']:.4%}`",
            "",
            "## Outputs",
            f"- `{grid_path}`",
            f"- `{baseline_path}`",
            f"- `{top15_path}`",
        ]
    )
    (out_dir / "robustness_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved: {grid_path}")
    print(f"Saved: {baseline_path}")
    print(f"Saved: {top15_path}")
    print(f"Saved: {out_dir / 'robustness_summary.md'}")


def _strict_walk_forward(
    adjusted_universe: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    min_train_periods: int,
    buy_threshold: float,
    sell_threshold: float,
    tcost_bps: float,
) -> pd.DataFrame:
    if len(returns) <= min_train_periods + 1:
        raise ValueError("Not enough data for requested min_train_periods.")

    columns = list(returns.columns)
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    prev_weight: pd.Series | None = None

    for i in range(min_train_periods - 1, len(returns.index) - 1):
        decision_date = returns.index[i]
        realized_date = returns.index[i + 1]

        history_universe = {
            asset: df.loc[:decision_date].copy() for asset, df in adjusted_universe.items()
        }
        history_signals = build_primary_signal_variant1(
            history_universe,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )
        signal_series = history_signals["signal"]
        signal_t = str(signal_series.iloc[-1]) if pd.notna(signal_series.iloc[-1]) else "NaN"

        weights_hist = weights_from_primary_signal(signal_series, returns_columns=columns)
        weight_t = pd.to_numeric(weights_hist.iloc[-1], errors="coerce").fillna(0.0)
        weight_t = weight_t.clip(lower=0.0)
        denom = float(weight_t.sum())
        if denom > 0.0:
            weight_t = weight_t / denom

        next_rets = returns.loc[realized_date, columns]
        gross_return = float((weight_t * next_rets).sum())

        if prev_weight is None:
            turnover = 0.0
        else:
            turnover = 0.5 * float((weight_t - prev_weight).abs().sum())
        cost = turnover * (tcost_bps / 10000.0)
        net_return = gross_return - cost

        row: dict[str, float | str | pd.Timestamp] = {
            "decision_date": decision_date,
            "realized_date": realized_date,
            "signal": signal_t,
            "gross_return": gross_return,
            "net_return": net_return,
            "turnover": turnover,
        }
        for col in columns:
            row[f"w_{col}"] = float(weight_t[col])

        rows.append(row)
        prev_weight = weight_t

    out = pd.DataFrame(rows).set_index("decision_date").sort_index()
    out["equity_gross"] = (1.0 + out["gross_return"]).cumprod()
    out["equity_net"] = (1.0 + out["net_return"]).cumprod()
    return out


def run_walk_forward(
    root: Path = Path("test"),
    out_dir: Path | None = None,
    min_train_periods: int = 120,
    duration: float = 8.5,
    buy_threshold: float = 0.0001,
    sell_threshold: float = -0.0001,
    tcost_bps: float = 0.0,
) -> None:
    """Run strict walk-forward validation for PrimaryV1."""
    root = root.resolve()
    out_dir = (out_dir or (root / "reports" / "walk_forward")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_dir = root / "data" / "clean"
    universe = load_universe(clean_dir, list(DEFAULT_ASSETS))
    adjusted_universe = apply_treasury_total_return(universe, duration=duration)
    returns = universe_returns_matrix(adjusted_universe)

    wf_backtest = _strict_walk_forward(
        adjusted_universe=adjusted_universe,
        returns=returns,
        min_train_periods=min_train_periods,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        tcost_bps=tcost_bps,
    )

    full_signals = build_primary_signal_variant1(
        adjusted_universe,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )
    full_weights = weights_from_primary_signal(
        signal=full_signals["signal"],
        returns_columns=list(returns.columns),
    )
    equal_weight_row = pd.Series(1.0 / len(returns.columns), index=returns.columns, dtype=float)
    full_weights = full_weights.reindex(returns.index).ffill().fillna(equal_weight_row)
    full_backtest = backtest_from_weights(returns=returns, weights=full_weights, tcost_bps=tcost_bps)
    full_backtest = full_backtest.loc[wf_backtest.index.min() :].copy()

    summary = perf_table(
        {
            "WalkForwardStrict": wf_backtest,
            "StandardCausal": full_backtest,
        }
    )
    summary["delta_vs_standard"] = summary["ann_return"] - summary.loc["StandardCausal", "ann_return"]

    wf_path = out_dir / "walk_forward_backtest.csv"
    std_path = out_dir / "standard_causal_backtest_slice.csv"
    summary_path = out_dir / "walk_forward_summary.csv"
    wf_backtest.to_csv(wf_path, index=True)
    full_backtest.to_csv(std_path, index=True)
    summary.to_csv(summary_path, index=True)

    protocol = [
        "# Walk-Forward Protocol",
        "",
        "- Train window: expanding from first observation to decision date `t`.",
        "- Decision at `t`: compute signal using only data through `t`.",
        "- Realization at `t+1`: apply weight decided at `t` to next-period return.",
        f"- Minimum train periods before first OOS decision: `{min_train_periods}`.",
        f"- Treasury duration assumption: `{duration}`.",
        f"- Thresholds: buy `{buy_threshold}`, sell `{sell_threshold}`.",
        f"- Transaction cost: `{tcost_bps}` bps.",
        f"- OOS decisions evaluated: `{len(wf_backtest)}`.",
        f"- First OOS decision date: `{wf_backtest.index.min().date().isoformat()}`.",
        f"- Last OOS decision date: `{wf_backtest.index.max().date().isoformat()}`.",
        "",
        "## Outputs",
        f"- `{wf_path}`",
        f"- `{std_path}`",
        f"- `{summary_path}`",
    ]
    (out_dir / "walk_forward_protocol.md").write_text("\n".join(protocol), encoding="utf-8")

    print(f"Saved: {wf_path}")
    print(f"Saved: {std_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {out_dir / 'walk_forward_protocol.md'}")


def _git_commit_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(PROJECT_ROOT.parent), "rev-parse", "HEAD"], text=True
            )
            .strip()
        )
    except Exception:
        return "unknown"


def _run_and_log(command: list[str], log_path: Path) -> None:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        env=env,
    )

    log_lines = [
        f"$ {' '.join(shlex.quote(tok) for tok in command)}",
        "",
        "STDOUT:",
        completed.stdout,
        "",
        "STDERR:",
        completed.stderr,
        "",
        f"exit_code={completed.returncode}",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    print(f"Ran: {' '.join(command)}")
    print(f"Log: {log_path}")
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {command}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_validation_suite(
    root: Path = Path("test"),
    clean_root: bool = False,
    skip_pytest: bool = False,
    skip_robustness: bool = False,
    skip_walk_forward: bool = False,
) -> None:
    """Run full isolated validation suite and write reproducibility artifacts."""
    target_root = (PROJECT_ROOT / root).resolve()
    reports_dir = target_root / "reports"
    repro_dir = reports_dir / "reproducibility"
    repro_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()

    commands: list[tuple[str, list[str]]] = []

    setup_cmd = [
        sys.executable,
        str(script_path),
        "setup-test-root",
        "--target-root",
        str(root),
    ]
    if clean_root:
        setup_cmd.append("--clean")
    commands.append(("setup_test_root", setup_cmd))

    if not skip_pytest:
        commands.append(("pytest", [sys.executable, "-m", "pytest", "-q"]))

    commands.append(
        (
            "run_all",
            [sys.executable, str(script_path), "run-all", "--root", str(root)],
        )
    )

    if not skip_robustness:
        commands.append(
            (
                "run_robustness",
                [
                    sys.executable,
                    str(script_path),
                    "run-robustness",
                    "--root",
                    str(root),
                    "--out-dir",
                    str(target_root / "reports" / "robustness"),
                ],
            )
        )

    if not skip_walk_forward:
        commands.append(
            (
                "run_walk_forward",
                [
                    sys.executable,
                    str(script_path),
                    "run-walk-forward",
                    "--root",
                    str(root),
                    "--out-dir",
                    str(target_root / "reports" / "walk_forward"),
                ],
            )
        )

    for i, (label, command) in enumerate(commands, start=1):
        log_path = repro_dir / f"{i:02d}_{label}.log"
        _run_and_log(command, log_path)

    artifacts: dict[str, str] = {}
    if reports_dir.exists():
        for path in sorted(reports_dir.rglob("*")):
            if path.is_file():
                artifacts[str(path.relative_to(PROJECT_ROOT))] = _sha256(path)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "isolated_root": str(target_root),
        "git_commit": _git_commit_sha(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "commands": [command for _, command in commands],
        "artifact_sha256": artifacts,
    }
    manifest_path = repro_dir / "validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary_lines = [
        "# Validation Suite Summary",
        "",
        f"- Isolated root: `{target_root}`",
        f"- Git commit: `{manifest['git_commit']}`",
        f"- Python: `{manifest['python_version']}`",
        f"- Platform: `{manifest['platform']}`",
        f"- Commands executed: `{len(commands)}`",
        f"- Hashed artifacts: `{len(artifacts)}`",
        "",
        "## Key Outputs",
        f"- `{target_root / 'reports' / 'benchmarks_summary.csv'}`",
        f"- `{target_root / 'reports' / 'primary_v1_summary.csv'}`",
        f"- `{target_root / 'reports' / 'robustness' / 'robustness_grid_results.csv'}`",
        f"- `{target_root / 'reports' / 'walk_forward' / 'walk_forward_summary.csv'}`",
        f"- `{manifest_path}`",
    ]
    summary_path = repro_dir / "validation_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Saved: {manifest_path}")
    print(f"Saved: {summary_path}")
    print("Validation suite completed successfully.")


def _run_subprocess(command: list[str]) -> None:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    print(f"$ {' '.join(shlex.quote(tok) for tok in command)}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def run_modes(mode: str) -> None:
    """Simple runner for project mode, test mode, or both."""
    script_path = Path(__file__).resolve()
    project_cmd = [sys.executable, str(script_path), "run-all"]
    test_cmd = [
        sys.executable,
        str(script_path),
        "run-validation-suite",
        "--root",
        "test",
        "--clean-root",
    ]

    if mode == "project":
        _run_subprocess(project_cmd)
        return
    if mode == "test":
        _run_subprocess(test_cmd)
        return

    _run_subprocess(project_cmd)
    _run_subprocess(test_cmd)


# ==============================
# Unified CLI
# ==============================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-file runner for the full primary_model project."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    core_help = {
        "prepare-data": "Run raw Excel to clean CSV data preparation.",
        "data-qc": "Run data quality report generation.",
        "run-primary-v1": "Run PrimaryV1 strategy and summary output.",
        "run-benchmarks": "Run benchmark suite and reports.",
        "run-all": "Run full core pipeline (prepare-data, data-qc, strategy, benchmarks).",
    }
    for cmd, help_text in core_help.items():
        core_parser = subparsers.add_parser(cmd, help=help_text)
        core_parser.add_argument(
            "--root",
            type=Path,
            default=PROJECT_ROOT,
            help="Project root containing data/, reports/, and inputs.",
        )

    mode_parser = subparsers.add_parser(
        "run-modes",
        aliases=["run_modes"],
        help="Run project workflow, validation workflow, or both.",
    )
    mode_parser.add_argument(
        "--mode",
        choices=["project", "test", "both"],
        required=True,
        help="Which execution mode to run.",
    )

    setup_parser = subparsers.add_parser(
        "setup-test-root",
        aliases=["setup_test_root"],
        help="Bootstrap isolated test root with raw input files.",
    )
    setup_parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("test"),
        help="Target isolated root to populate (default: test).",
    )
    setup_parser.add_argument(
        "--source-raw",
        type=Path,
        default=Path("project_results/data/raw"),
        help="Source raw-data directory with Excel files.",
    )
    setup_parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing target data/clean and reports before setup.",
    )

    robustness_parser = subparsers.add_parser(
        "run-robustness",
        aliases=["run_robustness"],
        help="Run robustness grid for costs, thresholds, and duration assumptions.",
    )
    robustness_parser.add_argument(
        "--root",
        type=Path,
        default=Path("test"),
        help="Project root containing data/clean for evaluation.",
    )
    robustness_parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <root>/reports/robustness).",
    )
    robustness_parser.add_argument(
        "--tcost-grid-bps",
        type=str,
        default="0,5,10,25,50",
        help="Comma-separated transaction cost grid in bps.",
    )
    robustness_parser.add_argument(
        "--buy-grid",
        type=str,
        default="0.0001,0.25,0.50",
        help="Comma-separated buy-threshold grid.",
    )
    robustness_parser.add_argument(
        "--sell-grid",
        type=str,
        default="-0.0001,-0.25,-0.50",
        help="Comma-separated sell-threshold grid.",
    )
    robustness_parser.add_argument(
        "--duration-grid",
        type=str,
        default="6.0,8.5,10.0",
        help="Comma-separated treasury-duration grid.",
    )

    walk_parser = subparsers.add_parser(
        "run-walk-forward",
        aliases=["run_walk_forward"],
        help="Run strict walk-forward validation.",
    )
    walk_parser.add_argument(
        "--root",
        type=Path,
        default=Path("test"),
        help="Project root containing data/clean for evaluation.",
    )
    walk_parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <root>/reports/walk_forward).",
    )
    walk_parser.add_argument(
        "--min-train-periods",
        type=int,
        default=120,
        help="Minimum in-sample history before first OOS decision.",
    )
    walk_parser.add_argument(
        "--duration",
        type=float,
        default=8.5,
        help="Treasury duration used for return approximation.",
    )
    walk_parser.add_argument(
        "--buy-threshold",
        type=float,
        default=0.0001,
        help="BUY threshold for signal mapping.",
    )
    walk_parser.add_argument(
        "--sell-threshold",
        type=float,
        default=-0.0001,
        help="SELL threshold for signal mapping.",
    )
    walk_parser.add_argument(
        "--tcost-bps",
        type=float,
        default=0.0,
        help="Transaction costs in bps.",
    )

    validation_parser = subparsers.add_parser(
        "run-validation-suite",
        aliases=["run_validation_suite"],
        help="Run full isolated validation suite and reproducibility logging.",
    )
    validation_parser.add_argument(
        "--root",
        type=Path,
        default=Path("test"),
        help="Isolated root for data/reports (default: test).",
    )
    validation_parser.add_argument(
        "--clean-root",
        action="store_true",
        help="Clean target root reports/clean data before running.",
    )
    validation_parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip unit/integration tests.",
    )
    validation_parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Skip robustness grid.",
    )
    validation_parser.add_argument(
        "--skip-walk-forward",
        action="store_true",
        help="Skip strict walk-forward run.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")

    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command

    if command in {"prepare-data", "data-qc", "run-primary-v1", "run-benchmarks", "run-all"}:
        root = args.root.resolve()
        if command == "prepare-data":
            prepare_data(root)
        elif command == "data-qc":
            run_data_qc(root)
        elif command == "run-primary-v1":
            run_primary_variant1(root)
        elif command == "run-benchmarks":
            run_benchmarks(root)
        elif command == "run-all":
            run_all(root)
        return

    if command in {"run-modes", "run_modes"}:
        run_modes(args.mode)
        return

    if command in {"setup-test-root", "setup_test_root"}:
        setup_test_root(
            target_root=args.target_root,
            source_raw=args.source_raw,
            clean=args.clean,
        )
        return

    if command in {"run-robustness", "run_robustness"}:
        run_robustness(
            root=args.root,
            out_dir=args.out_dir,
            tcost_grid_bps=args.tcost_grid_bps,
            buy_grid=args.buy_grid,
            sell_grid=args.sell_grid,
            duration_grid=args.duration_grid,
        )
        return

    if command in {"run-walk-forward", "run_walk_forward"}:
        run_walk_forward(
            root=args.root,
            out_dir=args.out_dir,
            min_train_periods=args.min_train_periods,
            duration=args.duration,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
            tcost_bps=args.tcost_bps,
        )
        return

    if command in {"run-validation-suite", "run_validation_suite"}:
        run_validation_suite(
            root=args.root,
            clean_root=args.clean_root,
            skip_pytest=args.skip_pytest,
            skip_robustness=args.skip_robustness,
            skip_walk_forward=args.skip_walk_forward,
        )
        return

    parser.error(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
