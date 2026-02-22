"""Portfolio weight construction functions for model and benchmark strategies."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


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


__all__ = [
    "primary_signal_to_weights",
    "weights_6040",
    "weights_buy_hold_spx",
    "weights_equal_weight",
    "weights_from_primary_signal",
    "weights_simple_trend",
]
