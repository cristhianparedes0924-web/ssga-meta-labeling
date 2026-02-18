"""Benchmark weight generators."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _empty_weights(index: pd.Index, columns: Iterable[str]) -> pd.DataFrame:
    cols = list(columns)
    return pd.DataFrame(0.0, index=index, columns=cols, dtype=float)


def weights_equal_weight(returns: pd.DataFrame) -> pd.DataFrame:
    """Build equal-weight portfolio weights for all dates/assets."""
    if returns.shape[1] == 0:
        raise ValueError("returns must contain at least one asset column.")

    w = _empty_weights(returns.index, returns.columns)
    w.loc[:, :] = 1.0 / returns.shape[1]
    return w


def weights_buy_hold_spx(returns: pd.DataFrame, spx_col: str = "spx") -> pd.DataFrame:
    """Build static 100% SPX weights aligned to returns."""
    if spx_col not in returns.columns:
        raise ValueError(f"Column '{spx_col}' not found in returns.")

    w = _empty_weights(returns.index, returns.columns)
    w[spx_col] = 1.0
    return w


def weights_6040(
    returns: pd.DataFrame, spx_col: str = "spx", ust_col: str = "treasury_10y"
) -> pd.DataFrame:
    """Build static 60/40 SPX/Treasury weights aligned to returns."""
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
    """Build trend-following weights using fast/slow SMA on a price series.

    At time ``t``:
    - if ``SMA_fast(t) > SMA_slow(t)``, allocate 100% to ``risk_on`` assets
    - else allocate 100% to ``risk_off`` assets
    with equal weights within the active bucket.
    """
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
