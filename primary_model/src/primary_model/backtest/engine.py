"""Backtest engine and performance metric functions."""

from __future__ import annotations

import numpy as np
import pandas as pd


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


__all__ = [
    "backtest_from_weights",
]
