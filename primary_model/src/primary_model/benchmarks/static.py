"""
Static benchmarks module.

TODO: Implement static allocation baselines like EqualWeight25, BuyHoldSPX, and 60/40.
"""

import pandas as pd
from typing import Iterable

def _empty_weights(index: pd.Index, columns: Iterable[str]) -> pd.DataFrame:
    cols = list(columns)
    return pd.DataFrame(0.0, index=index, columns=cols, dtype=float)

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
