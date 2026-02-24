"""
Trend benchmarks module.

TODO: Implement simple moving average crossover and basic trend benchmarks.
"""

import pandas as pd
from primary_model.benchmarks.static import _empty_weights

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
