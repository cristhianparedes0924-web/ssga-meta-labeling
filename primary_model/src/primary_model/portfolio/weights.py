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


from primary_model.benchmarks.static import weights_equal_weight, weights_buy_hold_spx, weights_6040
from primary_model.benchmarks.trend import weights_simple_trend


__all__ = [
    "primary_signal_to_weights",
    "weights_6040",
    "weights_buy_hold_spx",
    "weights_equal_weight",
    "weights_from_primary_signal",
    "weights_simple_trend",
]
