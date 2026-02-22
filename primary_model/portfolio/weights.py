"""Portfolio-weight adapters over the frozen unified core."""

from __future__ import annotations

import pandas as pd

from primary_model_unified import (
    weights_6040,
    weights_buy_hold_spx,
    weights_equal_weight,
    weights_from_primary_signal,
    weights_simple_trend,
)


def primary_signal_to_weights(signal: pd.Series, returns_columns: list[str]) -> pd.DataFrame:
    """Alias for converting BUY/HOLD/SELL signals into long-only weights."""
    return weights_from_primary_signal(signal=signal, returns_columns=returns_columns)


__all__ = [
    "primary_signal_to_weights",
    "weights_6040",
    "weights_buy_hold_spx",
    "weights_equal_weight",
    "weights_from_primary_signal",
    "weights_simple_trend",
]
