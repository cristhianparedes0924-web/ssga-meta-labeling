"""Treasury return approximation utilities."""

from __future__ import annotations

import pandas as pd


def compute_bond_total_return_from_yield(
    yield_level_percent: pd.Series,
    duration: float = 8.5,
    periods_per_year: int = 12,
    include_carry: bool = True,
) -> pd.Series:
    """Approximate bond total return from a yield-level series.

    `yield_level_percent` is a yield series (in percent), not a price index.
    The calculation uses a modified-duration approximation:
    `price_return = -duration * delta(yield_decimal)`.
    `duration` is an assumption and should be sensitivity-tested later.
    """
    y = pd.to_numeric(yield_level_percent, errors="coerce") / 100.0
    dy = y.diff()
    price_return = -duration * dy

    if include_carry:
        carry = y.shift(1) / periods_per_year
        total_return = price_return + carry
    else:
        total_return = price_return

    return total_return
