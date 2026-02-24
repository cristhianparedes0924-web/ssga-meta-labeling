"""
Tests for the newly migrated analytics package.
Proves that performance metrics and diagnostics run predictably on minimal synthetic datastreams.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from primary_model.analytics.performance import (
    annualized_return,
    annualized_vol,
    max_drawdown,
    sharpe_ratio,
    perf_table,
)
from primary_model.analytics.diagnostics import asset_sanity_table


def test_annualized_return_synthetic():
    """Verify return math is stable out to 12-month extrapolations."""
    # 10% per month compounding
    r = pd.Series([0.1] * 12)
    val = annualized_return(r, periods_per_year=12)
    # (1.1^12) - 1
    expected = (1.1 ** 12) - 1.0
    assert_allclose(val, expected, rtol=1e-5)


def test_max_drawdown_synthetic():
    """Verify drawdown picks the deepest valley correctly."""
    equity = pd.Series([1.0, 1.1, 0.99, 1.2, 0.6, 0.8])
    # Peak is 1.2
    # Trough after peak is 0.6
    # Drawdown = 0.6/1.2 - 1 = -0.50
    val = max_drawdown(equity)
    assert_allclose(val, -0.50, rtol=1e-5)


def test_perf_table_empty():
    """Verify table builder fails cleanly on empty data."""
    df = perf_table({})
    assert df.empty
    assert "ann_return" in df.columns


def test_diagnostics_imports_cleanly():
    """Verify the sanity table builder binds correctly."""
    returns = pd.DataFrame({"A": [0.01, -0.01], "B": [0.02, 0.0]})
    table = asset_sanity_table(returns)
    assert len(table) == 2
    assert "ann_mean" in table.columns
    assert "ann_vol" in table.columns
