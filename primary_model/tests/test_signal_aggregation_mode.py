from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from primary_model.signals.variant1 import build_primary_signal_variant1


def _mock_universe(periods: int = 96) -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2010-01-31", periods=periods, freq="ME")
    t = np.arange(periods, dtype=float)

    spx_ret = 0.004 + 0.015 * np.sin(t / 6.0)
    bcom_ret = 0.002 + 0.012 * np.sin(t / 5.0 + 0.4)
    corp_ret = 0.003 + 0.008 * np.cos(t / 7.0)

    spx_price = 100.0 * np.cumprod(1.0 + spx_ret)
    bcom_price = 90.0 * np.cumprod(1.0 + bcom_ret)
    corp_price = 110.0 * np.cumprod(1.0 + corp_ret)
    ust_yield = 2.0 + 0.5 * np.sin(t / 9.0) + 0.3 * (t / periods)
    ust_ret = np.full(periods, 0.001)

    def _frame(price: np.ndarray, ret: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({"Price": price, "Return": ret}, index=dates)

    return {
        "spx": _frame(spx_price, spx_ret),
        "bcom": _frame(bcom_price, bcom_ret),
        "treasury_10y": _frame(ust_yield, ust_ret),
        "corp_bonds": _frame(corp_price, corp_ret),
    }


def test_equal_weight_aggregation_mode_runs() -> None:
    universe = _mock_universe(periods=120)
    signal_frame = build_primary_signal_variant1(universe, aggregation_mode="equal_weight")
    assert "composite_score" in signal_frame.columns
    assert signal_frame["composite_score"].notna().sum() > 24


def test_invalid_aggregation_mode_raises() -> None:
    universe = _mock_universe(periods=60)
    with pytest.raises(ValueError, match="aggregation_mode"):
        build_primary_signal_variant1(universe, aggregation_mode="not_a_mode")


def test_indicator_weights_override_aggregation_mode() -> None:
    universe = _mock_universe(periods=120)
    custom_weights = {
        "spx_trend_z": 1.0,
        "bcom_trend_z": 1.0,
        "credit_vs_rates_z": 1.0,
        "risk_breadth_z": 1.0,
    }
    dynamic = build_primary_signal_variant1(
        universe,
        aggregation_mode="dynamic",
        indicator_weights=custom_weights,
    )
    equal = build_primary_signal_variant1(
        universe,
        aggregation_mode="equal_weight",
        indicator_weights=custom_weights,
    )
    np.testing.assert_allclose(
        dynamic["composite_score"].to_numpy(),
        equal["composite_score"].to_numpy(),
        equal_nan=True,
        atol=1e-12,
    )
