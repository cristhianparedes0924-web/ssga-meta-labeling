from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.engine import backtest_from_weights
from signals.variant1 import build_primary_signal_variant1, expanding_zscore


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


def test_expanding_zscore_is_prefix_invariant() -> None:
    base = pd.Series([1.0, 1.1, 0.9, 1.2, 1.0, 1.3], dtype=float)
    extended = pd.concat([base, pd.Series([500.0, -400.0], dtype=float)], ignore_index=True)

    z_base = expanding_zscore(base, min_periods=3)
    z_extended = expanding_zscore(extended, min_periods=3).iloc[: len(base)]

    np.testing.assert_allclose(
        z_base.to_numpy(),
        z_extended.to_numpy(),
        equal_nan=True,
        atol=1e-12,
    )


def test_primary_signal_has_no_future_dependency_in_prefix() -> None:
    universe = _mock_universe(periods=96)
    full = build_primary_signal_variant1(universe)

    cutoff = full.index[79]
    truncated_universe = {asset: frame.loc[:cutoff].copy() for asset, frame in universe.items()}
    truncated = build_primary_signal_variant1(truncated_universe)

    common = truncated.index
    numeric_cols = [col for col in truncated.columns if col != "signal"]

    np.testing.assert_allclose(
        full.loc[common, numeric_cols].to_numpy(),
        truncated[numeric_cols].to_numpy(),
        equal_nan=True,
        atol=1e-12,
    )
    assert full.loc[common, "signal"].fillna("NaN").tolist() == truncated["signal"].fillna("NaN").tolist()


def test_backtest_uses_next_period_returns() -> None:
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    returns = pd.DataFrame(
        {
            "a": [0.10, 0.20, 0.30, 0.40],
            "b": [-0.10, -0.20, -0.30, -0.40],
        },
        index=dates,
    )
    weights = pd.DataFrame(
        {
            "a": [1.0, 0.0, 1.0, 0.0],
            "b": [0.0, 1.0, 0.0, 1.0],
        },
        index=dates,
    )

    backtest = backtest_from_weights(returns=returns, weights=weights, tcost_bps=0.0)

    expected = pd.Series([0.20, -0.30, 0.40], index=dates[:3], dtype=float)
    np.testing.assert_allclose(
        backtest["gross_return"].to_numpy(),
        expected.to_numpy(),
        atol=1e-12,
    )
