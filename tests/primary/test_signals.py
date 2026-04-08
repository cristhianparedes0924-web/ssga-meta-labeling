from __future__ import annotations

import numpy as np
import pandas as pd

from metalabel.primary.signals import (
    _dynamic_composite_score,
    build_primary_signal_variant1,
    score_to_signal,
)


def _sample_universe(n: int = 36) -> dict[str, pd.DataFrame]:
    idx = pd.date_range("1999-01-31", periods=n, freq="ME")
    spx_price = pd.Series(np.linspace(100.0, 140.0, n), index=idx)
    bcom_price = pd.Series(np.linspace(80.0, 90.0, n), index=idx)
    ust_yield = pd.Series(np.linspace(3.5, 4.0, n), index=idx)
    corp_price = pd.Series(np.linspace(95.0, 105.0, n), index=idx)

    return {
        "spx": pd.DataFrame({"Price": spx_price, "Return": spx_price.pct_change().fillna(0.0)}, index=idx),
        "bcom": pd.DataFrame({"Price": bcom_price, "Return": bcom_price.pct_change().fillna(0.0)}, index=idx),
        "treasury_10y": pd.DataFrame({"Price": ust_yield, "Return": pd.Series(np.linspace(0.001, 0.002, n), index=idx)}, index=idx),
        "corp_bonds": pd.DataFrame({"Price": corp_price, "Return": corp_price.pct_change().fillna(0.0)}, index=idx),
    }


def test_score_to_signal_respects_boundaries() -> None:
    score = pd.Series([-0.50, -0.31, 0.0, 0.31, 0.50], index=pd.date_range("2000-01-31", periods=5, freq="ME"))
    signal = score_to_signal(score, buy_threshold=0.31, sell_threshold=-0.31)
    assert signal.tolist() == ["SELL", "HOLD", "HOLD", "HOLD", "BUY"]


def test_build_primary_signal_variant1_emits_expected_columns() -> None:
    signals = build_primary_signal_variant1(_sample_universe())
    expected = {
        "spx_trend",
        "bcom_trend",
        "credit_vs_rates",
        "risk_breadth",
        "bcom_accel",
        "yield_mom",
        "spx_trend_z",
        "bcom_trend_z",
        "credit_vs_rates_z",
        "risk_breadth_z",
        "bcom_accel_z",
        "yield_mom_z",
        "composite_score",
        "signal",
    }
    assert expected.issubset(signals.columns)


def test_dynamic_composite_score_smoke_returns_indexed_series() -> None:
    idx = pd.date_range("2001-01-31", periods=60, freq="ME")
    zscores = pd.DataFrame(
        {
            "spx_trend_z": np.linspace(-1.5, 1.5, 60),
            "credit_vs_rates_z": np.sin(np.linspace(0.0, 4.0, 60)),
            "risk_breadth_z": np.cos(np.linspace(0.0, 4.0, 60)),
            "bcom_accel_z": np.linspace(0.8, -0.8, 60),
            "yield_mom_z": np.linspace(-0.5, 0.5, 60),
            "bcom_trend_z": np.nan,
        },
        index=idx,
    )
    target_ret = pd.Series(np.linspace(-0.03, 0.04, 60), index=idx)
    spx_returns = pd.Series(np.linspace(-0.02, 0.03, 60), index=idx)

    score = _dynamic_composite_score(zscores=zscores, target_ret=target_ret, spx_returns=spx_returns)

    assert isinstance(score, pd.Series)
    pd.testing.assert_index_equal(score.index, idx)
    assert score.notna().any()


def test_dynamic_composite_score_uses_only_active_columns_for_ic_windows() -> None:
    idx = pd.date_range("2002-01-31", periods=60, freq="ME")
    active_signal = pd.Series(np.linspace(-1.0, 1.0, 60), index=idx)
    zscores = pd.DataFrame(
        {
            "spx_trend_z": active_signal,
            "bcom_trend_z": np.nan,
        },
        index=idx,
    )
    target_ret = active_signal.shift(1).fillna(0.0)

    score = _dynamic_composite_score(zscores=zscores, target_ret=target_ret)

    assert pd.notna(score.iloc[-1])
    assert score.iloc[-1] == active_signal.iloc[-1]
