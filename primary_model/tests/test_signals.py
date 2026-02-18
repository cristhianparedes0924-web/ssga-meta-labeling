from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from primary_model.signals import (  # noqa: E402
    build_primary_signal_variant1,
    expanding_zscore,
    score_to_signal,
)


def _asset_frame(prices: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"Price": prices, "Return": prices.pct_change()})


def test_expanding_zscore_uses_only_history() -> None:
    index = pd.date_range("2020-01-31", periods=4, freq="ME")
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)

    z = expanding_zscore(series, min_periods=2, ddof=1)

    assert pd.isna(z.iloc[0])
    assert pd.isna(z.iloc[1])
    assert z.iloc[2] == pytest.approx((3.0 - 1.5) / np.sqrt(0.5))
    assert z.iloc[3] == pytest.approx((4.0 - 2.0) / 1.0)


def test_score_to_signal_thresholds() -> None:
    index = pd.date_range("2020-01-31", periods=5, freq="ME")
    score = pd.Series([-1.0, -0.5, 0.0, 0.5, 0.8], index=index)

    signal = score_to_signal(score, buy_threshold=0.5, sell_threshold=-0.5)

    assert signal.tolist() == ["SELL", "HOLD", "HOLD", "HOLD", "BUY"]


def test_build_primary_signal_variant1_output_schema() -> None:
    dates = pd.date_range("2018-01-31", periods=36, freq="ME")
    spx_price = pd.Series(np.linspace(100.0, 190.0, len(dates)), index=dates)
    bcom_price = pd.Series(np.linspace(90.0, 130.0, len(dates)), index=dates)
    treasury_yield = pd.Series(np.linspace(3.0, 4.0, len(dates)), index=dates)
    corp_price = pd.Series(np.linspace(100.0, 125.0, len(dates)), index=dates)

    universe = {
        "spx": _asset_frame(spx_price),
        "bcom": _asset_frame(bcom_price),
        "treasury_10y": _asset_frame(treasury_yield),
        "corp_bonds": _asset_frame(corp_price),
    }

    out = build_primary_signal_variant1(
        universe=universe,
        trend_window=6,
        relative_window=3,
        zscore_min_periods=6,
    )

    expected_cols = [
        "spx_trend",
        "bcom_trend",
        "credit_vs_rates",
        "risk_breadth",
        "spx_trend_z",
        "bcom_trend_z",
        "credit_vs_rates_z",
        "risk_breadth_z",
        "composite_score",
        "signal",
    ]
    assert out.columns.tolist() == expected_cols
    assert out.index.equals(dates)

    valid = {"BUY", "HOLD", "SELL"}
    realized = set(out["signal"].dropna().unique().tolist())
    assert realized.issubset(valid)
    assert len(out["composite_score"].dropna()) > 0
