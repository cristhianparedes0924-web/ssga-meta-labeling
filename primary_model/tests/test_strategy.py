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

from primary_model.strategy import weights_from_primary_signal  # noqa: E402


def test_weights_from_primary_signal_buy_hold_sell_hold() -> None:
    dates = pd.date_range("2024-01-31", periods=4, freq="ME")
    signal = pd.Series(["BUY", "HOLD", "SELL", "HOLD"], index=dates)
    returns_columns = ["spx", "bcom", "treasury_10y", "corp_bonds"]

    w = weights_from_primary_signal(
        signal=signal,
        returns_columns=returns_columns,
    )

    buy_expected = pd.Series(
        {"spx": 1.0 / 3.0, "bcom": 1.0 / 3.0, "treasury_10y": 0.0, "corp_bonds": 1.0 / 3.0}
    )
    sell_expected = pd.Series({"spx": 0.0, "bcom": 0.0, "treasury_10y": 1.0, "corp_bonds": 0.0})

    pd.testing.assert_series_equal(w.iloc[0], buy_expected, check_names=False)
    pd.testing.assert_series_equal(w.iloc[1], w.iloc[0], check_names=False)
    pd.testing.assert_series_equal(w.iloc[2], sell_expected, check_names=False)
    pd.testing.assert_series_equal(w.iloc[3], w.iloc[2], check_names=False)

    assert np.allclose(w.sum(axis=1).values, np.ones(len(w)))
    assert (w >= 0.0).all().all()
    assert w.iloc[3]["treasury_10y"] == pytest.approx(1.0)


def test_weights_from_primary_signal_pre_signal_mode_risk_off() -> None:
    dates = pd.date_range("2024-01-31", periods=3, freq="ME")
    signal = pd.Series([np.nan, "BUY", "HOLD"], index=dates)
    returns_columns = ["spx", "bcom", "treasury_10y", "corp_bonds"]

    w = weights_from_primary_signal(
        signal=signal,
        returns_columns=returns_columns,
        pre_signal_mode="risk_off",
    )

    assert w.iloc[0]["treasury_10y"] == pytest.approx(1.0)
    assert w.iloc[0][["spx", "bcom", "corp_bonds"]].sum() == pytest.approx(0.0)
    assert w.iloc[1]["spx"] == pytest.approx(1.0 / 3.0)
    assert w.iloc[1]["bcom"] == pytest.approx(1.0 / 3.0)
    assert w.iloc[1]["corp_bonds"] == pytest.approx(1.0 / 3.0)
    assert w.iloc[1]["treasury_10y"] == pytest.approx(0.0)
    pd.testing.assert_series_equal(w.iloc[2], w.iloc[1], check_names=False)
    assert np.allclose(w.sum(axis=1).values, np.ones(len(w)))
    assert (w >= 0.0).all().all()


def test_weights_from_primary_signal_leading_nans_are_safe_carry() -> None:
    dates = pd.date_range("2024-01-31", periods=5, freq="ME")
    signal = pd.Series([np.nan, np.nan, "SELL", np.nan, "HOLD"], index=dates)
    returns_columns = ["spx", "bcom", "treasury_10y", "corp_bonds"]

    w = weights_from_primary_signal(
        signal=signal,
        returns_columns=returns_columns,
        pre_signal_mode="equal_weight",
    )

    expected_equal = pd.Series(
        {"spx": 0.25, "bcom": 0.25, "treasury_10y": 0.25, "corp_bonds": 0.25}
    )
    pd.testing.assert_series_equal(w.iloc[0], expected_equal, check_names=False)
    pd.testing.assert_series_equal(w.iloc[1], expected_equal, check_names=False)
    assert w.iloc[2]["treasury_10y"] == pytest.approx(1.0)
    assert w.iloc[2][["spx", "bcom", "corp_bonds"]].sum() == pytest.approx(0.0)
    pd.testing.assert_series_equal(w.iloc[3], w.iloc[2], check_names=False)
    pd.testing.assert_series_equal(w.iloc[4], w.iloc[3], check_names=False)
    assert np.allclose(w.sum(axis=1).values, np.ones(len(w)))
    assert (w >= 0.0).all().all()
