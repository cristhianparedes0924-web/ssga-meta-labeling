from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import metalabel.primary.signals as signals_module
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


def _long_dynamic_score_inputs(n: int = 160) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    idx = pd.date_range("2001-01-31", periods=n, freq="ME")
    phase = np.linspace(0.0, 10.0, n)
    zscores = pd.DataFrame(
        {
            "spx_trend_z": np.sin(phase),
            "credit_vs_rates_z": np.cos(phase),
            "risk_breadth_z": np.sin(phase * 0.5),
            "bcom_accel_z": np.cos(phase * 0.5),
            "yield_mom_z": np.linspace(-0.5, 0.5, n),
            "bcom_trend_z": np.nan,
        },
        index=idx,
    )
    target_ret = (
        0.6 * zscores["spx_trend_z"].shift(1).fillna(0.0)
        - 0.3 * zscores["credit_vs_rates_z"].shift(1).fillna(0.0)
        + 0.1 * zscores["yield_mom_z"].shift(1).fillna(0.0)
    )
    spx_returns = pd.Series(
        np.concatenate(
            [
                0.01 * np.sin(np.linspace(0.0, 9.0, n - 40)),
                0.04 * np.sin(np.linspace(0.0, 6.0, 40)),
            ]
        ),
        index=idx,
    )
    return zscores, target_ret, spx_returns


def test_dynamic_composite_score_enters_hmm_branch_when_proxy_history_is_long_enough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zscores, target_ret, spx_returns = _long_dynamic_score_inputs()
    hmm_fit_lengths: list[int] = []
    hmm_predict_lengths: list[int] = []
    ic_window_lengths: list[int] = []

    class FakeGaussianHMM:
        def __init__(self, n_components: int, covariance_type: str, n_iter: int, random_state: int) -> None:
            assert n_components == 2
            assert covariance_type == "diag"
            assert n_iter == 300
            assert random_state == 42
            self.means_ = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        def fit(self, x: np.ndarray) -> "FakeGaussianHMM":
            hmm_fit_lengths.append(len(x))
            return self

        def predict(self, x: np.ndarray) -> np.ndarray:
            hmm_predict_lengths.append(len(x))
            states = np.zeros(len(x), dtype=int)
            states[-12:] = 1
            return states

    original_ic_mask = signals_module._positive_spearman_ic_mask

    def capture_ic_window(ic_window: pd.DataFrame, columns: pd.Index) -> pd.Series:
        ic_window_lengths.append(len(ic_window))
        return original_ic_mask(ic_window, columns)

    monkeypatch.setattr(signals_module, "GaussianHMM", FakeGaussianHMM)
    monkeypatch.setattr(signals_module, "_positive_spearman_ic_mask", capture_ic_window)

    score = _dynamic_composite_score(zscores=zscores, target_ret=target_ret, spx_returns=spx_returns)

    assert isinstance(score, pd.Series)
    pd.testing.assert_index_equal(score.index, zscores.index)
    assert hmm_fit_lengths
    assert hmm_predict_lengths == hmm_fit_lengths
    assert min(hmm_fit_lengths) >= 24
    assert any(length == 12 for length in ic_window_lengths)
    assert score.notna().any()


def test_dynamic_composite_score_uses_only_active_columns_for_ic_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.date_range("2002-01-31", periods=80, freq="ME")
    active_signal = pd.Series(np.linspace(-1.0, 1.0, 80), index=idx)
    zscores = pd.DataFrame(
        {
            "spx_trend_z": active_signal,
            "credit_vs_rates_z": np.nan,
            "risk_breadth_z": np.nan,
            "bcom_trend_z": np.nan,
        },
        index=idx,
    )
    target_ret = active_signal.shift(1).fillna(0.0)
    seen_windows: list[tuple[list[str], int]] = []

    original_ic_mask = signals_module._positive_spearman_ic_mask

    def capture_ic_window(ic_window: pd.DataFrame, columns: pd.Index) -> pd.Series:
        seen_windows.append((list(ic_window.columns), len(ic_window)))
        return original_ic_mask(ic_window, columns)

    monkeypatch.setattr(signals_module, "_positive_spearman_ic_mask", capture_ic_window)

    score = _dynamic_composite_score(zscores=zscores, target_ret=target_ret)

    assert seen_windows
    assert all(cols == ["spx_trend_z", "target_ret"] for cols, _ in seen_windows)
    assert max(length for _, length in seen_windows) >= 12
    assert pd.notna(score.iloc[-1])
    assert np.isclose(score.iloc[-1], active_signal.iloc[-1])
