from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import metalabel.validation as validation


def _make_monthly_validation_inputs(n_periods: int = 24) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    index = pd.date_range("2020-01-31", periods=n_periods, freq="ME")
    returns = pd.DataFrame(
        {
            "spx": np.full(n_periods, 0.01, dtype=float),
            "bcom": np.full(n_periods, 0.005, dtype=float),
            "treasury_10y": np.full(n_periods, 0.002, dtype=float),
            "corp_bonds": np.full(n_periods, 0.003, dtype=float),
        },
        index=index,
    )
    adjusted_universe = {
        asset: pd.DataFrame(
            {
                "Price": 100.0 * (1.0 + returns[asset]).cumprod(),
                "Return": returns[asset],
            },
            index=index,
        )
        for asset in returns.columns
    }
    return adjusted_universe, returns


def _install_validation_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build_primary_signal_variant1(
        adjusted_universe: dict[str, pd.DataFrame],
        trend_window: int,
        relative_window: int,
        zscore_min_periods: int,
        buy_threshold: float,
        sell_threshold: float,
    ) -> pd.DataFrame:
        index = next(iter(adjusted_universe.values())).index
        labels = np.array(["BUY", "HOLD", "SELL"], dtype=object)
        signal = pd.Series(labels[np.arange(len(index)) % len(labels)], index=index, name="signal")
        return pd.DataFrame({"signal": signal})

    def fake_weights_from_primary_signal(
        signal: pd.Series,
        returns_columns: list[str],
    ) -> pd.DataFrame:
        weights = np.full((len(signal), len(returns_columns)), 1.0 / len(returns_columns), dtype=float)
        return pd.DataFrame(weights, index=signal.index, columns=returns_columns)

    def fake_backtest_from_weights(
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        tcost_bps: float,
    ) -> pd.DataFrame:
        net_return = returns.iloc[1:, 0].astype(float).reset_index(drop=True)
        decision_index = returns.index[:-1]
        gross_return = pd.Series(net_return.to_numpy(), index=decision_index, dtype=float)
        turnover = pd.Series(np.zeros(len(decision_index), dtype=float), index=decision_index)
        equity = (1.0 + gross_return).cumprod()
        return pd.DataFrame(
            {
                "gross_return": gross_return,
                "net_return": gross_return,
                "turnover": turnover,
                "equity_gross": equity,
                "equity_net": equity,
            },
            index=decision_index,
        )

    monkeypatch.setattr(validation, "build_primary_signal_variant1", fake_build_primary_signal_variant1)
    monkeypatch.setattr(validation, "weights_from_primary_signal", fake_weights_from_primary_signal)
    monkeypatch.setattr(validation, "backtest_from_weights", fake_backtest_from_weights)


def test_monthly_cv_one_month_behavior_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_validation_stubs(monkeypatch)
    adjusted_universe, returns = _make_monthly_validation_inputs()

    fold_summary, oos_backtest = validation._monthly_cross_validation(
        adjusted_universe=adjusted_universe,
        returns=returns,
        min_train_periods=6,
        test_window_months=1,
        buy_threshold=0.31,
        sell_threshold=-0.31,
        tcost_bps=0.0,
        trend_window=6,
        relative_window=3,
        zscore_min_periods=12,
    )

    assert len(fold_summary) == 18
    assert fold_summary["test_window_months"].eq(1).all()
    assert fold_summary["test_start_month"].equals(fold_summary["test_end_month"])
    assert fold_summary["fold_label"].iloc[0] == "2020-07"
    assert len(oos_backtest) == 18
    assert oos_backtest.index.is_monotonic_increasing


def test_monthly_cv_three_month_fold_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_validation_stubs(monkeypatch)
    adjusted_universe, returns = _make_monthly_validation_inputs()

    fold_summary, oos_backtest = validation._monthly_cross_validation(
        adjusted_universe=adjusted_universe,
        returns=returns,
        min_train_periods=6,
        test_window_months=3,
        buy_threshold=0.31,
        sell_threshold=-0.31,
        tcost_bps=0.0,
        trend_window=6,
        relative_window=3,
        zscore_min_periods=12,
    )

    assert fold_summary["fold_label"].tolist()[0] == "2020-07 to 2020-09"
    assert fold_summary["test_start_month"].tolist() == [
        "2020-07",
        "2020-10",
        "2021-01",
        "2021-04",
        "2021-07",
        "2021-10",
    ]
    assert fold_summary["observations_in_fold"].tolist() == [3] * 6
    assert len(oos_backtest) == 18


def test_monthly_cv_six_month_fold_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_validation_stubs(monkeypatch)
    adjusted_universe, returns = _make_monthly_validation_inputs()

    fold_summary, oos_backtest = validation._monthly_cross_validation(
        adjusted_universe=adjusted_universe,
        returns=returns,
        min_train_periods=6,
        test_window_months=6,
        buy_threshold=0.31,
        sell_threshold=-0.31,
        tcost_bps=0.0,
        trend_window=6,
        relative_window=3,
        zscore_min_periods=12,
    )

    assert fold_summary["fold_label"].tolist() == [
        "2020-07 to 2020-12",
        "2021-01 to 2021-06",
        "2021-07 to 2021-12",
    ]
    assert fold_summary["observations_in_fold"].tolist() == [6, 6, 6]
    assert len(oos_backtest) == 18


def test_monthly_cv_folds_are_causal_and_non_overlapping(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_validation_stubs(monkeypatch)
    adjusted_universe, returns = _make_monthly_validation_inputs()

    fold_summary, oos_backtest = validation._monthly_cross_validation(
        adjusted_universe=adjusted_universe,
        returns=returns,
        min_train_periods=6,
        test_window_months=3,
        buy_threshold=0.31,
        sell_threshold=-0.31,
        tcost_bps=0.0,
        trend_window=6,
        relative_window=3,
        zscore_min_periods=12,
    )

    start_periods = pd.PeriodIndex(fold_summary["test_start_month"], freq="M")
    end_periods = pd.PeriodIndex(fold_summary["test_end_month"], freq="M")
    train_end_dates = pd.to_datetime(fold_summary["train_end_date"])

    assert ((train_end_dates.dt.to_period("M") < start_periods)).all()
    assert (start_periods[1:] > end_periods[:-1]).all()

    realized_periods = oos_backtest.index.to_period("M")
    assert realized_periods.is_monotonic_increasing
    assert realized_periods.is_unique


def test_monthly_cv_concatenated_results_are_sorted_and_recomputed(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_validation_stubs(monkeypatch)
    adjusted_universe, returns = _make_monthly_validation_inputs()

    _, oos_backtest = validation._monthly_cross_validation(
        adjusted_universe=adjusted_universe,
        returns=returns,
        min_train_periods=6,
        test_window_months=3,
        buy_threshold=0.31,
        sell_threshold=-0.31,
        tcost_bps=0.0,
        trend_window=6,
        relative_window=3,
        zscore_min_periods=12,
    )

    expected_equity = (1.0 + oos_backtest["net_return"]).cumprod()
    pd.testing.assert_series_equal(oos_backtest["equity_net"], expected_equity, check_names=False)
    assert oos_backtest.index.is_monotonic_increasing


def test_monthly_cv_insufficient_data_for_requested_horizon_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_validation_stubs(monkeypatch)
    adjusted_universe, returns = _make_monthly_validation_inputs(n_periods=10)

    with pytest.raises(ValueError, match="test_window_months=3"):
        validation._monthly_cross_validation(
            adjusted_universe=adjusted_universe,
            returns=returns,
            min_train_periods=9,
            test_window_months=3,
            buy_threshold=0.31,
            sell_threshold=-0.31,
            tcost_bps=0.0,
            trend_window=6,
            relative_window=3,
            zscore_min_periods=12,
        )
