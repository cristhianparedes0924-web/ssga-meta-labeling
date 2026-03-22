from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import metalabel.validation as validation


def _make_monthly_validation_inputs(n_periods: int = 48) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    index = pd.date_range("2018-01-31", periods=n_periods, freq="ME")
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
        decision_index = returns.index[:-1]
        net_return = pd.Series(returns.iloc[1:, 0].to_numpy(dtype=float), index=decision_index)
        turnover = pd.Series(np.zeros(len(decision_index), dtype=float), index=decision_index)
        equity = (1.0 + net_return).cumprod()
        return pd.DataFrame(
            {
                "gross_return": net_return,
                "net_return": net_return,
                "turnover": turnover,
                "equity_gross": equity,
                "equity_net": equity,
            },
            index=decision_index,
        )

    monkeypatch.setattr(validation, "build_primary_signal_variant1", fake_build_primary_signal_variant1)
    monkeypatch.setattr(validation, "weights_from_primary_signal", fake_weights_from_primary_signal)
    monkeypatch.setattr(validation, "backtest_from_weights", fake_backtest_from_weights)


def test_monthly_cv_expanding_uses_all_prior_history(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_validation_stubs(monkeypatch)
    adjusted_universe, returns = _make_monthly_validation_inputs()

    fold_summary, oos_backtest = validation._monthly_cross_validation(
        adjusted_universe=adjusted_universe,
        returns=returns,
        min_train_periods=12,
        window_type="expanding",
        rolling_train_months=None,
        buy_threshold=0.31,
        sell_threshold=-0.31,
        tcost_bps=0.0,
        trend_window=6,
        relative_window=3,
        zscore_min_periods=12,
    )

    assert len(fold_summary) == 36
    assert fold_summary["window_type"].eq("expanding").all()
    assert fold_summary["test_month"].iloc[0] == "2019-01"
    assert fold_summary["train_start_date"].iloc[0] == "2018-01-31"
    assert fold_summary["train_start_date"].iloc[-1] == "2018-01-31"
    assert len(oos_backtest) == 36
    assert oos_backtest.index.is_monotonic_increasing


def test_monthly_cv_rolling_three_month_window_moves_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_validation_stubs(monkeypatch)
    adjusted_universe, returns = _make_monthly_validation_inputs()

    fold_summary, oos_backtest = validation._monthly_cross_validation(
        adjusted_universe=adjusted_universe,
        returns=returns,
        min_train_periods=12,
        window_type="rolling",
        rolling_train_months=3,
        buy_threshold=0.31,
        sell_threshold=-0.31,
        tcost_bps=0.0,
        trend_window=6,
        relative_window=3,
        zscore_min_periods=12,
    )

    assert fold_summary["window_type"].eq("rolling").all()
    assert fold_summary["rolling_train_months"].eq(3).all()
    assert fold_summary["test_month"].iloc[0] == "2019-01"
    assert fold_summary["train_start_date"].iloc[0] == "2018-10-31"
    assert fold_summary["train_end_date"].iloc[0] == "2018-12-31"
    assert fold_summary["train_start_date"].iloc[1] == "2018-11-30"
    assert len(oos_backtest) == 36


def test_monthly_cv_rolling_six_month_window_is_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_validation_stubs(monkeypatch)
    adjusted_universe, returns = _make_monthly_validation_inputs()

    fold_summary, oos_backtest = validation._monthly_cross_validation(
        adjusted_universe=adjusted_universe,
        returns=returns,
        min_train_periods=12,
        window_type="rolling",
        rolling_train_months=6,
        buy_threshold=0.31,
        sell_threshold=-0.31,
        tcost_bps=0.0,
        trend_window=6,
        relative_window=3,
        zscore_min_periods=12,
    )

    assert fold_summary["rolling_train_months"].eq(6).all()
    assert fold_summary["train_start_date"].iloc[0] == "2018-07-31"
    assert fold_summary["train_end_date"].iloc[0] == "2018-12-31"
    assert len(oos_backtest) == 36


def test_monthly_cv_folds_remain_causal_and_monthly(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_validation_stubs(monkeypatch)
    adjusted_universe, returns = _make_monthly_validation_inputs()

    fold_summary, oos_backtest = validation._monthly_cross_validation(
        adjusted_universe=adjusted_universe,
        returns=returns,
        min_train_periods=12,
        window_type="rolling",
        rolling_train_months=12,
        buy_threshold=0.31,
        sell_threshold=-0.31,
        tcost_bps=0.0,
        trend_window=6,
        relative_window=3,
        zscore_min_periods=12,
    )

    train_end_months = pd.to_datetime(fold_summary["train_end_date"]).dt.to_period("M")
    test_months = pd.PeriodIndex(fold_summary["test_month"], freq="M")

    assert (train_end_months < test_months).all()
    assert test_months.is_monotonic_increasing
    assert test_months.is_unique
    assert oos_backtest.index.to_period("M").is_unique


def test_monthly_cv_concatenated_results_are_sorted_and_recomputed(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_validation_stubs(monkeypatch)
    adjusted_universe, returns = _make_monthly_validation_inputs()

    _, oos_backtest = validation._monthly_cross_validation(
        adjusted_universe=adjusted_universe,
        returns=returns,
        min_train_periods=12,
        window_type="rolling",
        rolling_train_months=24,
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


def test_monthly_cv_invalid_rolling_window_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_validation_stubs(monkeypatch)
    adjusted_universe, returns = _make_monthly_validation_inputs()

    with pytest.raises(ValueError, match="rolling_train_months must be one of"):
        validation._monthly_cross_validation(
            adjusted_universe=adjusted_universe,
            returns=returns,
            min_train_periods=12,
            window_type="rolling",
            rolling_train_months=9,
            buy_threshold=0.31,
            sell_threshold=-0.31,
            tcost_bps=0.0,
            trend_window=6,
            relative_window=3,
            zscore_min_periods=12,
        )


def test_monthly_cv_insufficient_data_for_requested_window_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_validation_stubs(monkeypatch)
    adjusted_universe, returns = _make_monthly_validation_inputs(n_periods=10)

    with pytest.raises(ValueError, match="window_type='rolling'"):
        validation._monthly_cross_validation(
            adjusted_universe=adjusted_universe,
            returns=returns,
            min_train_periods=9,
            window_type="rolling",
            rolling_train_months=36,
            buy_threshold=0.31,
            sell_threshold=-0.31,
            tcost_bps=0.0,
            trend_window=6,
            relative_window=3,
            zscore_min_periods=12,
        )
