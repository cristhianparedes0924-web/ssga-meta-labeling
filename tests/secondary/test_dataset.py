from __future__ import annotations

import numpy as np
import pandas as pd

from metalabel.data import apply_treasury_total_return, universe_returns_matrix
from metalabel.primary.backtest import backtest_from_weights
from metalabel.primary.portfolio import weights_from_primary_signal
from metalabel.primary.signals import build_primary_signal_variant1
from metalabel.secondary import build_secondary_dataset


def _synthetic_universe(n: int = 36) -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2020-01-31", periods=n, freq="ME", name="Date")
    t = np.arange(n)

    spx_price = 100.0 + np.cumsum(np.sin(t / 2.0) * 4.0 + 1.5)
    bcom_price = 80.0 + np.cumsum(np.cos(t / 3.0) * 3.0)
    corp_price = 95.0 + np.cumsum(np.sin(t / 4.0) * 2.0 + 0.5)
    treasury_yield = 4.0 + np.sin(t / 5.0) * 0.6 + np.where(t > 18, 0.5, 0.0)

    def frame(price: np.ndarray) -> pd.DataFrame:
        series = pd.Series(price, index=idx)
        return pd.DataFrame(
            {
                "Price": series,
                "Return": series.pct_change().fillna(0.0),
            },
            index=idx,
        )

    return {
        "spx": frame(spx_price),
        "bcom": frame(bcom_price),
        "corp_bonds": frame(corp_price),
        "treasury_10y": pd.DataFrame(
            {
                "Price": pd.Series(treasury_yield, index=idx),
                "Return": pd.Series(np.zeros(n), index=idx),
            },
            index=idx,
        ),
    }


def _test_config() -> dict[str, dict[str, float | int]]:
    return {
        "primary": {
            "trend_window": 6,
            "relative_window": 3,
            "zscore_min_periods": 6,
            "duration": 8.5,
            "buy_threshold": 0.1,
            "sell_threshold": -0.1,
            "tcost_bps": 0.0,
        }
    }


def _indicator_weights() -> dict[str, float]:
    return {
        "spx_trend_z": 1.0,
        "bcom_trend_z": 1.0,
        "credit_vs_rates_z": 1.0,
        "risk_breadth_z": 1.0,
        "bcom_accel_z": 1.0,
        "yield_mom_z": 1.0,
    }


def _expected_primary_backtest(
    universe: dict[str, pd.DataFrame],
    config: dict[str, dict[str, float | int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary_cfg = config["primary"]
    adjusted = apply_treasury_total_return(universe, duration=float(primary_cfg["duration"]))
    signals = build_primary_signal_variant1(
        adjusted,
        trend_window=int(primary_cfg["trend_window"]),
        relative_window=int(primary_cfg["relative_window"]),
        zscore_min_periods=int(primary_cfg["zscore_min_periods"]),
        indicator_weights=_indicator_weights(),
        buy_threshold=float(primary_cfg["buy_threshold"]),
        sell_threshold=float(primary_cfg["sell_threshold"]),
    )
    returns = universe_returns_matrix(adjusted)
    weights = weights_from_primary_signal(signals["signal"], returns_columns=list(returns.columns))
    weights = weights.reindex(returns.index).ffill()
    equal_weight_row = pd.Series(1.0 / len(returns.columns), index=returns.columns, dtype=float)
    weights = weights.fillna(equal_weight_row)
    backtest = backtest_from_weights(
        returns=returns,
        weights=weights,
        tcost_bps=float(primary_cfg["tcost_bps"]),
    )
    return signals, backtest


def test_build_secondary_dataset_filters_actionable_events_only() -> None:
    universe = _synthetic_universe()
    config = _test_config()

    actionable = build_secondary_dataset(
        universe=universe,
        config=config,
        trailing_window=3,
        indicator_weights=_indicator_weights(),
    )
    with_hold = build_secondary_dataset(
        universe=universe,
        config=config,
        include_hold=True,
        trailing_window=3,
        indicator_weights=_indicator_weights(),
    )

    assert set(actionable["primary_signal"]) <= {"BUY", "SELL"}
    assert "HOLD" in set(with_hold["primary_signal"])
    assert len(with_hold) > len(actionable)


def test_build_secondary_dataset_uses_next_period_primary_strategy_return_for_label() -> None:
    universe = _synthetic_universe()
    config = _test_config()
    dataset = build_secondary_dataset(
        universe=universe,
        config=config,
        trailing_window=3,
        indicator_weights=_indicator_weights(),
    )
    signals, backtest = _expected_primary_backtest(universe, config)

    expected = pd.DataFrame(
        {
            "date": backtest.index,
            "expected_signal": signals.reindex(backtest.index)["signal"],
            "expected_return": backtest["net_return"],
            "expected_label": (backtest["net_return"] > 0.0).astype(int),
        }
    )
    expected = expected[expected["expected_signal"].isin(["BUY", "SELL"])].reset_index(drop=True)

    merged = dataset.merge(expected, on="date", how="inner")
    assert len(merged) == len(dataset)
    assert np.allclose(
        merged["meta_target_return"].to_numpy(),
        merged["expected_return"].to_numpy(),
    )
    assert merged["meta_label"].tolist() == merged["expected_label"].tolist()


def test_build_secondary_dataset_trailing_features_are_shifted() -> None:
    universe = _synthetic_universe()
    config = _test_config()
    dataset = build_secondary_dataset(
        universe=universe,
        config=config,
        include_hold=True,
        trailing_window=3,
        indicator_weights=_indicator_weights(),
    ).set_index("date")
    _, backtest = _expected_primary_backtest(universe, config)

    prior_net_return = backtest["net_return"].shift(1)
    prior_turnover = backtest["turnover"].shift(1)
    expected_hit_rate = prior_net_return.gt(0.0).astype(float).where(prior_net_return.notna()).rolling(
        3,
        min_periods=1,
    ).mean()
    expected_avg_return = prior_net_return.rolling(3, min_periods=1).mean()
    expected_vol = prior_net_return.rolling(3, min_periods=2).std(ddof=0)
    expected_avg_turnover = prior_turnover.rolling(3, min_periods=1).mean()

    assert np.allclose(
        dataset["trailing_hit_rate_3"].to_numpy(),
        expected_hit_rate.loc[dataset.index].to_numpy(),
        equal_nan=True,
    )
    assert np.allclose(
        dataset["trailing_avg_net_return_3"].to_numpy(),
        expected_avg_return.loc[dataset.index].to_numpy(),
        equal_nan=True,
    )
    assert np.allclose(
        dataset["trailing_vol_net_return_3"].to_numpy(),
        expected_vol.loc[dataset.index].to_numpy(),
        equal_nan=True,
    )
    assert np.allclose(
        dataset["trailing_avg_turnover_3"].to_numpy(),
        expected_avg_turnover.loc[dataset.index].to_numpy(),
        equal_nan=True,
    )
    leaky_avg_return = backtest["net_return"].rolling(3, min_periods=1).mean()
    aligned_expected = expected_avg_return.loc[dataset.index]
    aligned_leaky = leaky_avg_return.loc[dataset.index]
    comparison = pd.DataFrame(
        {
            "shifted": aligned_expected,
            "leaky": aligned_leaky,
        }
    ).dropna()
    assert (comparison["shifted"] - comparison["leaky"]).abs().gt(1e-12).any()


def test_build_secondary_dataset_exposes_expected_columns() -> None:
    dataset = build_secondary_dataset(
        universe=_synthetic_universe(),
        config=_test_config(),
        trailing_window=3,
        indicator_weights=_indicator_weights(),
    )

    expected_columns = {
        "date",
        "realized_date",
        "primary_signal",
        "composite_score",
        "meta_target_gross_return",
        "meta_target_return",
        "meta_label",
        "event_turnover",
        "spx_trend",
        "credit_vs_rates",
        "spx_trend_z",
        "yield_mom_z",
        "weight_spx",
        "weight_treasury_10y",
        "trailing_hit_rate_3",
        "trailing_avg_net_return_3",
        "trailing_vol_net_return_3",
        "trailing_avg_turnover_3",
        "signal_streak",
    }

    assert expected_columns.issubset(dataset.columns)
