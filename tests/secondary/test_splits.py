from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metalabel.secondary import build_secondary_dataset, expanding_forward_splits, holdout_split_by_time


def _simple_secondary_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-31",
                    "2020-01-31",
                    "2020-02-29",
                    "2020-03-31",
                    "2020-04-30",
                    "2020-04-30",
                    "2020-05-31",
                    "2020-06-30",
                ]
            ),
            "primary_signal": ["BUY", "SELL", "BUY", "SELL", "BUY", "SELL", "BUY", "SELL"],
            "meta_label": [1, 0, 1, 0, 1, 1, 0, 1],
            "meta_target_return": [0.01, -0.01, 0.02, -0.02, 0.03, 0.01, -0.03, 0.02],
        }
    )


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


def test_holdout_split_by_time_is_chronological_and_non_overlapping() -> None:
    split = holdout_split_by_time(
        _simple_secondary_rows(),
        validation_periods=2,
        min_train_periods=2,
    )

    assert split.train["date"].is_monotonic_increasing
    assert split.validation["date"].is_monotonic_increasing
    assert split.train_end < split.validation_start
    assert split.train["date"].max() < split.validation["date"].min()
    assert set(split.train["date"]).isdisjoint(set(split.validation["date"]))


def test_holdout_split_by_time_keeps_same_date_rows_together() -> None:
    split = holdout_split_by_time(
        _simple_secondary_rows(),
        validation_periods=3,
        min_train_periods=2,
    )

    validation_date_counts = split.validation["date"].value_counts()
    assert pd.Timestamp("2020-04-30") in validation_date_counts.index
    assert validation_date_counts.loc[pd.Timestamp("2020-04-30")] == 2
    assert pd.Timestamp("2020-04-30") not in set(split.train["date"])


def test_expanding_forward_splits_expand_training_window() -> None:
    splits = expanding_forward_splits(
        _simple_secondary_rows(),
        min_train_periods=2,
        validation_periods=1,
        step_periods=1,
    )

    train_unique_counts = [split.train["date"].nunique() for split in splits]
    validation_starts = [split.validation_start for split in splits]

    assert len(splits) == 4
    assert train_unique_counts == [2, 3, 4, 5]
    assert validation_starts == list(pd.to_datetime(["2020-03-31", "2020-04-30", "2020-05-31", "2020-06-30"]))
    assert all(split.train_end < split.validation_start for split in splits)


def test_expanding_forward_splits_work_with_secondary_dataset_builder_output() -> None:
    dataset = build_secondary_dataset(
        universe=_synthetic_universe(),
        config=_test_config(),
        trailing_window=3,
        indicator_weights=_indicator_weights(),
    )

    splits = expanding_forward_splits(
        dataset,
        min_train_periods=5,
        validation_periods=2,
        step_periods=2,
        max_splits=2,
    )

    assert len(splits) == 2
    assert all(split.train_end < split.validation_start for split in splits)
    assert all(split.train["date"].is_monotonic_increasing for split in splits)
    assert all(split.validation["date"].is_monotonic_increasing for split in splits)


def test_temporal_split_utilities_fail_clearly_on_invalid_inputs() -> None:
    dataset = _simple_secondary_rows()

    with pytest.raises(ValueError, match="validation_periods must be >= 1"):
        holdout_split_by_time(dataset, validation_periods=0)

    with pytest.raises(ValueError, match="Not enough unique decision dates"):
        holdout_split_by_time(dataset.iloc[:2], validation_periods=1, min_train_periods=2)

    with pytest.raises(ValueError, match="max_splits must be >= 1"):
        expanding_forward_splits(dataset, min_train_periods=2, max_splits=0)

    with pytest.raises(ValueError, match="missing required date column"):
        expanding_forward_splits(dataset.rename(columns={"date": "decision_date"}), min_train_periods=2)
