"""Tests for the M2 secondary meta-labeling model and causal split utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metalabel.secondary.split import causal_train_test_split, walk_forward_splits
from metalabel.secondary.model import M2_FEATURES, prepare_features, run_walk_forward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 100) -> pd.DataFrame:
    """Minimal synthetic secondary dataset for testing."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2000-01-31", periods=n, freq="ME")
    df = pd.DataFrame(
        {
            "date": dates,
            "primary_signal": rng.choice(["BUY", "SELL"], size=n),
            "composite_score": rng.standard_normal(n),
            "spx_trend_z": rng.standard_normal(n),
            "bcom_trend_z": rng.standard_normal(n),
            "credit_vs_rates_z": rng.standard_normal(n),
            "risk_breadth_z": rng.standard_normal(n),
            "bcom_accel_z": rng.standard_normal(n),
            "yield_mom_z": rng.standard_normal(n),
            "trailing_hit_rate_12": rng.uniform(0.4, 0.9, n),
            "trailing_avg_net_return_12": rng.standard_normal(n) * 0.01,
            "trailing_vol_net_return_12": rng.uniform(0.01, 0.05, n),
            "signal_streak": rng.integers(1, 6, n).astype(float),
            "vix_level_z": rng.standard_normal(n),
            "vix_change_z": rng.standard_normal(n),
            "vix_trend": rng.standard_normal(n) * 0.1,
            "vix_high_regime": rng.integers(0, 2, n).astype(float),
            "vix_rising": rng.integers(0, 2, n).astype(float),
            "oas_level_z": rng.standard_normal(n),
            "oas_change_z": rng.standard_normal(n),
            "oas_trend": rng.standard_normal(n) * 0.1,
            "oas_wide_regime": rng.integers(0, 2, n).astype(float),
            "oas_widening": rng.integers(0, 2, n).astype(float),
            "meta_label": rng.integers(0, 2, n),
            "meta_target_return": rng.standard_normal(n) * 0.02,
        }
    )
    return df


# ---------------------------------------------------------------------------
# causal_train_test_split
# ---------------------------------------------------------------------------

class TestCausalTrainTestSplit:
    def test_sizes(self):
        df = _make_df(100)
        train, test = causal_train_test_split(df, min_train_size=60)
        assert len(train) == 60
        assert len(test) == 40

    def test_no_overlap(self):
        df = _make_df(100)
        train, test = causal_train_test_split(df, min_train_size=60)
        assert set(train.index).isdisjoint(set(test.index))

    def test_train_before_test(self):
        df = _make_df(100)
        train, test = causal_train_test_split(df, min_train_size=60)
        assert train["date"].max() < test["date"].min()

    def test_invalid_min_train(self):
        df = _make_df(10)
        with pytest.raises(ValueError):
            causal_train_test_split(df, min_train_size=10)

    def test_min_train_zero_raises(self):
        df = _make_df(10)
        with pytest.raises(ValueError):
            causal_train_test_split(df, min_train_size=0)


# ---------------------------------------------------------------------------
# walk_forward_splits
# ---------------------------------------------------------------------------

class TestWalkForwardSplits:
    def test_first_split_train_size(self):
        df = _make_df(80)
        first_train, first_test = next(walk_forward_splits(df, min_train_size=60, step=1))
        assert len(first_train) == 60
        assert len(first_test) == 1

    def test_expanding_window(self):
        df = _make_df(63)
        sizes = [len(tr) for tr, _ in walk_forward_splits(df, min_train_size=60, step=1)]
        assert sizes == [60, 61, 62]

    def test_no_test_row_in_train(self):
        df = _make_df(70)
        for train, test in walk_forward_splits(df, min_train_size=60, step=1):
            assert set(train.index).isdisjoint(set(test.index))

    def test_step_larger_than_one(self):
        df = _make_df(70)
        splits = list(walk_forward_splits(df, min_train_size=60, step=5))
        assert len(splits) == 2  # rows 60-64, 65-69

    def test_invalid_step(self):
        df = _make_df(70)
        with pytest.raises(ValueError):
            list(walk_forward_splits(df, step=0))


# ---------------------------------------------------------------------------
# prepare_features
# ---------------------------------------------------------------------------

class TestPrepareFeatures:
    def test_output_columns_match_m2_features(self):
        df = _make_df(10)
        X = prepare_features(df)
        assert list(X.columns) == M2_FEATURES

    def test_no_nans_after_prepare(self):
        df = _make_df(20)
        df.loc[5, "bcom_accel_z"] = float("nan")
        X = prepare_features(df)
        assert not X.isnull().any().any()

    def test_buy_encoded_as_one(self):
        df = _make_df(5)
        df["primary_signal"] = "BUY"
        X = prepare_features(df)
        assert (X["primary_signal_buy"] == 1.0).all()

    def test_sell_encoded_as_zero(self):
        df = _make_df(5)
        df["primary_signal"] = "SELL"
        X = prepare_features(df)
        assert (X["primary_signal_buy"] == 0.0).all()

    def test_missing_column_raises(self):
        df = _make_df(5).drop(columns=["vix_level_z"])
        with pytest.raises(KeyError):
            prepare_features(df)


# ---------------------------------------------------------------------------
# run_walk_forward
# ---------------------------------------------------------------------------

class TestRunWalkForward:
    def test_output_length(self):
        df = _make_df(80)
        preds = run_walk_forward(df, min_train_size=60)
        assert len(preds) == 20

    def test_output_columns(self):
        df = _make_df(70)
        preds = run_walk_forward(df, min_train_size=60)
        expected = {"date", "primary_signal", "meta_label", "meta_target_return",
                    "m2_prob", "m2_approve"}
        assert expected.issubset(set(preds.columns))

    def test_probs_in_unit_interval(self):
        df = _make_df(70)
        preds = run_walk_forward(df, min_train_size=60)
        assert (preds["m2_prob"] >= 0).all() and (preds["m2_prob"] <= 1).all()

    def test_approve_binary(self):
        df = _make_df(70)
        preds = run_walk_forward(df, min_train_size=60)
        assert set(preds["m2_approve"].unique()).issubset({0, 1})

    def test_predictions_time_ordered(self):
        df = _make_df(80)
        preds = run_walk_forward(df, min_train_size=60)
        assert (preds["date"].diff().dropna() > pd.Timedelta(0)).all()

    def test_threshold_effect(self):
        df = _make_df(80)
        preds_high = run_walk_forward(df, min_train_size=60, threshold=0.9)
        preds_low  = run_walk_forward(df, min_train_size=60, threshold=0.1)
        assert preds_high["m2_approve"].sum() <= preds_low["m2_approve"].sum()
