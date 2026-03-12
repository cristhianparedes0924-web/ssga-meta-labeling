"""Tests for supplemental VIX and Liquidity (OAS) feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metalabel.secondary.features import (
    _expanding_regime,
    build_liquidity_features,
    build_supplemental_features,
    build_vix_features,
)


def _level_series(n: int = 60, base: float = 20.0, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-31", periods=n, freq="ME", name="Date")
    values = base + np.cumsum(rng.normal(0, 1.0, n))
    return pd.Series(values, index=idx)


# ---------------------------------------------------------------------------
# _expanding_regime
# ---------------------------------------------------------------------------

def test_expanding_regime_values_are_zero_or_one() -> None:
    s = _level_series()
    regime = _expanding_regime(s, min_periods=12)
    valid = regime.dropna()
    assert set(valid.unique()).issubset({0.0, 1.0})


def test_expanding_regime_nan_before_min_periods() -> None:
    s = _level_series(n=30)
    regime = _expanding_regime(s, min_periods=12)
    # The first value is always NaN (no prior history); regime uses shift(1) then expanding
    # So we expect NaN for the first min_periods rows
    assert regime.iloc[:12].isna().all()


def test_expanding_regime_no_look_ahead() -> None:
    """Regime at t must not depend on values after t."""
    s = _level_series(n=40)
    regime_full = _expanding_regime(s, min_periods=12)

    # Truncate series and recompute — regime values on the shared prefix must match
    regime_short = _expanding_regime(s.iloc[:30], min_periods=12)
    shared = regime_full.iloc[:30].dropna()
    short_shared = regime_short.dropna()
    pd.testing.assert_series_equal(shared, short_shared)


# ---------------------------------------------------------------------------
# build_vix_features
# ---------------------------------------------------------------------------

def test_build_vix_features_returns_expected_columns() -> None:
    vix = _level_series(n=60, base=20.0)
    result = build_vix_features(vix, trend_window=6, zscore_min_periods=12)
    expected = {"vix_level_z", "vix_change", "vix_change_z", "vix_trend", "vix_high_regime", "vix_rising"}
    assert expected == set(result.columns)


def test_build_vix_features_index_matches_input() -> None:
    vix = _level_series(n=60, base=20.0)
    result = build_vix_features(vix)
    assert result.index.equals(vix.index)


def test_build_vix_features_no_look_ahead_in_zscore() -> None:
    """Z-score on truncated input must match z-score on full input for shared rows."""
    vix = _level_series(n=60)
    full = build_vix_features(vix, zscore_min_periods=12)
    short = build_vix_features(vix.iloc[:40], zscore_min_periods=12)

    shared_idx = short.index
    np.testing.assert_allclose(
        full.loc[shared_idx, "vix_level_z"].to_numpy(),
        short["vix_level_z"].to_numpy(),
        equal_nan=True,
    )


def test_build_vix_features_rising_is_binary() -> None:
    vix = _level_series(n=60)
    result = build_vix_features(vix)
    valid = result["vix_rising"].dropna()
    assert set(valid.unique()).issubset({0.0, 1.0})


def test_build_vix_features_change_matches_diff() -> None:
    vix = _level_series(n=60)
    result = build_vix_features(vix)
    expected_change = vix.diff()
    pd.testing.assert_series_equal(
        result["vix_change"].rename(None),
        expected_change.rename(None),
        check_names=False,
    )


# ---------------------------------------------------------------------------
# build_liquidity_features
# ---------------------------------------------------------------------------

def test_build_liquidity_features_returns_expected_columns() -> None:
    oas = _level_series(n=60, base=0.8, seed=7)
    result = build_liquidity_features(oas, trend_window=6, zscore_min_periods=12)
    expected = {"oas_level_z", "oas_change", "oas_change_z", "oas_trend", "oas_wide_regime", "oas_widening"}
    assert expected == set(result.columns)


def test_build_liquidity_features_index_matches_input() -> None:
    oas = _level_series(n=60, base=0.8, seed=7)
    result = build_liquidity_features(oas)
    assert result.index.equals(oas.index)


def test_build_liquidity_features_widening_is_binary() -> None:
    oas = _level_series(n=60, base=0.8, seed=7)
    result = build_liquidity_features(oas)
    valid = result["oas_widening"].dropna()
    assert set(valid.unique()).issubset({0.0, 1.0})


def test_build_liquidity_features_no_look_ahead() -> None:
    oas = _level_series(n=60, base=0.8, seed=7)
    full = build_liquidity_features(oas, zscore_min_periods=12)
    short = build_liquidity_features(oas.iloc[:40], zscore_min_periods=12)

    shared_idx = short.index
    np.testing.assert_allclose(
        full.loc[shared_idx, "oas_level_z"].to_numpy(),
        short["oas_level_z"].to_numpy(),
        equal_nan=True,
    )


# ---------------------------------------------------------------------------
# build_supplemental_features — integration (requires clean CSVs)
# ---------------------------------------------------------------------------

def test_build_supplemental_features_raises_if_csvs_missing(tmp_path: pytest.fixture) -> None:
    with pytest.raises(FileNotFoundError, match="Supplemental clean CSVs not found"):
        build_supplemental_features(root=tmp_path)


def test_build_supplemental_features_returns_12_columns(tmp_path: pytest.fixture) -> None:
    """Smoke test: write minimal CSVs and verify 12 columns are returned."""
    clean_dir = tmp_path / "data" / "clean"
    clean_dir.mkdir(parents=True)

    idx = pd.date_range("2010-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(0)

    vix_df = pd.DataFrame({"Date": idx, "Price": 20.0 + rng.normal(0, 3, 60)})
    vix_df.to_csv(clean_dir / "vix.csv", index=False, date_format="%Y-%m-%d")

    oas_df = pd.DataFrame({"Date": idx, "Price": 0.8 + rng.normal(0, 0.1, 60)})
    oas_df.to_csv(clean_dir / "liquidity.csv", index=False, date_format="%Y-%m-%d")

    result = build_supplemental_features(root=tmp_path)
    assert result.shape[1] == 12
    assert set(result.columns) == {
        "vix_level_z", "vix_change", "vix_change_z", "vix_trend", "vix_high_regime", "vix_rising",
        "oas_level_z", "oas_change", "oas_change_z", "oas_trend", "oas_wide_regime", "oas_widening",
    }
