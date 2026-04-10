from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import metalabel.primary.pipeline as pipeline
import metalabel.validation as validation


def _config() -> dict[str, object]:
    return {
        "primary": {
            "duration": 8.5,
            "trend_window": 6,
            "relative_window": 3,
            "zscore_min_periods": 12,
            "buy_threshold": 0.31,
            "sell_threshold": -0.31,
            "tcost_bps": 0.0,
            "benchmark_key": "EqualWeight25",
        },
        "validation": {
            "min_train_periods": 3,
        },
    }


def _sample_universe() -> dict[str, pd.DataFrame]:
    index = pd.date_range("2020-01-31", periods=6, freq="ME")
    returns = pd.DataFrame(
        {
            "spx": [0.02, 0.01, -0.01, 0.03, 0.01, 0.02],
            "bcom": [0.01, 0.00, 0.01, 0.00, 0.01, 0.00],
            "treasury_10y": [0.003, 0.004, 0.002, 0.003, 0.004, 0.003],
            "corp_bonds": [0.005, 0.004, 0.006, 0.005, 0.004, 0.005],
        },
        index=index,
    )
    return {
        asset: pd.DataFrame(
            {
                "Price": 100.0 * (1.0 + returns[asset]).cumprod(),
                "Return": returns[asset],
            },
            index=index,
        )
        for asset in returns.columns
    }


def _install_pipeline_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    universe = _sample_universe()
    returns = pd.concat(
        [df["Return"].rename(asset) for asset, df in universe.items()],
        axis=1,
    )

    def fake_load_universe(clean_dir: Path, assets: list[str]) -> dict[str, pd.DataFrame]:
        return {asset: universe[asset].copy() for asset in assets}

    def fake_apply_treasury_total_return(
        loaded_universe: dict[str, pd.DataFrame],
        duration: float,
    ) -> dict[str, pd.DataFrame]:
        return {asset: df.copy() for asset, df in loaded_universe.items()}

    def fake_build_primary_signal_variant1(
        adjusted_universe: dict[str, pd.DataFrame],
        trend_window: int,
        relative_window: int,
        zscore_min_periods: int,
        buy_threshold: float,
        sell_threshold: float,
    ) -> pd.DataFrame:
        index = next(iter(adjusted_universe.values())).index
        signal = pd.Series(["BUY", "HOLD", "SELL", "BUY", "HOLD", "BUY"], index=index, name="signal")
        score = pd.Series(np.linspace(-0.5, 0.5, len(index)), index=index, name="composite_score")
        return pd.DataFrame({"signal": signal, "composite_score": score})

    def fake_universe_returns_matrix(universe_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return returns.copy()

    def fake_weights_from_primary_signal(signal: pd.Series, returns_columns: list[str]) -> pd.DataFrame:
        weights = np.full((len(signal), len(returns_columns)), 1.0 / len(returns_columns), dtype=float)
        return pd.DataFrame(weights, index=signal.index, columns=returns_columns)

    def fake_backtest_from_weights(
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        tcost_bps: float,
    ) -> pd.DataFrame:
        index = returns.index[:-1]
        net = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01], index=index)
        gross = net.copy()
        turnover = pd.Series(np.zeros(len(index), dtype=float), index=index)
        equity = (1.0 + net).cumprod()
        return pd.DataFrame(
            {
                "gross_return": gross,
                "net_return": net,
                "turnover": turnover,
                "equity_gross": equity,
                "equity_net": equity,
            },
            index=index,
        )

    def fake_classification_table(
        signal: pd.Series,
        score: pd.Series,
        forward_returns: pd.Series,
    ) -> pd.DataFrame:
        return pd.DataFrame({"precision": [0.5], "recall": [0.5], "auc_score": [0.5]})

    monkeypatch.setattr(pipeline, "load_universe", fake_load_universe)
    monkeypatch.setattr(pipeline, "apply_treasury_total_return", fake_apply_treasury_total_return)
    monkeypatch.setattr(pipeline, "build_primary_signal_variant1", fake_build_primary_signal_variant1)
    monkeypatch.setattr(pipeline, "universe_returns_matrix", fake_universe_returns_matrix)
    monkeypatch.setattr(pipeline, "weights_from_primary_signal", fake_weights_from_primary_signal)
    monkeypatch.setattr(pipeline, "backtest_from_weights", fake_backtest_from_weights)
    monkeypatch.setattr(pipeline, "classification_table", fake_classification_table)


def test_run_primary_variant1_summary_is_labeled_full_sample(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_pipeline_stubs(monkeypatch)

    pipeline.run_primary_variant1(tmp_path, config=_config())

    summary = pd.read_csv(tmp_path / "reports" / "results" / "primary_v1_summary.csv", index_col=0)
    clf = pd.read_csv(tmp_path / "reports" / "results" / "primary_v1_classification.csv")

    assert summary["evaluation_scope"].eq("full_sample_causal").all()
    assert clf["evaluation_scope"].eq("full_sample_causal").all()


def test_run_benchmarks_outputs_are_labeled_full_sample_and_point_to_oos(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_pipeline_stubs(monkeypatch)

    pipeline.run_benchmarks(tmp_path, config=_config())

    summary = pd.read_csv(tmp_path / "reports" / "results" / "benchmarks_summary.csv", index_col=0)
    html = (tmp_path / "reports" / "results" / "benchmarks_summary.html").read_text(encoding="utf-8")

    assert summary["evaluation_scope"].eq("full_sample_causal").all()
    assert "Full-Sample Causal Backtests" in html
    assert "primary_v1_oos_summary.csv" in html


def test_run_walk_forward_exports_official_primary_v1_oos_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    universe = _sample_universe()
    returns = pd.concat(
        [df["Return"].rename(asset) for asset, df in universe.items()],
        axis=1,
    )

    def fake_resolve_within_project(path: Path, label: str) -> Path:
        return Path(path)

    def fake_load_universe(clean_dir: Path, assets: list[str]) -> dict[str, pd.DataFrame]:
        return {asset: universe[asset].copy() for asset in assets}

    def fake_apply_treasury_total_return(
        loaded_universe: dict[str, pd.DataFrame],
        duration: float,
    ) -> dict[str, pd.DataFrame]:
        return {asset: df.copy() for asset, df in loaded_universe.items()}

    def fake_universe_returns_matrix(universe_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return returns.copy()

    def fake_strict_walk_forward(**kwargs: object) -> pd.DataFrame:
        index = returns.index[2:-1]
        net = pd.Series([0.03, 0.01, 0.02], index=index)
        gross = net.copy()
        turnover = pd.Series(np.zeros(len(index), dtype=float), index=index)
        equity = (1.0 + net).cumprod()
        return pd.DataFrame(
            {
                "gross_return": gross,
                "net_return": net,
                "turnover": turnover,
                "equity_gross": equity,
                "equity_net": equity,
            },
            index=index,
        )

    def fake_build_primary_signal_variant1(
        adjusted_universe: dict[str, pd.DataFrame],
        trend_window: int,
        relative_window: int,
        zscore_min_periods: int,
        buy_threshold: float,
        sell_threshold: float,
    ) -> pd.DataFrame:
        index = next(iter(adjusted_universe.values())).index
        signal = pd.Series(["BUY"] * len(index), index=index, name="signal")
        return pd.DataFrame({"signal": signal})

    def fake_weights_from_primary_signal(signal: pd.Series, returns_columns: list[str]) -> pd.DataFrame:
        weights = np.full((len(signal), len(returns_columns)), 1.0 / len(returns_columns), dtype=float)
        return pd.DataFrame(weights, index=signal.index, columns=returns_columns)

    def fake_backtest_from_weights(
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        tcost_bps: float,
    ) -> pd.DataFrame:
        index = returns.index[:-1]
        net = pd.Series([0.005, 0.005, 0.005, 0.005, 0.005], index=index)
        gross = net.copy()
        turnover = pd.Series(np.zeros(len(index), dtype=float), index=index)
        equity = (1.0 + net).cumprod()
        return pd.DataFrame(
            {
                "gross_return": gross,
                "net_return": net,
                "turnover": turnover,
                "equity_gross": equity,
                "equity_net": equity,
            },
            index=index,
        )

    monkeypatch.setattr(validation, "_resolve_within_project", fake_resolve_within_project)
    monkeypatch.setattr(validation, "load_universe", fake_load_universe)
    monkeypatch.setattr(validation, "apply_treasury_total_return", fake_apply_treasury_total_return)
    monkeypatch.setattr(validation, "universe_returns_matrix", fake_universe_returns_matrix)
    monkeypatch.setattr(validation, "_strict_walk_forward", fake_strict_walk_forward)
    monkeypatch.setattr(validation, "build_primary_signal_variant1", fake_build_primary_signal_variant1)
    monkeypatch.setattr(validation, "weights_from_primary_signal", fake_weights_from_primary_signal)
    monkeypatch.setattr(validation, "backtest_from_weights", fake_backtest_from_weights)

    validation.run_walk_forward(root=tmp_path, config=_config())

    official = pd.read_csv(tmp_path / "reports" / "results" / "primary_v1_oos_summary.csv", index_col=0)
    walk_forward = pd.read_csv(
        tmp_path / "reports" / "results" / "walk_forward" / "walk_forward_summary.csv",
        index_col=0,
    )

    assert official.index.tolist() == ["PrimaryV1"]
    assert official["evaluation_scope"].iloc[0] == "oos_walk_forward"
    assert official["source_validation"].iloc[0] == "walk_forward"
    assert official["ann_return"].iloc[0] == walk_forward.loc["WalkForwardStrict", "ann_return"]
