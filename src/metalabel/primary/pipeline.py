"""Primary-model orchestration workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from metalabel import load_primary_config
from metalabel.data import DEFAULT_ASSETS, apply_treasury_total_return, load_universe, prepare_data, universe_returns_matrix
from metalabel.primary.backtest import backtest_from_weights
from metalabel.primary.metrics import classification_table, perf_table
from metalabel.primary.portfolio import weights_6040, weights_buy_hold_spx, weights_equal_weight, weights_from_primary_signal, weights_simple_trend
from metalabel.primary.signals import build_primary_signal_variant1
from metalabel.reporting import _asset_sanity_table, _corr_vs_equal_weight, _excess_vs_equal_weight, _strategy_return_table, build_benchmark_html, plot_drawdowns, plot_equity_curves, plot_rolling_sharpe, reports_assets_dir, reports_results_dir


def _resolved_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return load_primary_config() if config is None else dict(config)


def _primary_settings(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = _resolved_config(config)
    return dict(cfg.get("primary", cfg))


def run_primary_variant1(root: Path, config: Mapping[str, Any] | None = None) -> None:
    """Run primary strategy pipeline and save backtest summary outputs."""
    primary_cfg = _primary_settings(config)
    clean_dir = root / "data" / "clean"
    reports_dir = reports_results_dir(root)
    reports_dir.mkdir(parents=True, exist_ok=True)

    universe = load_universe(clean_dir, DEFAULT_ASSETS)
    adj_universe = apply_treasury_total_return(universe, duration=float(primary_cfg["duration"]))
    signals = build_primary_signal_variant1(
        adj_universe,
        trend_window=int(primary_cfg["trend_window"]),
        relative_window=int(primary_cfg["relative_window"]),
        zscore_min_periods=int(primary_cfg["zscore_min_periods"]),
        buy_threshold=float(primary_cfg["buy_threshold"]),
        sell_threshold=float(primary_cfg["sell_threshold"]),
    )
    returns = universe_returns_matrix(adj_universe)

    weights_raw = weights_from_primary_signal(
        signal=signals["signal"],
        returns_columns=list(returns.columns),
    )
    weights = weights_raw.reindex(returns.index).ffill()
    equal_weight_row = pd.Series(1.0 / len(returns.columns), index=returns.columns, dtype=float)
    weights = weights.fillna(equal_weight_row)

    backtest = backtest_from_weights(
        returns=returns,
        weights=weights,
        tcost_bps=float(primary_cfg["tcost_bps"]),
    )
    summary = perf_table({"PrimaryV1": backtest}, periods_per_year=12)
    summary.insert(0, "evaluation_scope", "full_sample_causal")
    spx_fwd = adj_universe["spx"]["Return"].shift(-1)
    clf = classification_table(
        signal=signals["signal"],
        score=signals["composite_score"],
        forward_returns=spx_fwd,
    )
    clf.insert(0, "evaluation_scope", "full_sample_causal")

    avg_turnover = float(backtest["turnover"].mean())

    print("Performance table (net, full-sample causal backtest):")
    print(summary.to_string())
    print()

    print("Classification metrics (BUY vs SPX up, AUC on composite score; full-sample causal):")
    print(clf.to_string(index=False))
    print()

    print(f"Average turnover: {avg_turnover:.6f}")

    backtest_path = reports_dir / "primary_v1_backtest.csv"
    summary_path = reports_dir / "primary_v1_summary.csv"
    clf_path = reports_dir / "primary_v1_classification.csv"
    backtest.to_csv(backtest_path, index=True)
    summary.to_csv(summary_path, index=True)
    clf.to_csv(clf_path, index=False)

    print(f"Saved CSV: {backtest_path}")
    print(f"Saved CSV: {summary_path}")
    print(f"Saved CSV: {clf_path}")


def run_benchmarks(root: Path, config: Mapping[str, Any] | None = None) -> None:
    """Run benchmark comparisons and export summary reports and plots."""
    primary_cfg = _primary_settings(config)
    clean_dir = root / "data" / "clean"
    reports_dir = reports_results_dir(root)
    reports_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = reports_assets_dir(root)
    assets_dir.mkdir(parents=True, exist_ok=True)

    universe = load_universe(clean_dir, DEFAULT_ASSETS)
    adj_universe = apply_treasury_total_return(universe, duration=float(primary_cfg["duration"]))
    returns = universe_returns_matrix(adj_universe)
    primary_signals = build_primary_signal_variant1(
        adj_universe,
        trend_window=int(primary_cfg["trend_window"]),
        relative_window=int(primary_cfg["relative_window"]),
        zscore_min_periods=int(primary_cfg["zscore_min_periods"]),
        buy_threshold=float(primary_cfg["buy_threshold"]),
        sell_threshold=float(primary_cfg["sell_threshold"]),
    )

    weights_by_name = {
        "EqualWeight25": weights_equal_weight(returns),
        "BuyHoldSPX": weights_buy_hold_spx(returns),
        "60/40": weights_6040(returns),
        "SimpleTrend": weights_simple_trend(
            prices=adj_universe["spx"]["Price"],
            returns_cols=list(returns.columns),
        ),
    }

    primary_weights = weights_from_primary_signal(
        signal=primary_signals["signal"],
        returns_columns=list(returns.columns),
    )
    primary_weights = primary_weights.reindex(returns.index).ffill()
    equal_weight_row = pd.Series(1.0 / len(returns.columns), index=returns.columns, dtype=float)
    primary_weights = primary_weights.fillna(equal_weight_row)
    weights_by_name["PrimaryV1"] = primary_weights

    backtests = {
        name: backtest_from_weights(
            returns=returns,
            weights=w,
            tcost_bps=float(primary_cfg["tcost_bps"]),
        )
        for name, w in weights_by_name.items()
    }

    summary = perf_table(
        backtests,
        periods_per_year=12,
        benchmark_key=str(primary_cfg["benchmark_key"]),
    )
    summary.insert(0, "evaluation_scope", "full_sample_causal")
    strategy_returns = _strategy_return_table(backtests)
    excess = _excess_vs_equal_weight(strategy_returns, ew_col="EqualWeight25")
    corr = _corr_vs_equal_weight(strategy_returns, ew_col="EqualWeight25")
    asset_stats = _asset_sanity_table(returns)
    asset_corr = returns.corr()

    print("Benchmark performance table (net, full-sample causal backtests):")
    print(summary.to_string())
    print()

    spx_fwd = adj_universe["spx"]["Return"].shift(-1)
    primary_clf = classification_table(
        signal=primary_signals["signal"],
        score=primary_signals["composite_score"],
        forward_returns=spx_fwd,
    )
    primary_clf.insert(0, "evaluation_scope", "full_sample_causal")
    primary_avg_turnover = float(backtests["PrimaryV1"]["turnover"].mean())
    print("PrimaryV1 classification metrics (full-sample causal):")
    print(primary_clf.to_string(index=False))
    print()
    print(f"PrimaryV1 average turnover: {primary_avg_turnover:.6f}")
    print()

    print("Excess vs EqualWeight25:")
    print(excess.to_string())
    print()

    print("Correlation with EqualWeight25 net returns:")
    print(corr.to_string())
    print()

    print("Asset-level sanity stats (adjusted returns):")
    print(asset_stats.to_string())
    print()

    print("Asset returns correlation matrix (adjusted returns):")
    print(asset_corr.to_string())
    print()

    equity_plot_path = plot_equity_curves(backtests, assets_dir / "equity_curves.png")
    drawdowns_plot_path = plot_drawdowns(backtests, assets_dir / "drawdowns.png")
    rolling_sharpe_plot_path = plot_rolling_sharpe(
        backtests, assets_dir / "rolling_sharpe.png", window=12
    )

    csv_path = reports_dir / "benchmarks_summary.csv"
    html_path = reports_dir / "benchmarks_summary.html"
    primary_signal_path = reports_dir / "primary_v1_signal.csv"
    primary_weights_path = reports_dir / "primary_v1_weights.csv"
    primary_clf_path = reports_dir / "primary_v1_classification.csv"

    summary.to_csv(csv_path, index=True)
    primary_signals.to_csv(primary_signal_path, index=True)
    primary_weights.to_csv(primary_weights_path, index=True)
    primary_clf.to_csv(primary_clf_path, index=False)
    html_path.write_text(
        build_benchmark_html(
            summary=summary,
            excess=excess,
            corr=corr,
            asset_stats=asset_stats,
            asset_corr=asset_corr,
            equity_plot_path=equity_plot_path,
            drawdowns_plot_path=drawdowns_plot_path,
            rolling_sharpe_plot_path=rolling_sharpe_plot_path,
        ),
        encoding="utf-8",
    )

    print(f"Saved CSV: {csv_path}")
    print(f"Saved CSV: {primary_signal_path}")
    print(f"Saved CSV: {primary_weights_path}")
    print(f"Saved CSV: {primary_clf_path}")
    print(f"Saved HTML: {html_path}")
    print(f"Saved plot: {equity_plot_path}")
    print(f"Saved plot: {drawdowns_plot_path}")
    print(f"Saved plot: {rolling_sharpe_plot_path}")


def run_all(root: Path, config: Mapping[str, Any] | None = None) -> None:
    from metalabel.validation import run_data_qc

    prepare_data(root)
    run_data_qc(root, config=config)
    run_primary_variant1(root, config=config)
    run_benchmarks(root, config=config)
