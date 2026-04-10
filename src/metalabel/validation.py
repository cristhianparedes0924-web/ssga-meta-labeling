"""Validation, QC, robustness, and reproducibility workflows."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from metalabel import PROJECT_ROOT, load_primary_config
from metalabel.data import DEFAULT_ASSETS, _resolve_local_raw_dir, _resolve_within_project, apply_treasury_total_return, load_universe, universe_returns_matrix
from metalabel.primary.backtest import backtest_from_weights
from metalabel.primary.metrics import perf_table
from metalabel.primary.portfolio import _normalize_long_only_weights, weights_from_primary_signal
from metalabel.primary.signals import build_primary_signal_variant1, score_to_signal
from metalabel.reporting import annualized_stats, build_asset_summary, build_data_qc_html, maybe_yield_warning, reports_results_dir


def _resolved_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return load_primary_config() if config is None else dict(config)


def _primary_settings(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = _resolved_config(config)
    return dict(cfg.get("primary", cfg))


def _validation_settings(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = _resolved_config(config)
    return dict(cfg.get("validation", {}))


def run_data_qc(root: Path, config: Mapping[str, Any] | None = None) -> None:
    """Run data quality checks and write a compact HTML QC report."""
    primary_cfg = _primary_settings(config)
    clean_dir = root / "data" / "clean"
    reports_dir = reports_results_dir(root)
    reports_dir.mkdir(parents=True, exist_ok=True)
    html_path = reports_dir / "data_qc.html"

    universe = load_universe(clean_dir, DEFAULT_ASSETS)
    asset_summary = build_asset_summary(universe)
    raw_returns_matrix = universe_returns_matrix(universe)
    adj_universe = apply_treasury_total_return(universe, duration=float(primary_cfg["duration"]))
    adj_returns_matrix = universe_returns_matrix(adj_universe)

    overlap_summary = pd.DataFrame(
        [
            {
                "raw_overlap_rows": int(len(raw_returns_matrix)),
                "adjusted_overlap_rows": int(len(adj_returns_matrix)),
                "overlap_min_date": (
                    raw_returns_matrix.index.min().date().isoformat()
                    if len(raw_returns_matrix)
                    else None
                ),
                "overlap_max_date": (
                    raw_returns_matrix.index.max().date().isoformat()
                    if len(raw_returns_matrix)
                    else None
                ),
            }
        ]
    )

    corr = raw_returns_matrix.corr()
    raw_ann_stats = annualized_stats(raw_returns_matrix)
    adj_ann_stats = annualized_stats(adj_returns_matrix)

    warning_line = maybe_yield_warning(universe)

    print("Asset summary:")
    print(asset_summary.to_string(index=False))
    print()

    if warning_line:
        print(warning_line)
        print()

    print("Overlap summary:")
    print(overlap_summary.to_string(index=False))
    print()

    print("Returns correlation matrix:")
    print(corr.to_string())
    print()

    print("Raw annualized mean/vol (monthly assumption):")
    print(raw_ann_stats.to_string())
    print()

    print("Adjusted annualized mean/vol (monthly assumption, treasury duration=8.5):")
    print(adj_ann_stats.to_string())
    print()

    html_path.write_text(
        build_data_qc_html(
            asset_summary=asset_summary,
            overlap_summary=overlap_summary,
            corr=corr,
            raw_ann_stats=raw_ann_stats,
            adj_ann_stats=adj_ann_stats,
            warning_line=warning_line,
        ),
        encoding="utf-8",
    )
    print(f"HTML report written to: {html_path}")


def setup_test_root(
    target_root: Path = Path("test"),
    source_raw: Path = Path("data/raw"),
    clean: bool = False,
) -> None:
    """Bootstrap an isolated test root with bundled raw input files."""
    target_root = _resolve_within_project(target_root, "target_root")
    source_raw = _resolve_local_raw_dir(source_raw)

    target_raw = target_root / "data" / "raw"
    target_clean = target_root / "data" / "clean"
    target_reports = target_root / "reports"

    if clean:
        if target_clean.exists():
            shutil.rmtree(target_clean)
        if target_reports.exists():
            shutil.rmtree(target_reports)

    target_raw.mkdir(parents=True, exist_ok=True)
    target_clean.mkdir(parents=True, exist_ok=True)
    (target_root / "reports" / "assets").mkdir(parents=True, exist_ok=True)
    (target_root / "reports" / "results").mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for path in sorted(source_raw.iterdir()):
        if path.is_file():
            dst = target_raw / path.name
            shutil.copy2(path, dst)
            copied.append(dst)

    print(f"Target root: {target_root}")
    print(f"Copied {len(copied)} raw files to: {target_raw}")
    for path in copied:
        print(f"- {path}")


def _parse_float_list(value: str) -> list[float]:
    out: list[float] = []
    for item in value.split(","):
        token = item.strip()
        if token:
            out.append(float(token))
    if not out:
        raise ValueError("At least one numeric value is required.")
    return out


def run_robustness(
    root: Path = PROJECT_ROOT,
    out_dir: Path | None = None,
    tcost_grid_bps: str | None = None,
    buy_grid: str | None = None,
    sell_grid: str | None = None,
    duration_grid: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> None:
    """Run robustness grid for costs, thresholds, and treasury duration."""
    primary_cfg = _primary_settings(config)
    validation_cfg = _validation_settings(config)
    robustness_cfg = validation_cfg.get("robustness", {})

    root = _resolve_within_project(root, "root")
    out_dir = (
        _resolve_within_project(out_dir, "out_dir")
        if out_dir is not None
        else (reports_results_dir(root) / "robustness")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    tcost_grid = _parse_float_list(
        str(tcost_grid_bps or robustness_cfg.get("tcost_grid_bps", "0,5,10,25,50"))
    )
    buy_grid_values = _parse_float_list(
        str(buy_grid or robustness_cfg.get("buy_grid", primary_cfg["buy_threshold"]))
    )
    sell_grid_values = _parse_float_list(
        str(sell_grid or robustness_cfg.get("sell_grid", primary_cfg["sell_threshold"]))
    )
    duration_grid_values = _parse_float_list(
        str(duration_grid or robustness_cfg.get("duration_grid", primary_cfg["duration"]))
    )

    clean_dir = root / "data" / "clean"
    universe = load_universe(clean_dir, list(DEFAULT_ASSETS))

    rows: list[dict[str, float | int | str]] = []
    scenario_counter = 0

    for duration in duration_grid_values:
        adjusted = apply_treasury_total_return(universe, duration=duration)
        returns = universe_returns_matrix(adjusted)
        equal_weight_row = pd.Series(
            1.0 / len(returns.columns), index=returns.columns, dtype=float
        )

        for buy_threshold in buy_grid_values:
            for sell_threshold in sell_grid_values:
                if buy_threshold <= sell_threshold:
                    continue

                signals = build_primary_signal_variant1(
                    adjusted,
                    trend_window=int(primary_cfg["trend_window"]),
                    relative_window=int(primary_cfg["relative_window"]),
                    zscore_min_periods=int(primary_cfg["zscore_min_periods"]),
                    buy_threshold=buy_threshold,
                    sell_threshold=sell_threshold,
                )
                signal_series = signals["signal"]
                signal_counts = (
                    signal_series.value_counts(dropna=False)
                    .reindex(["BUY", "HOLD", "SELL"], fill_value=0)
                    .astype(int)
                )

                weights = weights_from_primary_signal(
                    signal=signal_series,
                    returns_columns=list(returns.columns),
                )
                weights = weights.reindex(returns.index).ffill().fillna(equal_weight_row)

                scenario_counter += 1
                scenario_name = f"S{scenario_counter:03d}"

                for tcost_bps_value in tcost_grid:
                    backtest = backtest_from_weights(
                        returns=returns,
                        weights=weights,
                        tcost_bps=tcost_bps_value,
                    )
                    summary = perf_table({"PrimaryV1": backtest}).iloc[0]

                    rows.append(
                        {
                            "scenario": scenario_name,
                            "duration": duration,
                            "buy_threshold": buy_threshold,
                            "sell_threshold": sell_threshold,
                            "tcost_bps": tcost_bps_value,
                            "ann_return": float(summary["ann_return"]),
                            "ann_vol": float(summary["ann_vol"]),
                            "sharpe": float(summary["sharpe"]),
                            "max_drawdown": float(summary["max_drawdown"]),
                            "calmar": float(summary["calmar"]),
                            "avg_turnover": float(summary["avg_turnover"]),
                            "ending_equity_net": float(backtest["equity_net"].iloc[-1]),
                            "buy_count": int(signal_counts["BUY"]),
                            "hold_count": int(signal_counts["HOLD"]),
                            "sell_count": int(signal_counts["SELL"]),
                            "periods": int(len(backtest)),
                        }
                    )

    grid = pd.DataFrame(rows).sort_values(
        by=["sharpe", "ann_return"], ascending=[False, False]
    )
    grid_path = out_dir / "robustness_grid_results.csv"
    grid.to_csv(grid_path, index=False)

    baseline = grid[
        (grid["duration"].round(6) == round(float(primary_cfg["duration"]), 6))
        & (grid["buy_threshold"].round(6) == round(float(primary_cfg["buy_threshold"]), 6))
        & (grid["sell_threshold"].round(6) == round(float(primary_cfg["sell_threshold"]), 6))
    ].sort_values("tcost_bps")
    baseline_path = out_dir / "baseline_cost_sensitivity.csv"
    baseline.to_csv(baseline_path, index=False)

    top15_path = out_dir / "top15_by_sharpe.csv"
    grid.head(15).to_csv(top15_path, index=False)

    md_lines = [
        "# Robustness Summary",
        "",
        f"- Scenarios evaluated: `{len(grid)}`",
        f"- Duration grid: `{duration_grid_values}`",
        f"- Buy thresholds: `{buy_grid_values}`",
        f"- Sell thresholds: `{sell_grid_values}`",
        f"- Transaction-cost grid (bps): `{tcost_grid}`",
        "",
        "## Best Sharpe Scenario",
    ]
    best = grid.iloc[0]
    md_lines.extend(
        [
            f"- Scenario: `{best['scenario']}`",
            f"- Duration: `{best['duration']}`",
            f"- Thresholds: buy `{best['buy_threshold']}`, sell `{best['sell_threshold']}`",
            f"- Cost (bps): `{best['tcost_bps']}`",
            f"- Sharpe: `{best['sharpe']:.4f}`",
            f"- Ann. return: `{best['ann_return']:.4%}`",
            f"- Max drawdown: `{best['max_drawdown']:.4%}`",
            "",
            "## Outputs",
            f"- `{grid_path}`",
            f"- `{baseline_path}`",
            f"- `{top15_path}`",
        ]
    )
    (out_dir / "robustness_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved: {grid_path}")
    print(f"Saved: {baseline_path}")
    print(f"Saved: {top15_path}")
    print(f"Saved: {out_dir / 'robustness_summary.md'}")


def _strict_walk_forward(
    adjusted_universe: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    min_train_periods: int,
    buy_threshold: float,
    sell_threshold: float,
    tcost_bps: float,
    trend_window: int,
    relative_window: int,
    zscore_min_periods: int,
) -> pd.DataFrame:
    if len(returns) <= min_train_periods + 1:
        raise ValueError("Not enough data for requested min_train_periods.")

    columns = list(returns.columns)
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    prev_weight: pd.Series | None = None

    for i in range(min_train_periods - 1, len(returns.index) - 1):
        decision_date = returns.index[i]
        realized_date = returns.index[i + 1]

        history_universe = {
            asset: df.loc[:decision_date].copy() for asset, df in adjusted_universe.items()
        }
        history_signals = build_primary_signal_variant1(
            history_universe,
            trend_window=trend_window,
            relative_window=relative_window,
            zscore_min_periods=zscore_min_periods,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )
        signal_series = history_signals["signal"]
        signal_t = str(signal_series.iloc[-1]) if pd.notna(signal_series.iloc[-1]) else "NaN"

        weights_hist = weights_from_primary_signal(signal_series, returns_columns=columns)
        weight_t = pd.to_numeric(weights_hist.iloc[-1], errors="coerce").fillna(0.0)
        weight_t = weight_t.clip(lower=0.0)
        denom = float(weight_t.sum())
        if denom > 0.0:
            weight_t = weight_t / denom

        next_rets = returns.loc[realized_date, columns]
        gross_return = float((weight_t * next_rets).sum())

        if prev_weight is None:
            turnover = 0.0
        else:
            turnover = 0.5 * float((weight_t - prev_weight).abs().sum())
        cost = turnover * (tcost_bps / 10000.0)
        net_return = gross_return - cost

        row: dict[str, float | str | pd.Timestamp] = {
            "decision_date": decision_date,
            "realized_date": realized_date,
            "signal": signal_t,
            "gross_return": gross_return,
            "net_return": net_return,
            "turnover": turnover,
        }
        for col in columns:
            row[f"w_{col}"] = float(weight_t[col])

        rows.append(row)
        prev_weight = weight_t

    out = pd.DataFrame(rows).set_index("decision_date").sort_index()
    out["equity_gross"] = (1.0 + out["gross_return"]).cumprod()
    out["equity_net"] = (1.0 + out["net_return"]).cumprod()
    return out


def run_walk_forward(
    root: Path = PROJECT_ROOT,
    out_dir: Path | None = None,
    min_train_periods: int | None = None,
    duration: float | None = None,
    buy_threshold: float | None = None,
    sell_threshold: float | None = None,
    tcost_bps: float | None = None,
    config: Mapping[str, Any] | None = None,
) -> None:
    """Run strict walk-forward validation for PrimaryV1."""
    primary_cfg = _primary_settings(config)
    validation_cfg = _validation_settings(config)

    min_train_periods = int(min_train_periods or validation_cfg.get("min_train_periods", 120))
    duration = float(duration if duration is not None else primary_cfg["duration"])
    buy_threshold = float(buy_threshold if buy_threshold is not None else primary_cfg["buy_threshold"])
    sell_threshold = float(sell_threshold if sell_threshold is not None else primary_cfg["sell_threshold"])
    tcost_bps = float(tcost_bps if tcost_bps is not None else primary_cfg["tcost_bps"])

    root = _resolve_within_project(root, "root")
    out_dir = (
        _resolve_within_project(out_dir, "out_dir")
        if out_dir is not None
        else (reports_results_dir(root) / "walk_forward")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_dir = root / "data" / "clean"
    universe = load_universe(clean_dir, list(DEFAULT_ASSETS))
    adjusted_universe = apply_treasury_total_return(universe, duration=duration)
    returns = universe_returns_matrix(adjusted_universe)

    wf_backtest = _strict_walk_forward(
        adjusted_universe=adjusted_universe,
        returns=returns,
        min_train_periods=min_train_periods,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        tcost_bps=tcost_bps,
        trend_window=int(primary_cfg["trend_window"]),
        relative_window=int(primary_cfg["relative_window"]),
        zscore_min_periods=int(primary_cfg["zscore_min_periods"]),
    )

    full_signals = build_primary_signal_variant1(
        adjusted_universe,
        trend_window=int(primary_cfg["trend_window"]),
        relative_window=int(primary_cfg["relative_window"]),
        zscore_min_periods=int(primary_cfg["zscore_min_periods"]),
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )
    full_weights = weights_from_primary_signal(
        signal=full_signals["signal"],
        returns_columns=list(returns.columns),
    )
    equal_weight_row = pd.Series(1.0 / len(returns.columns), index=returns.columns, dtype=float)
    full_weights = full_weights.reindex(returns.index).ffill().fillna(equal_weight_row)
    full_backtest = backtest_from_weights(returns=returns, weights=full_weights, tcost_bps=tcost_bps)
    full_backtest = full_backtest.loc[wf_backtest.index.min() :].copy()

    summary = perf_table(
        {
            "WalkForwardStrict": wf_backtest,
            "StandardCausal": full_backtest,
        }
    )
    summary.insert(
        0,
        "evaluation_scope",
        [
            "oos_walk_forward",
            "full_sample_causal_slice",
        ],
    )
    summary["delta_vs_standard"] = summary["ann_return"] - summary.loc["StandardCausal", "ann_return"]

    official_oos_summary = summary.loc[["WalkForwardStrict"]].copy()
    official_oos_summary.index = pd.Index(["PrimaryV1"], name=summary.index.name)
    official_oos_summary["source_validation"] = "walk_forward"

    wf_path = out_dir / "walk_forward_backtest.csv"
    std_path = out_dir / "standard_causal_backtest_slice.csv"
    summary_path = out_dir / "walk_forward_summary.csv"
    official_oos_path = reports_results_dir(root) / "primary_v1_oos_summary.csv"
    wf_backtest.to_csv(wf_path, index=True)
    full_backtest.to_csv(std_path, index=True)
    summary.to_csv(summary_path, index=True)
    official_oos_summary.to_csv(official_oos_path, index=True)

    protocol = [
        "# Walk-Forward Protocol",
        "",
        "- Train window: expanding from first observation to decision date `t`.",
        "- Decision at `t`: compute signal using only data through `t`.",
        "- Realization at `t+1`: apply weight decided at `t` to next-period return.",
        f"- Minimum train periods before first OOS decision: `{min_train_periods}`.",
        f"- Treasury duration assumption: `{duration}`.",
        f"- Thresholds: buy `{buy_threshold}`, sell `{sell_threshold}`.",
        f"- Transaction cost: `{tcost_bps}` bps.",
        f"- OOS decisions evaluated: `{len(wf_backtest)}`.",
        f"- First OOS decision date: `{wf_backtest.index.min().date().isoformat()}`.",
        f"- Last OOS decision date: `{wf_backtest.index.max().date().isoformat()}`.",
        "",
        "## Outputs",
        f"- `{wf_path}`",
        f"- `{std_path}`",
        f"- `{summary_path}`",
        f"- `{official_oos_path}`",
    ]
    (out_dir / "walk_forward_protocol.md").write_text("\n".join(protocol), encoding="utf-8")

    print(f"Saved: {wf_path}")
    print(f"Saved: {std_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {official_oos_path}")
    print(f"Saved: {out_dir / 'walk_forward_protocol.md'}")


def _monthly_cross_validation(
    adjusted_universe: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    min_train_periods: int,
    window_type: str,
    rolling_train_months: int | None,
    buy_threshold: float,
    sell_threshold: float,
    tcost_bps: float,
    trend_window: int,
    relative_window: int,
    zscore_min_periods: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run expanding month-based cross-validation using causal fold boundaries."""
    allowed_window_types = {"expanding", "rolling"}
    allowed_rolling_months = {3, 6, 12, 24, 36}

    if min_train_periods < 1:
        raise ValueError("min_train_periods must be >= 1.")
    if window_type not in allowed_window_types:
        raise ValueError(
            f"window_type must be one of {sorted(allowed_window_types)}, got {window_type!r}."
        )
    if window_type == "rolling":
        if rolling_train_months is None:
            raise ValueError("rolling_train_months is required when window_type='rolling'.")
        if rolling_train_months not in allowed_rolling_months:
            raise ValueError(
                "rolling_train_months must be one of "
                f"{sorted(allowed_rolling_months)} when window_type='rolling'."
            )
    elif rolling_train_months is not None and rolling_train_months not in allowed_rolling_months:
        raise ValueError(
            f"rolling_train_months must be one of {sorted(allowed_rolling_months)} when provided."
        )
    if returns.empty:
        raise ValueError("returns must contain at least one observation.")
    if returns.shape[1] == 0:
        raise ValueError("returns must contain at least one asset column.")
    if len(returns) <= min_train_periods:
        raise ValueError("Not enough data for requested min_train_periods.")

    returns = returns.sort_index()
    periods = returns.index.to_period("M")
    columns = list(returns.columns)
    equal_weight_row = pd.Series(1.0 / len(columns), index=columns, dtype=float)

    fold_summary_rows: list[dict[str, float | int | str]] = []
    oos_folds: list[pd.DataFrame] = []
    unique_periods = periods.unique().sort_values()

    for test_period in unique_periods:
        train_periods_all = unique_periods[unique_periods < test_period]
        if len(train_periods_all) < min_train_periods:
            continue

        if window_type == "expanding":
            train_periods = train_periods_all
        else:
            if len(train_periods_all) < int(rolling_train_months):
                continue
            train_periods = train_periods_all[-int(rolling_train_months) :]

        fold_dates = returns.index[periods == test_period]
        if len(fold_dates) == 0:
            continue

        fold_start_date = fold_dates.min()
        fold_end_date = fold_dates.max()
        train_dates = returns.index[periods.isin(train_periods)]
        if len(train_dates) == 0:
            continue

        train_start_date = train_dates.min()
        train_end_date = train_dates.max()
        subset_returns = returns.loc[:fold_end_date].copy()
        if len(subset_returns) < 2:
            continue

        subset_universe = {
            asset: df.loc[:fold_end_date].copy() for asset, df in adjusted_universe.items()
        }
        signals = build_primary_signal_variant1(
            subset_universe,
            trend_window=trend_window,
            relative_window=relative_window,
            zscore_min_periods=zscore_min_periods,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )
        weights_raw = weights_from_primary_signal(
            signal=signals["signal"],
            returns_columns=columns,
        )
        weights = weights_raw.reindex(subset_returns.index).ffill().fillna(equal_weight_row)

        subset_backtest = backtest_from_weights(
            returns=subset_returns,
            weights=weights,
            tcost_bps=tcost_bps,
        )
        if subset_backtest.empty:
            continue

        realized_dates = pd.Series(
            subset_returns.index[1:],
            index=subset_backtest.index,
            name="realized_date",
        )
        realized_periods = realized_dates.dt.to_period("M")
        fold_mask = realized_periods == test_period
        if not bool(fold_mask.any()):
            continue

        fold_backtest = subset_backtest.loc[fold_mask].copy()
        decision_dates = fold_backtest.index
        fold_realized_dates = pd.DatetimeIndex(realized_dates.loc[decision_dates], name="realized_date")
        fold_signals = signals["signal"].reindex(decision_dates)
        fold_weights = _normalize_long_only_weights(weights.loc[decision_dates, columns])
        fold_label = test_period.strftime("%Y-%m")
        rolling_months_value = int(rolling_train_months) if rolling_train_months is not None else np.nan

        fold_output = fold_backtest.copy()
        fold_output.insert(0, "decision_date", decision_dates)
        fold_output.insert(1, "fold_label", fold_label)
        fold_output.insert(2, "window_type", window_type)
        fold_output.insert(3, "rolling_train_months", rolling_months_value)
        fold_output.insert(4, "test_month", test_period.strftime("%Y-%m"))
        fold_output.insert(5, "train_start_date", train_start_date.date().isoformat())
        fold_output.insert(6, "train_end_date", train_end_date.date().isoformat())
        fold_output.insert(7, "signal", fold_signals.values)
        for col in columns:
            fold_output[f"w_{col}"] = pd.to_numeric(fold_weights[col], errors="coerce").to_numpy()
        fold_output.index = fold_realized_dates
        fold_output = fold_output.sort_index()

        fold_eval = fold_output[["gross_return", "net_return", "turnover"]].copy()
        fold_eval["equity_gross"] = (1.0 + fold_eval["gross_return"]).cumprod()
        fold_eval["equity_net"] = (1.0 + fold_eval["net_return"]).cumprod()
        fold_perf = perf_table({fold_label: fold_eval}).loc[fold_label]

        signal_counts = (
            fold_signals.value_counts(dropna=True)
            .reindex(["BUY", "HOLD", "SELL"], fill_value=0)
            .astype(int)
        )
        fold_summary_rows.append(
            {
                "fold_label": fold_label,
                "window_type": window_type,
                "rolling_train_months": rolling_months_value,
                "test_month": test_period.strftime("%Y-%m"),
                "train_start_date": train_start_date.date().isoformat(),
                "train_end_date": train_end_date.date().isoformat(),
                "observations_in_fold": int(len(fold_output)),
                "ann_return": float(fold_perf["ann_return"]),
                "ann_vol": float(fold_perf["ann_vol"]),
                "sharpe": float(fold_perf["sharpe"]),
                "max_drawdown": float(fold_perf["max_drawdown"]),
                "calmar": float(fold_perf["calmar"]),
                "avg_turnover": float(fold_perf["avg_turnover"]),
                "buy_count": int(signal_counts["BUY"]),
                "hold_count": int(signal_counts["HOLD"]),
                "sell_count": int(signal_counts["SELL"]),
                "ending_equity_net": float(fold_eval["equity_net"].iloc[-1]),
            }
        )
        oos_folds.append(fold_output)

    if not oos_folds:
        raise ValueError(
            "No monthly CV folds available for the requested "
            f"min_train_periods={min_train_periods}, window_type={window_type!r}, "
            f"rolling_train_months={rolling_train_months!r}."
        )

    oos_backtest = pd.concat(oos_folds, axis=0).sort_index()
    oos_backtest["equity_gross"] = (1.0 + oos_backtest["gross_return"]).cumprod()
    oos_backtest["equity_net"] = (1.0 + oos_backtest["net_return"]).cumprod()

    fold_summary = pd.DataFrame(fold_summary_rows).sort_values("test_month").reset_index(drop=True)
    return fold_summary, oos_backtest


def run_monthly_cv(
    root: Path = PROJECT_ROOT,
    out_dir: Path | None = None,
    min_train_periods: int | None = None,
    window_type: str | None = None,
    rolling_train_months: int | None = None,
    duration: float | None = None,
    buy_threshold: float | None = None,
    sell_threshold: float | None = None,
    tcost_bps: float | None = None,
    config: Mapping[str, Any] | None = None,
) -> None:
    """Run expanding month-based cross-validation for PrimaryV1."""
    primary_cfg = _primary_settings(config)
    validation_cfg = _validation_settings(config)
    monthly_cv_cfg = dict(validation_cfg.get("monthly_cv", {}))

    min_train_periods = int(min_train_periods or validation_cfg.get("min_train_periods", 120))
    window_type = str(window_type or monthly_cv_cfg.get("window_type", "expanding")).strip().lower()
    rolling_train_months = (
        int(rolling_train_months)
        if rolling_train_months is not None
        else (
            int(monthly_cv_cfg["rolling_train_months"])
            if monthly_cv_cfg.get("rolling_train_months") is not None
            else None
        )
    )
    duration = float(duration if duration is not None else primary_cfg["duration"])
    buy_threshold = float(buy_threshold if buy_threshold is not None else primary_cfg["buy_threshold"])
    sell_threshold = float(sell_threshold if sell_threshold is not None else primary_cfg["sell_threshold"])
    tcost_bps = float(tcost_bps if tcost_bps is not None else primary_cfg["tcost_bps"])

    root = _resolve_within_project(root, "root")
    out_dir = (
        _resolve_within_project(out_dir, "out_dir")
        if out_dir is not None
        else (reports_results_dir(root) / "monthly_cv")
    )
    if window_type == "rolling":
        if rolling_train_months is None:
            raise ValueError("rolling_train_months is required when window_type='rolling'.")
        out_dir = out_dir / f"rolling_{rolling_train_months:02d}m"
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_dir = root / "data" / "clean"
    universe = load_universe(clean_dir, list(DEFAULT_ASSETS))
    adjusted_universe = apply_treasury_total_return(universe, duration=duration)
    returns = universe_returns_matrix(adjusted_universe)

    fold_summary, oos_backtest = _monthly_cross_validation(
        adjusted_universe=adjusted_universe,
        returns=returns,
        min_train_periods=min_train_periods,
        window_type=window_type,
        rolling_train_months=rolling_train_months,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        tcost_bps=tcost_bps,
        trend_window=int(primary_cfg["trend_window"]),
        relative_window=int(primary_cfg["relative_window"]),
        zscore_min_periods=int(primary_cfg["zscore_min_periods"]),
    )
    summary = perf_table({"MonthlyCV": oos_backtest})
    summary.insert(0, "evaluation_scope", "oos_monthly_cv")

    fold_summary_path = out_dir / "monthly_cv_fold_summary.csv"
    oos_path = out_dir / "monthly_cv_oos_backtest.csv"
    summary_path = out_dir / "monthly_cv_summary.csv"
    fold_summary.to_csv(fold_summary_path, index=False)
    oos_backtest.to_csv(oos_path, index=True)
    summary.to_csv(summary_path, index=True)

    protocol = [
        "# Monthly Cross-Validation Protocol",
        "",
        f"- Window type: `{window_type}`.",
        (
            f"- Rolling train window months: `{rolling_train_months}`."
            if window_type == "rolling"
            else "- Rolling train window months: `N/A`."
        ),
        "- Fold design: one out-of-sample calendar month per fold with strictly causal training history.",
        "- Expanding mode uses all available history from the start of the sample through the month before the test month.",
        "- Rolling mode uses only the trailing `rolling_train_months` calendar months before the test month.",
        "- Within each fold, only returns realized in the test month are retained in the OOS backtest.",
        f"- Minimum train periods before an eligible month: `{min_train_periods}`.",
        f"- Treasury duration assumption: `{duration}`.",
        f"- Thresholds: buy `{buy_threshold}`, sell `{sell_threshold}`.",
        f"- Transaction cost: `{tcost_bps}` bps.",
        f"- Folds evaluated: `{len(fold_summary)}`.",
        f"- First test month: `{fold_summary['test_month'].iloc[0]}`.",
        f"- Last test month: `{fold_summary['test_month'].iloc[-1]}`.",
        f"- OOS observations concatenated: `{len(oos_backtest)}`.",
        "",
        "## Outputs",
        f"- `{fold_summary_path}`",
        f"- `{oos_path}`",
        f"- `{summary_path}`",
    ]
    protocol_path = out_dir / "monthly_cv_protocol.md"
    protocol_path.write_text("\n".join(protocol), encoding="utf-8")

    print(f"Saved: {fold_summary_path}")
    print(f"Saved: {oos_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {protocol_path}")


def _assert_allclose(name: str, actual: np.ndarray, expected: np.ndarray, atol: float = 1e-10) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape mismatch {actual.shape} vs {expected.shape}")
    if not np.allclose(actual, expected, atol=atol, rtol=0.0, equal_nan=True):
        raise AssertionError(f"{name}: values mismatch. actual={actual}, expected={expected}")


def run_self_tests() -> None:
    """Run built-in native tests contained in this package."""
    tests_run = 0

    score = pd.Series([-0.50, -0.31, 0.0, 0.31, 0.50], index=pd.date_range("2000-01-31", periods=5, freq="ME"))
    signal = score_to_signal(score, buy_threshold=0.31, sell_threshold=-0.31)
    expected_signal = pd.Series(["SELL", "HOLD", "HOLD", "HOLD", "BUY"], index=score.index, dtype=object)
    if signal.tolist() != expected_signal.tolist():
        raise AssertionError(f"score_to_signal mismatch: {signal.tolist()} vs {expected_signal.tolist()}")
    tests_run += 1

    idx = pd.date_range("2001-01-31", periods=5, freq="ME")
    sig = pd.Series(["BUY", "HOLD", "SELL", "HOLD", np.nan], index=idx)
    cols = ["spx", "bcom", "treasury_10y", "corp_bonds"]
    w = weights_from_primary_signal(sig, returns_columns=cols)

    buy_expected = np.array([0.40, 0.15, 0.00, 0.45], dtype=float)
    sell_expected = np.array([0.05, 0.00, 0.60, 0.35], dtype=float)
    _assert_allclose("BUY row", w.loc[idx[0], cols].to_numpy(dtype=float), buy_expected)
    _assert_allclose("HOLD carry after BUY", w.loc[idx[1], cols].to_numpy(dtype=float), buy_expected)
    _assert_allclose("SELL row", w.loc[idx[2], cols].to_numpy(dtype=float), sell_expected)
    _assert_allclose("HOLD carry after SELL", w.loc[idx[3], cols].to_numpy(dtype=float), sell_expected)
    _assert_allclose("NaN signal carries previous", w.loc[idx[4], cols].to_numpy(dtype=float), sell_expected)
    tests_run += 1

    ret_idx = pd.date_range("2002-01-31", periods=3, freq="ME")
    returns = pd.DataFrame({"spx": [0.10, 0.20, 0.30]}, index=ret_idx)
    weights = pd.DataFrame({"spx": [1.0, 1.0, 1.0]}, index=ret_idx)
    bt = backtest_from_weights(returns=returns, weights=weights, tcost_bps=0.0)

    if len(bt) != 2:
        raise AssertionError(f"backtest length mismatch: expected 2, got {len(bt)}")
    _assert_allclose("net_return shift", bt["net_return"].to_numpy(dtype=float), np.array([0.20, 0.30], dtype=float))
    _assert_allclose("equity_net", bt["equity_net"].to_numpy(dtype=float), np.array([1.2, 1.56], dtype=float))
    _assert_allclose("turnover", bt["turnover"].to_numpy(dtype=float), np.array([0.0, 0.0], dtype=float))
    tests_run += 1

    n = 36
    uidx = pd.date_range("1999-01-31", periods=n, freq="ME")
    spx_price = pd.Series(np.linspace(100.0, 140.0, n), index=uidx)
    bcom_price = pd.Series(np.linspace(80.0, 90.0, n), index=uidx)
    ust_yield = pd.Series(np.linspace(3.5, 4.0, n), index=uidx)
    corp_price = pd.Series(np.linspace(95.0, 105.0, n), index=uidx)

    spx_ret = spx_price.pct_change().fillna(0.0)
    bcom_ret = bcom_price.pct_change().fillna(0.0)
    ust_ret = pd.Series(np.linspace(0.001, 0.002, n), index=uidx)
    corp_ret = corp_price.pct_change().fillna(0.0)

    universe = {
        "spx": pd.DataFrame({"Price": spx_price, "Return": spx_ret}, index=uidx),
        "bcom": pd.DataFrame({"Price": bcom_price, "Return": bcom_ret}, index=uidx),
        "treasury_10y": pd.DataFrame({"Price": ust_yield, "Return": ust_ret}, index=uidx),
        "corp_bonds": pd.DataFrame({"Price": corp_price, "Return": corp_ret}, index=uidx),
    }
    sig_df = build_primary_signal_variant1(universe)
    required_cols = {
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
    missing = required_cols.difference(sig_df.columns)
    if missing:
        raise AssertionError(f"build_primary_signal_variant1 missing columns: {sorted(missing)}")
    tests_run += 1

    print(f"Native self-tests passed: {tests_run}/{tests_run}")


def _git_commit_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
            )
            .strip()
        )
    except Exception:
        return "unknown"


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    src_path = str(PROJECT_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"
    return env


def _run_and_log(command: list[str], log_path: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        env=_subprocess_env(),
    )

    log_lines = [
        f"$ {' '.join(shlex.quote(tok) for tok in command)}",
        "",
        "STDOUT:",
        completed.stdout,
        "",
        "STDERR:",
        completed.stderr,
        "",
        f"exit_code={completed.returncode}",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    print(f"Ran: {' '.join(command)}")
    print(f"Log: {log_path}")
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {command}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_validation_suite(
    root: Path = Path("test"),
    clean_root: bool = False,
    skip_pytest: bool = False,
    skip_robustness: bool = False,
    skip_walk_forward: bool = False,
) -> None:
    """Run full isolated validation suite and write reproducibility artifacts."""
    target_root = _resolve_within_project(root, "root")
    reports_root = target_root / "reports"
    results_dir = reports_results_dir(target_root)
    repro_dir = results_dir / "reproducibility"
    repro_dir.mkdir(parents=True, exist_ok=True)

    commands: list[tuple[str, list[str]]] = []

    setup_cmd = [
        sys.executable,
        "-m",
        "metalabel.cli",
        "setup-test-root",
        "--target-root",
        str(root),
    ]
    if clean_root:
        setup_cmd.append("--clean")
    commands.append(("setup_test_root", setup_cmd))

    if not skip_pytest:
        commands.append(("self_tests", [sys.executable, "-m", "metalabel.cli", "run-self-tests"]))
        commands.append(("pytest", [sys.executable, "-m", "pytest"]))

    commands.append(
        (
            "run_all",
            [sys.executable, "-m", "metalabel.cli", "run-all", "--root", str(root)],
        )
    )

    if not skip_robustness:
        commands.append(
            (
                "run_robustness",
                [
                    sys.executable,
                    "-m",
                    "metalabel.cli",
                    "run-robustness",
                    "--root",
                    str(root),
                    "--out-dir",
                    str(reports_results_dir(target_root) / "robustness"),
                ],
            )
        )

    if not skip_walk_forward:
        commands.append(
            (
                "run_walk_forward",
                [
                    sys.executable,
                    "-m",
                    "metalabel.cli",
                    "run-walk-forward",
                    "--root",
                    str(root),
                    "--out-dir",
                    str(reports_results_dir(target_root) / "walk_forward"),
                ],
            )
        )

    for i, (label, command) in enumerate(commands, start=1):
        log_path = repro_dir / f"{i:02d}_{label}.log"
        _run_and_log(command, log_path)

    artifacts: dict[str, str] = {}
    if reports_root.exists():
        for path in sorted(reports_root.rglob("*")):
            if path.is_file():
                artifacts[str(path.relative_to(PROJECT_ROOT))] = _sha256(path)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "isolated_root": str(target_root),
        "git_commit": _git_commit_sha(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "commands": [command for _, command in commands],
        "artifact_sha256": artifacts,
    }
    manifest_path = repro_dir / "validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary_lines = [
        "# Validation Suite Summary",
        "",
        f"- Isolated root: `{target_root}`",
        f"- Git commit: `{manifest['git_commit']}`",
        f"- Python: `{manifest['python_version']}`",
        f"- Platform: `{manifest['platform']}`",
        f"- Commands executed: `{len(commands)}`",
        f"- Hashed artifacts: `{len(artifacts)}`",
        "",
        "## Key Outputs",
        f"- `{target_root / 'reports' / 'results' / 'benchmarks_summary.csv'}`",
        f"- `{target_root / 'reports' / 'results' / 'primary_v1_oos_summary.csv'}`",
        f"- `{target_root / 'reports' / 'results' / 'robustness' / 'robustness_grid_results.csv'}`",
        f"- `{target_root / 'reports' / 'results' / 'walk_forward' / 'walk_forward_summary.csv'}`",
        f"- `{manifest_path}`",
    ]
    summary_path = repro_dir / "validation_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Saved: {manifest_path}")
    print(f"Saved: {summary_path}")
    print("Validation suite completed successfully.")


def _run_subprocess(command: list[str]) -> None:
    print(f"$ {' '.join(shlex.quote(tok) for tok in command)}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=_subprocess_env())
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def run_modes(mode: str) -> None:
    """Simple runner for project mode, test mode, or both."""
    project_cmd = [sys.executable, "-m", "metalabel.cli", "run-all"]
    test_cmd = [
        sys.executable,
        "-m",
        "metalabel.cli",
        "run-validation-suite",
        "--root",
        "test",
        "--clean-root",
    ]

    if mode == "project":
        _run_subprocess(project_cmd)
        return
    if mode == "test":
        _run_subprocess(test_cmd)
        return

    _run_subprocess(project_cmd)
    _run_subprocess(test_cmd)
