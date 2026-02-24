"""Robustness grid evaluation module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from primary_model.analytics.performance import perf_table
from primary_model.backtest.engine import backtest_from_weights
from primary_model.data.loader import (
    DEFAULT_ASSETS,
    apply_treasury_total_return,
    load_universe,
    universe_returns_matrix,
)
from primary_model.portfolio.weights import weights_from_primary_signal
from primary_model.signals.variant1 import build_primary_signal_variant1
from primary_model.utils.artifacts import write_dataframe, write_markdown_protocol


@dataclass(frozen=True)
class RobustnessRunConfig:
    root: Path
    out_dir: Path
    tcost_grid: list[float]
    buy_grid: list[float]
    sell_grid: list[float]
    duration_grid: list[float]


def _parse_float_list(value: str) -> list[float]:
    out: list[float] = []
    for item in value.split(","):
        token = item.strip()
        if token:
            out.append(float(token))
    if not out:
        raise ValueError("At least one numeric value is required.")
    return out


def _resolve_run_config(
    config: Mapping[str, Any],
    cli_args: Any | None = None,
) -> RobustnessRunConfig:
    paths_cfg = config.get("paths", {})
    run_cfg = config.get("run", {})

    root_override = getattr(cli_args, "root", None) if cli_args else None
    out_dir_override = getattr(cli_args, "out_dir", None) if cli_args else None
    tcost_override = getattr(cli_args, "tcost_grid_bps", None) if cli_args else None
    buy_override = getattr(cli_args, "buy_grid", None) if cli_args else None
    sell_override = getattr(cli_args, "sell_grid", None) if cli_args else None
    duration_override = getattr(cli_args, "duration_grid", None) if cli_args else None

    root = Path(root_override or paths_cfg.get("root", "artifacts")).resolve()
    out_dir = Path(out_dir_override or (root / "reports" / "robustness")).resolve()

    tcost_grid = _parse_float_list(
        str(tcost_override or run_cfg.get("tcost_grid_bps", "0,5,10,25,50"))
    )
    buy_grid = _parse_float_list(
        str(buy_override or run_cfg.get("buy_grid", "0.0001,0.25,0.50"))
    )
    sell_grid = _parse_float_list(
        str(sell_override or run_cfg.get("sell_grid", "-0.0001,-0.25,-0.50"))
    )
    duration_grid = _parse_float_list(
        str(duration_override or run_cfg.get("duration_grid", "6.0,8.5,10.0"))
    )

    return RobustnessRunConfig(
        root=root,
        out_dir=out_dir,
        tcost_grid=tcost_grid,
        buy_grid=buy_grid,
        sell_grid=sell_grid,
        duration_grid=duration_grid,
    )


def run_experiment(
    config: dict[str, Any],
    cli_args: Any = None,
) -> dict[str, Any]:
    """Execute robustness grid evaluation and write artifacts."""
    run_config = _resolve_run_config(config, cli_args=cli_args)
    run_config.out_dir.mkdir(parents=True, exist_ok=True)

    clean_dir = run_config.root / "data" / "clean"
    universe = load_universe(clean_dir, list(DEFAULT_ASSETS))

    rows: list[dict[str, float | int | str]] = []
    scenario_counter = 0

    for duration in run_config.duration_grid:
        adjusted = apply_treasury_total_return(universe, duration=duration)
        returns = universe_returns_matrix(adjusted)
        equal_weight_row = pd.Series(
            1.0 / len(returns.columns), index=returns.columns, dtype=float
        )

        for buy_threshold in run_config.buy_grid:
            for sell_threshold in run_config.sell_grid:
                if buy_threshold <= sell_threshold:
                    continue

                signals = build_primary_signal_variant1(
                    adjusted,
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

                for tcost_bps in run_config.tcost_grid:
                    backtest = backtest_from_weights(
                        returns=returns,
                        weights=weights,
                        tcost_bps=tcost_bps,
                    )
                    summary = perf_table({"PrimaryV1": backtest}).iloc[0]

                    rows.append(
                        {
                            "scenario": scenario_name,
                            "duration": duration,
                            "buy_threshold": buy_threshold,
                            "sell_threshold": sell_threshold,
                            "tcost_bps": tcost_bps,
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
    grid_path = run_config.out_dir / "robustness_grid_results.csv"
    write_dataframe(grid, grid_path, index=False)

    baseline = grid[
        (grid["duration"].round(6) == 8.5)
        & (grid["buy_threshold"].round(6) == 0.0001)
        & (grid["sell_threshold"].round(6) == -0.0001)
    ].sort_values("tcost_bps")
    baseline_path = run_config.out_dir / "baseline_cost_sensitivity.csv"
    write_dataframe(baseline, baseline_path, index=False)

    top15_path = run_config.out_dir / "top15_by_sharpe.csv"
    write_dataframe(grid.head(15), top15_path, index=False)

    md_lines = [
        "# Robustness Summary",
        "",
        f"- Scenarios evaluated: `{len(grid)}`",
        f"- Duration grid: `{run_config.duration_grid}`",
        f"- Buy thresholds: `{run_config.buy_grid}`",
        f"- Sell thresholds: `{run_config.sell_grid}`",
        f"- Transaction-cost grid (bps): `{run_config.tcost_grid}`",
        "",
        "## Best Sharpe Scenario",
    ]
    best = grid.iloc[0]
    md_lines.extend(
        [
            f"- Scenario: `{best['scenario']}`",
            f"- Duration: `{best['duration']}`",
            (
                f"- Thresholds: buy `{best['buy_threshold']}`, "
                f"sell `{best['sell_threshold']}`"
            ),
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
    write_markdown_protocol(md_lines, run_config.out_dir / "robustness_summary.md")

    return {
        "status": "ok",
        "artifacts_written": 4,
        "out_dir": str(run_config.out_dir),
        "scenarios_evaluated": len(grid),
    }
