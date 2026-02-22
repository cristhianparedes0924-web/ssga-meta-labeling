#!/usr/bin/env python3
"""Run robustness grid for costs, thresholds, and treasury duration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_float_list(value: str) -> list[float]:
    out: list[float] = []
    for item in value.split(","):
        token = item.strip()
        if token:
            out.append(float(token))
    if not out:
        raise ValueError("At least one numeric value is required.")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PrimaryV1 robustness grid runner.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("test"),
        help="Project root containing data/clean for evaluation.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <root>/reports/robustness).",
    )
    parser.add_argument(
        "--tcost-grid-bps",
        type=str,
        default="0,5,10,25,50",
        help="Comma-separated transaction cost grid in bps.",
    )
    parser.add_argument(
        "--buy-grid",
        type=str,
        default="0.0001,0.25,0.50",
        help="Comma-separated buy-threshold grid.",
    )
    parser.add_argument(
        "--sell-grid",
        type=str,
        default="-0.0001,-0.25,-0.50",
        help="Comma-separated sell-threshold grid.",
    )
    parser.add_argument(
        "--duration-grid",
        type=str,
        default="6.0,8.5,10.0",
        help="Comma-separated treasury-duration grid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = (args.out_dir or (root / "reports" / "robustness")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from backtest.engine import backtest_from_weights
    from backtest.reporting import perf_table
    from data.loader import (
        DEFAULT_ASSETS,
        apply_treasury_total_return,
        load_universe,
        universe_returns_matrix,
    )
    from portfolio.weights import weights_from_primary_signal
    from signals.variant1 import build_primary_signal_variant1

    tcost_grid = _parse_float_list(args.tcost_grid_bps)
    buy_grid = _parse_float_list(args.buy_grid)
    sell_grid = _parse_float_list(args.sell_grid)
    duration_grid = _parse_float_list(args.duration_grid)

    clean_dir = root / "data" / "clean"
    universe = load_universe(clean_dir, list(DEFAULT_ASSETS))

    rows: list[dict[str, float | int | str]] = []
    scenario_counter = 0

    for duration in duration_grid:
        adjusted = apply_treasury_total_return(universe, duration=duration)
        returns = universe_returns_matrix(adjusted)
        equal_weight_row = pd.Series(
            1.0 / len(returns.columns), index=returns.columns, dtype=float
        )

        for buy_threshold in buy_grid:
            for sell_threshold in sell_grid:
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

                for tcost_bps in tcost_grid:
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
    grid_path = out_dir / "robustness_grid_results.csv"
    grid.to_csv(grid_path, index=False)

    baseline = grid[
        (grid["duration"].round(6) == 8.5)
        & (grid["buy_threshold"].round(6) == 0.0001)
        & (grid["sell_threshold"].round(6) == -0.0001)
    ].sort_values("tcost_bps")
    baseline_path = out_dir / "baseline_cost_sensitivity.csv"
    baseline.to_csv(baseline_path, index=False)

    top15_path = out_dir / "top15_by_sharpe.csv"
    grid.head(15).to_csv(top15_path, index=False)

    md_lines = [
        "# Robustness Summary",
        "",
        f"- Scenarios evaluated: `{len(grid)}`",
        f"- Duration grid: `{duration_grid}`",
        f"- Buy thresholds: `{buy_grid}`",
        f"- Sell thresholds: `{sell_grid}`",
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


if __name__ == "__main__":
    main()
