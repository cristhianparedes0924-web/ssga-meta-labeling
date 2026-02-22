#!/usr/bin/env python3
"""Run strict walk-forward validation for PrimaryV1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict walk-forward evaluator.")
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
        help="Output directory (default: <root>/reports/walk_forward).",
    )
    parser.add_argument(
        "--min-train-periods",
        type=int,
        default=120,
        help="Minimum in-sample history before first OOS decision.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=8.5,
        help="Treasury duration used for return approximation.",
    )
    parser.add_argument(
        "--buy-threshold",
        type=float,
        default=0.0001,
        help="BUY threshold for signal mapping.",
    )
    parser.add_argument(
        "--sell-threshold",
        type=float,
        default=-0.0001,
        help="SELL threshold for signal mapping.",
    )
    parser.add_argument(
        "--tcost-bps",
        type=float,
        default=0.0,
        help="Transaction costs in bps.",
    )
    return parser.parse_args()


def _strict_walk_forward(
    adjusted_universe: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    min_train_periods: int,
    buy_threshold: float,
    sell_threshold: float,
    tcost_bps: float,
    build_primary_signal_variant1,
    weights_from_primary_signal,
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


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = (args.out_dir or (root / "reports" / "walk_forward")).resolve()
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

    clean_dir = root / "data" / "clean"
    universe = load_universe(clean_dir, list(DEFAULT_ASSETS))
    adjusted_universe = apply_treasury_total_return(universe, duration=args.duration)
    returns = universe_returns_matrix(adjusted_universe)

    wf_backtest = _strict_walk_forward(
        adjusted_universe=adjusted_universe,
        returns=returns,
        min_train_periods=args.min_train_periods,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        tcost_bps=args.tcost_bps,
        build_primary_signal_variant1=build_primary_signal_variant1,
        weights_from_primary_signal=weights_from_primary_signal,
    )

    full_signals = build_primary_signal_variant1(
        adjusted_universe,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
    )
    full_weights = weights_from_primary_signal(
        signal=full_signals["signal"],
        returns_columns=list(returns.columns),
    )
    equal_weight_row = pd.Series(1.0 / len(returns.columns), index=returns.columns, dtype=float)
    full_weights = full_weights.reindex(returns.index).ffill().fillna(equal_weight_row)
    full_backtest = backtest_from_weights(returns=returns, weights=full_weights, tcost_bps=args.tcost_bps)
    full_backtest = full_backtest.loc[wf_backtest.index.min() :].copy()

    summary = perf_table(
        {
            "WalkForwardStrict": wf_backtest,
            "StandardCausal": full_backtest,
        }
    )
    summary["delta_vs_standard"] = summary["ann_return"] - summary.loc["StandardCausal", "ann_return"]

    wf_path = out_dir / "walk_forward_backtest.csv"
    std_path = out_dir / "standard_causal_backtest_slice.csv"
    summary_path = out_dir / "walk_forward_summary.csv"
    wf_backtest.to_csv(wf_path, index=True)
    full_backtest.to_csv(std_path, index=True)
    summary.to_csv(summary_path, index=True)

    protocol = [
        "# Walk-Forward Protocol",
        "",
        "- Train window: expanding from first observation to decision date `t`.",
        "- Decision at `t`: compute signal using only data through `t`.",
        "- Realization at `t+1`: apply weight decided at `t` to next-period return.",
        f"- Minimum train periods before first OOS decision: `{args.min_train_periods}`.",
        f"- Treasury duration assumption: `{args.duration}`.",
        f"- Thresholds: buy `{args.buy_threshold}`, sell `{args.sell_threshold}`.",
        f"- Transaction cost: `{args.tcost_bps}` bps.",
        f"- OOS decisions evaluated: `{len(wf_backtest)}`.",
        f"- First OOS decision date: `{wf_backtest.index.min().date().isoformat()}`.",
        f"- Last OOS decision date: `{wf_backtest.index.max().date().isoformat()}`.",
        "",
        "## Outputs",
        f"- `{wf_path}`",
        f"- `{std_path}`",
        f"- `{summary_path}`",
    ]
    (out_dir / "walk_forward_protocol.md").write_text("\n".join(protocol), encoding="utf-8")

    print(f"Saved: {wf_path}")
    print(f"Saved: {std_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {out_dir / 'walk_forward_protocol.md'}")


if __name__ == "__main__":
    main()
