#!/usr/bin/env python3
"""Run Primary Model Variant 1 backtest on real clean data."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from primary_model.backtest import backtest_from_weights  # noqa: E402
from primary_model.data import (  # noqa: E402
    apply_treasury_total_return,
    load_universe,
    universe_returns_matrix,
)
from primary_model.metrics import perf_table  # noqa: E402
from primary_model.signals import build_primary_signal_variant1  # noqa: E402
from primary_model.strategy import weights_from_primary_signal  # noqa: E402

ASSETS = ["spx", "bcom", "treasury_10y", "corp_bonds"]


def main() -> None:
    clean_dir = ROOT / "data" / "clean"
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    universe = load_universe(clean_dir, ASSETS)
    adj_universe = apply_treasury_total_return(universe, duration=8.5)
    signals = build_primary_signal_variant1(adj_universe)
    returns = universe_returns_matrix(adj_universe)

    weights_raw = weights_from_primary_signal(
        signal=signals["signal"],
        returns_columns=list(returns.columns),
    )
    weights = weights_raw.reindex(returns.index).ffill()
    equal_weight_row = pd.Series(1.0 / len(returns.columns), index=returns.columns, dtype=float)
    weights = weights.fillna(equal_weight_row)

    backtest = backtest_from_weights(returns=returns, weights=weights, tcost_bps=0.0)
    summary = perf_table({"PrimaryV1": backtest}, periods_per_year=12)

    signal_counts = (
        signals["signal"]
        .value_counts(dropna=True)
        .reindex(["BUY", "HOLD", "SELL"], fill_value=0)
        .astype(int)
    )
    avg_turnover = float(backtest["turnover"].mean())

    print("Performance table (net):")
    print(summary.to_string())
    print()

    print("Signal counts:")
    print(signal_counts.to_string())
    print()

    print(f"Average turnover: {avg_turnover:.6f}")

    backtest_path = reports_dir / "primary_v1_backtest.csv"
    summary_path = reports_dir / "primary_v1_summary.csv"
    backtest.to_csv(backtest_path, index=True)
    summary.to_csv(summary_path, index=True)

    print(f"Saved CSV: {backtest_path}")
    print(f"Saved CSV: {summary_path}")


if __name__ == "__main__":
    main()
