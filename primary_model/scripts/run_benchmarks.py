#!/usr/bin/env python3
"""Run benchmark backtests on real clean data and export summary tables."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from primary_model.backtest import backtest_from_weights  # noqa: E402
from primary_model.benchmarks import (  # noqa: E402
    weights_6040,
    weights_buy_hold_spx,
    weights_equal_weight,
    weights_simple_trend,
)
from primary_model.data import (  # noqa: E402
    apply_treasury_total_return,
    load_universe,
    universe_returns_matrix,
)
from primary_model.metrics import perf_table, sharpe_ratio  # noqa: E402
from primary_model.plots import (  # noqa: E402
    plot_drawdowns,
    plot_equity_curves,
    plot_rolling_sharpe,
)
from primary_model.signals import build_primary_signal_variant1  # noqa: E402
from primary_model.strategy import weights_from_primary_signal  # noqa: E402

ASSETS = ["spx", "bcom", "treasury_10y", "corp_bonds"]


def _strategy_return_table(backtests: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame(
        {name: bt["net_return"] for name, bt in backtests.items()}
    ).sort_index()


def _excess_vs_equal_weight(strategy_returns: pd.DataFrame, ew_col: str) -> pd.DataFrame:
    ew = strategy_returns[ew_col]
    rows = []
    for col in strategy_returns.columns:
        diff = strategy_returns[col] - ew
        rows.append(
            {
                "strategy": col,
                "mean_excess_annual": float(diff.mean() * 12.0),
                "months_won": int((diff > 0).sum()),
                "months_lost": int((diff < 0).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


def _corr_vs_equal_weight(strategy_returns: pd.DataFrame, ew_col: str) -> pd.DataFrame:
    ew = strategy_returns[ew_col]
    corr = strategy_returns.corrwith(ew)
    return corr.to_frame(name="corr_with_equal_weight")


def _asset_sanity_table(returns: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame(index=returns.columns)
    table["ann_mean"] = returns.mean() * 12.0
    table["ann_vol"] = returns.std(ddof=1) * np.sqrt(12.0)
    table["sharpe"] = [
        sharpe_ratio(returns[col], rf_annual=0.0, periods_per_year=12)
        for col in returns.columns
    ]
    return table


def main() -> None:
    clean_dir = ROOT / "data" / "clean"
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = reports_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    universe = load_universe(clean_dir, ASSETS)
    adj_universe = apply_treasury_total_return(universe, duration=8.5)
    returns = universe_returns_matrix(adj_universe)
    primary_signals = build_primary_signal_variant1(adj_universe)

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
        name: backtest_from_weights(returns=returns, weights=w, tcost_bps=0.0)
        for name, w in weights_by_name.items()
    }

    summary = perf_table(backtests, periods_per_year=12)
    strategy_returns = _strategy_return_table(backtests)
    excess = _excess_vs_equal_weight(strategy_returns, ew_col="EqualWeight25")
    corr = _corr_vs_equal_weight(strategy_returns, ew_col="EqualWeight25")
    asset_stats = _asset_sanity_table(returns)
    asset_corr = returns.corr()

    print("Benchmark performance table (net):")
    print(summary.to_string())
    print()

    primary_signal_counts = (
        primary_signals["signal"]
        .value_counts(dropna=True)
        .reindex(["BUY", "HOLD", "SELL"], fill_value=0)
        .astype(int)
    )
    primary_avg_turnover = float(backtests["PrimaryV1"]["turnover"].mean())
    print("PrimaryV1 signal counts:")
    print(primary_signal_counts.to_string())
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

    summary.to_csv(csv_path, index=True)
    primary_signals.to_csv(primary_signal_path, index=True)
    primary_weights.to_csv(primary_weights_path, index=True)

    html_parts = [
        (
            "<html><head><meta charset='utf-8'><title>Benchmark Summary</title>"
            "<style>"
            "body{font-family:Arial,sans-serif;margin:20px;color:#111;}"
            "h1,h2,h3{margin-top:20px;}"
            "table{border-collapse:collapse;margin-bottom:16px;}"
            "th,td{border:1px solid #d0d0d0;padding:6px 10px;text-align:right;}"
            "th:first-child,td:first-child{text-align:left;}"
            "img{max-width:100%;height:auto;border:1px solid #d0d0d0;margin-bottom:16px;}"
            "</style></head><body>"
        ),
        "<h1>Benchmark Summary</h1>",
        "<h2>Performance Table (Net)</h2>",
        summary.to_html(),
        "<h2>Excess vs EqualWeight25</h2>",
        excess.to_html(),
        "<h2>Correlation with EqualWeight25 Net Returns</h2>",
        corr.to_html(),
        "<h2>Benchmark Sanity Check</h2>",
        "<h3>Asset-Level Annualized Stats (Adjusted Returns)</h3>",
        asset_stats.to_html(),
        "<h3>Asset Return Correlation Matrix (Adjusted Returns)</h3>",
        asset_corr.to_html(),
        "<ul>",
        "<li>EqualWeight25 controls for exposure to this fixed four-asset universe by allocating 25% to each asset every month.</li>",
        "<li>It does not control for volatility targeting, risk parity weighting, or dynamic leverage constraints.</li>",
        "<li>It also does not control for timing skill; any excess over EqualWeight25 reflects allocation and timing differences.</li>",
        "</ul>",
        "<h2>Plots</h2>",
        "<h3>Equity Curves (Net)</h3>",
        f"<img src='assets/{equity_plot_path.name}' alt='Equity curves'>",
        "<h3>Drawdowns</h3>",
        f"<img src='assets/{drawdowns_plot_path.name}' alt='Drawdowns'>",
        "<h3>Rolling Sharpe (12M)</h3>",
        f"<img src='assets/{rolling_sharpe_plot_path.name}' alt='Rolling Sharpe'>",
        "</body></html>",
    ]
    html_path.write_text("\n".join(html_parts), encoding="utf-8")

    print(f"Saved CSV: {csv_path}")
    print(f"Saved CSV: {primary_signal_path}")
    print(f"Saved CSV: {primary_weights_path}")
    print(f"Saved HTML: {html_path}")
    print(f"Saved plot: {equity_plot_path}")
    print(f"Saved plot: {drawdowns_plot_path}")
    print(f"Saved plot: {rolling_sharpe_plot_path}")


if __name__ == "__main__":
    main()
