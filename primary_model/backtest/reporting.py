"""Benchmark reporting, plotting, and pipeline run helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from backtest.engine import (
    annualized_return,
    annualized_vol,
    backtest_from_weights,
    max_drawdown,
    sharpe_ratio,
)
from data.loader import (
    DEFAULT_ASSETS,
    apply_treasury_total_return,
    load_universe,
    universe_returns_matrix,
)
from portfolio.weights import (
    weights_6040,
    weights_buy_hold_spx,
    weights_equal_weight,
    weights_from_primary_signal,
    weights_simple_trend,
)
from signals.variant1 import build_primary_signal_variant1


def perf_table(
    backtests: dict[str, pd.DataFrame], periods_per_year: int = 12
) -> pd.DataFrame:
    """Build standardized performance summary table."""
    columns = [
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "calmar",
        "avg_turnover",
    ]
    if not backtests:
        return pd.DataFrame(columns=columns)

    rows: dict[str, dict[str, float]] = {}
    for name, df in backtests.items():
        required = {"net_return", "equity_net"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{name}: missing required columns: {sorted(missing)}")

        net = df["net_return"]
        equity = df["equity_net"]

        ann_ret = annualized_return(net, periods_per_year=periods_per_year)
        ann_vol = annualized_vol(net, periods_per_year=periods_per_year)
        shp = sharpe_ratio(net, rf_annual=0.0, periods_per_year=periods_per_year)
        mdd = max_drawdown(equity)

        calmar = float(np.nan)
        if not np.isnan(ann_ret) and not np.isnan(mdd) and mdd < 0.0:
            calmar = float(ann_ret / abs(mdd))

        avg_turnover = float(np.nan)
        if "turnover" in df.columns:
            avg_turnover = float(pd.to_numeric(df["turnover"], errors="coerce").mean())

        rows[name] = {
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": shp,
            "max_drawdown": mdd,
            "calmar": calmar,
            "avg_turnover": avg_turnover,
        }

    return pd.DataFrame.from_dict(rows, orient="index")[columns]


def plot_equity_curves(backtests: dict[str, pd.DataFrame], out_path: Path) -> Path:
    """Plot net equity curves for all strategies."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, df in backtests.items():
        if "equity_net" not in df.columns:
            raise ValueError(f"{name}: missing required column 'equity_net'.")
        equity = pd.to_numeric(df["equity_net"], errors="coerce")
        ax.plot(equity.index, equity.values, label=name, linewidth=1.8)

    ax.set_title("Equity Curves (Net)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_drawdowns(backtests: dict[str, pd.DataFrame], out_path: Path) -> Path:
    """Plot drawdown series for all strategies."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, df in backtests.items():
        if "equity_net" not in df.columns:
            raise ValueError(f"{name}: missing required column 'equity_net'.")
        equity = pd.to_numeric(df["equity_net"], errors="coerce")
        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        ax.plot(drawdown.index, drawdown.values, label=name, linewidth=1.8)

    ax.set_title("Drawdowns (Net Equity)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_rolling_sharpe(
    backtests: dict[str, pd.DataFrame], out_path: Path, window: int = 12
) -> Path:
    """Plot rolling Sharpe for each strategy using net returns."""
    if window <= 1:
        raise ValueError("window must be > 1 for rolling Sharpe.")

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, df in backtests.items():
        if "net_return" not in df.columns:
            raise ValueError(f"{name}: missing required column 'net_return'.")
        net = pd.to_numeric(df["net_return"], errors="coerce")
        rolling_mean = net.rolling(window=window, min_periods=window).mean()
        rolling_std = net.rolling(window=window, min_periods=window).std(ddof=1)
        rolling_sharpe = rolling_mean / rolling_std * np.sqrt(12.0)
        rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan)
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, label=name, linewidth=1.8)

    ax.set_title(f"Rolling Sharpe (Net, {window}-month)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


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


def run_primary_variant1(root: Path) -> None:
    """Run primary strategy pipeline and save backtest summary outputs."""
    clean_dir = root / "data" / "clean"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    universe = load_universe(clean_dir, DEFAULT_ASSETS)
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


def run_benchmarks(root: Path) -> None:
    """Run benchmark comparisons and export summary reports/plots."""
    clean_dir = root / "data" / "clean"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = reports_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    universe = load_universe(clean_dir, DEFAULT_ASSETS)
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


__all__ = [
    "perf_table",
    "plot_drawdowns",
    "plot_equity_curves",
    "plot_rolling_sharpe",
    "run_benchmarks",
    "run_primary_variant1",
]
