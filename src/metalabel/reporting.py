"""Plotting and reporting helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from metalabel.primary.metrics import sharpe_ratio


def reports_results_dir(root: Path) -> Path:
    return Path(root) / "reports" / "results"


def reports_assets_dir(root: Path) -> Path:
    return Path(root) / "reports" / "assets"


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


def build_asset_summary(universe: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create row-level quality summary for each asset."""
    rows: list[dict[str, object]] = []

    for asset, df in universe.items():
        rows.append(
            {
                "asset": asset,
                "rows": int(len(df)),
                "min_date": df.index.min().date().isoformat() if len(df) else None,
                "max_date": df.index.max().date().isoformat() if len(df) else None,
                "pct_missing_return": float(df["Return"].isna().mean() * 100.0),
                "price_min": float(df["Price"].min()) if len(df) else np.nan,
                "price_max": float(df["Price"].max()) if len(df) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def maybe_yield_warning(universe: dict[str, pd.DataFrame]) -> str | None:
    """Warn if treasury prices appear to be yield-level inputs."""
    treasury = universe.get("treasury_10y")
    if treasury is None or treasury.empty:
        return None

    price_min = float(treasury["Price"].min())
    price_max = float(treasury["Price"].max())
    if 0.0 <= price_min <= 20.0 and 0.0 <= price_max <= 20.0:
        return (
            "WARNING: treasury_10y Price appears yield-like "
            f"(range {price_min:.4f} to {price_max:.4f}). "
            "Treat this as a yield level series, not a bond total-return index."
        )
    return None


def annualized_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute annualized mean and vol under monthly frequency assumption."""
    return pd.DataFrame(
        {
            "ann_mean": returns.mean() * 12.0,
            "ann_vol": returns.std(ddof=1) * np.sqrt(12.0),
        }
    )


def build_benchmark_html(
    summary: pd.DataFrame,
    excess: pd.DataFrame,
    corr: pd.DataFrame,
    asset_stats: pd.DataFrame,
    asset_corr: pd.DataFrame,
    equity_plot_path: Path,
    drawdowns_plot_path: Path,
    rolling_sharpe_plot_path: Path,
) -> str:
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
        f"<img src='../assets/{equity_plot_path.name}' alt='Equity curves'>",
        "<h3>Drawdowns</h3>",
        f"<img src='../assets/{drawdowns_plot_path.name}' alt='Drawdowns'>",
        "<h3>Rolling Sharpe (12M)</h3>",
        f"<img src='../assets/{rolling_sharpe_plot_path.name}' alt='Rolling Sharpe'>",
        "</body></html>",
    ]
    return "\n".join(html_parts)


def build_data_qc_html(
    asset_summary: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    corr: pd.DataFrame,
    raw_ann_stats: pd.DataFrame,
    adj_ann_stats: pd.DataFrame,
    warning_line: str | None,
) -> str:
    html_parts = [
        "<html><head><meta charset='utf-8'><title>Data QC Report</title></head><body>",
        "<h1>Data QC Report</h1>",
        "<h2>Asset Summary</h2>",
        asset_summary.to_html(index=False),
        "<h2>Overlap Summary</h2>",
        overlap_summary.to_html(index=False),
        "<h2>Returns Correlation Matrix</h2>",
        corr.to_html(),
        "<h2>Treasury Return Handling</h2>",
        (
            "<p>Raw treasury returns reflect percent changes in yield levels. "
            "Adjusted treasury returns use a duration-based approximation to "
            "proxy bond total returns, while keeping the treasury Price as the yield level.</p>"
        ),
        "<h3>Raw Annualized Stats (Monthly Assumption)</h3>",
        raw_ann_stats.to_html(),
        "<h3>Adjusted Annualized Stats (Duration-Based Treasury, Monthly Assumption)</h3>",
        adj_ann_stats.to_html(),
    ]
    if warning_line:
        html_parts.extend(["<h2>Warning</h2>", f"<p>{warning_line}</p>"])
    html_parts.append("</body></html>")
    return "\n".join(html_parts)
