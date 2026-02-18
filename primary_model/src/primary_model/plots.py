"""Plot utilities for benchmark backtest outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_equity_curves(backtests: dict[str, pd.DataFrame], out_path: Path) -> Path:
    """Plot equity_net curves for all strategies and save to PNG."""
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
    """Plot drawdown curves derived from equity_net and save to PNG."""
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
    """Plot rolling annualized Sharpe on net_return and save to PNG."""
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
