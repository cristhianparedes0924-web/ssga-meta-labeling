"""Performance metrics calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(r: pd.Series, periods_per_year: int = 12) -> float:
    """Calculate annualized return from a return series."""
    clean = pd.to_numeric(r, errors="coerce").dropna()
    n = len(clean)
    if n == 0 or periods_per_year <= 0:
        return float(np.nan)

    gross = float((1.0 + clean).prod())
    if gross <= 0.0:
        return float(np.nan)

    return float(gross ** (periods_per_year / n) - 1.0)


def annualized_vol(r: pd.Series, periods_per_year: int = 12) -> float:
    """Calculate annualized volatility from a return series."""
    clean = pd.to_numeric(r, errors="coerce").dropna()
    if len(clean) < 2 or periods_per_year <= 0:
        return float(np.nan)

    return float(clean.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    r: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 12
) -> float:
    """Calculate annualized Sharpe ratio from a return series."""
    clean = pd.to_numeric(r, errors="coerce").dropna()
    if len(clean) < 2 or periods_per_year <= 0 or rf_annual <= -1.0:
        return float(np.nan)

    rf_period = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess = clean - rf_period
    std_excess = float(excess.std(ddof=1))
    if std_excess == 0.0 or np.isnan(std_excess):
        return float(np.nan)

    return float(excess.mean() / std_excess * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Calculate maximum drawdown from an equity curve."""
    clean = pd.to_numeric(equity, errors="coerce").dropna()
    if len(clean) == 0:
        return float(np.nan)

    running_max = clean.cummax()
    drawdown = clean / running_max - 1.0
    return float(drawdown.min())


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


__all__ = [
    "annualized_return",
    "annualized_vol",
    "max_drawdown",
    "perf_table",
    "sharpe_ratio",
]
