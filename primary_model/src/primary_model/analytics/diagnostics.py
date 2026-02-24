"""Lightweight diagnostics and sanity checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from primary_model.analytics.performance import sharpe_ratio


def strategy_return_table(backtests: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Extract net returns for a dict of backtests into a single DataFrame."""
    return pd.DataFrame(
        {name: bt["net_return"] for name, bt in backtests.items()}
    ).sort_index()


def excess_vs_equal_weight(strategy_returns: pd.DataFrame, ew_col: str) -> pd.DataFrame:
    """Compute excess returns of strategies against an equal-weight baseline."""
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


def corr_vs_equal_weight(strategy_returns: pd.DataFrame, ew_col: str) -> pd.DataFrame:
    """Compute correlation of strategies vs an equal-weight baseline."""
    ew = strategy_returns[ew_col]
    corr = strategy_returns.corrwith(ew)
    return corr.to_frame(name="corr_with_equal_weight")


def asset_sanity_table(returns: pd.DataFrame) -> pd.DataFrame:
    """Generate sanity metrics for raw or adjusted asset returns."""
    table = pd.DataFrame(index=returns.columns)
    table["ann_mean"] = returns.mean() * 12.0
    table["ann_vol"] = returns.std(ddof=1) * np.sqrt(12.0)
    table["sharpe"] = [
        sharpe_ratio(returns[col], rf_annual=0.0, periods_per_year=12)
        for col in returns.columns
    ]
    return table


__all__ = [
    "asset_sanity_table",
    "corr_vs_equal_weight",
    "excess_vs_equal_weight",
    "strategy_return_table",
]
