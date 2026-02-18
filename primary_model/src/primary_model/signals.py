"""Primary-model signal construction utilities."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

_REQUIRED_ASSETS = ("spx", "bcom", "treasury_10y", "corp_bonds")


def expanding_zscore(
    series: pd.Series,
    min_periods: int = 12,
    ddof: int = 1,
) -> pd.Series:
    """Compute expanding z-score using only prior observations.

    At time ``t``, normalization statistics are computed from ``[0, ..., t-1]``.
    This avoids look-ahead while still allowing the current value to be scored.
    """
    if min_periods < 1:
        raise ValueError("min_periods must be >= 1.")

    x = pd.to_numeric(series, errors="coerce")
    history = x.shift(1)
    hist_mean = history.expanding(min_periods=min_periods).mean()
    hist_std = history.expanding(min_periods=min_periods).std(ddof=ddof)
    z = (x - hist_mean) / hist_std
    return z.replace([np.inf, -np.inf], np.nan)


def build_variant1_indicators(
    universe: dict[str, pd.DataFrame],
    trend_window: int = 12,
    relative_window: int = 3,
) -> pd.DataFrame:
    """Build Variant 1 raw indicators from universe data.

    Indicators:
    - ``spx_trend``: SPX price relative to trailing moving average.
    - ``bcom_trend``: BCOM price relative to trailing moving average.
    - ``credit_vs_rates``: Corp-bond return minus treasury return (smoothed).
    - ``risk_breadth``: Mean risky-asset return minus treasury return (smoothed).
    """
    if trend_window < 1 or relative_window < 1:
        raise ValueError("trend_window and relative_window must be >= 1.")

    missing = [asset for asset in _REQUIRED_ASSETS if asset not in universe]
    if missing:
        raise ValueError(f"Universe is missing required asset(s): {missing}")

    spx_price = pd.to_numeric(universe["spx"]["Price"], errors="coerce")
    bcom_price = pd.to_numeric(universe["bcom"]["Price"], errors="coerce")

    spx_ret = pd.to_numeric(universe["spx"]["Return"], errors="coerce")
    bcom_ret = pd.to_numeric(universe["bcom"]["Return"], errors="coerce")
    corp_ret = pd.to_numeric(universe["corp_bonds"]["Return"], errors="coerce")
    ust_ret = pd.to_numeric(universe["treasury_10y"]["Return"], errors="coerce")

    spx_trend = spx_price / spx_price.rolling(trend_window, min_periods=trend_window).mean() - 1.0
    bcom_trend = (
        bcom_price / bcom_price.rolling(trend_window, min_periods=trend_window).mean() - 1.0
    )
    credit_vs_rates = (corp_ret - ust_ret).rolling(
        relative_window, min_periods=relative_window
    ).mean()
    risk_breadth = (pd.concat([spx_ret, bcom_ret, corp_ret], axis=1).mean(axis=1) - ust_ret).rolling(
        relative_window, min_periods=relative_window
    ).mean()

    indicators = pd.DataFrame(
        {
            "spx_trend": spx_trend,
            "bcom_trend": bcom_trend,
            "credit_vs_rates": credit_vs_rates,
            "risk_breadth": risk_breadth,
        }
    ).sort_index()
    return indicators


def _weight_series(
    columns: pd.Index,
    weights: Mapping[str, float] | None,
) -> pd.Series:
    if len(columns) == 0:
        raise ValueError("No indicator columns were provided.")

    if weights is None:
        return pd.Series(1.0, index=columns, dtype=float)

    provided = pd.Series(weights, dtype=float)
    missing = [col for col in columns if col not in provided.index]
    extra = [col for col in provided.index if col not in columns]
    if missing:
        raise ValueError(f"Missing indicator weight(s): {missing}")
    if extra:
        raise ValueError(f"Unknown indicator weight(s): {extra}")

    if np.isclose(provided.abs().sum(), 0.0):
        raise ValueError("At least one indicator weight must be non-zero.")
    return provided.reindex(columns)


def composite_score(
    zscores: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
) -> pd.Series:
    """Compute a weighted composite score from indicator z-scores."""
    w = _weight_series(zscores.columns, weights)
    weighted_sum = zscores.mul(w, axis=1).sum(axis=1, min_count=1)
    active_weight = zscores.notna().mul(w.abs(), axis=1).sum(axis=1)
    score = weighted_sum / active_weight.replace(0.0, np.nan)
    return score.replace([np.inf, -np.inf], np.nan)


def score_to_signal(
    score: pd.Series,
    buy_threshold: float = 0.5,
    sell_threshold: float = -0.5,
) -> pd.Series:
    """Convert a continuous score to discrete BUY/HOLD/SELL labels."""
    if buy_threshold <= sell_threshold:
        raise ValueError("buy_threshold must be greater than sell_threshold.")

    out = pd.Series("HOLD", index=score.index, dtype=object)
    clean_score = pd.to_numeric(score, errors="coerce")
    out.loc[clean_score > buy_threshold] = "BUY"
    out.loc[clean_score < sell_threshold] = "SELL"
    out.loc[clean_score.isna()] = np.nan
    return out


def build_primary_signal_variant1(
    universe: dict[str, pd.DataFrame],
    trend_window: int = 12,
    relative_window: int = 3,
    zscore_min_periods: int = 12,
    indicator_weights: Mapping[str, float] | None = None,
    buy_threshold: float = 0.5,
    sell_threshold: float = -0.5,
) -> pd.DataFrame:
    """Build Variant 1 indicators, z-scores, composite score, and signal."""
    indicators = build_variant1_indicators(
        universe=universe,
        trend_window=trend_window,
        relative_window=relative_window,
    )

    zscores = indicators.apply(
        expanding_zscore,
        axis=0,
        min_periods=zscore_min_periods,
    )
    zscores.columns = [f"{col}_z" for col in zscores.columns]

    score = composite_score(
        zscores=zscores,
        weights=indicator_weights,
    ).rename("composite_score")

    signal = score_to_signal(
        score=score,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    ).rename("signal")

    return pd.concat([indicators, zscores, score, signal], axis=1)
