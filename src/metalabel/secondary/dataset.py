"""Leakage-safe secondary dataset construction utilities.

Each secondary row represents a primary decision date ``t``. Feature columns
must only use information known at ``t``. The meta target uses the realized
next-period primary-strategy outcome from ``t`` to ``t+1``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from metalabel import PROJECT_ROOT, load_primary_config
from metalabel.data import DEFAULT_ASSETS, _resolve_within_project, apply_treasury_total_return, load_universe, universe_returns_matrix
from metalabel.primary.backtest import backtest_from_weights
from metalabel.primary.portfolio import weights_from_primary_signal
from metalabel.primary.signals import build_primary_signal_variant1
from metalabel.secondary.features import build_supplemental_features


_ACTIONABLE_SIGNALS = frozenset({"BUY", "SELL"})
_ALL_EVENT_SIGNALS = frozenset({"BUY", "SELL", "HOLD"})


def _resolved_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return load_primary_config() if config is None else dict(config)


def _primary_settings(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = _resolved_config(config)
    return dict(cfg.get("primary", cfg))


def _load_secondary_universe(root: Path) -> dict[str, pd.DataFrame]:
    clean_dir = root / "data" / "clean"
    return load_universe(clean_dir, list(DEFAULT_ASSETS))


def _realized_date_series(index: pd.Index) -> pd.Series:
    if len(index) < 2:
        return pd.Series(dtype="datetime64[ns]", name="realized_date")
    return pd.Series(index[1:], index=index[:-1], name="realized_date")


def _signal_streak(signal: pd.Series) -> pd.Series:
    streak = pd.Series(np.nan, index=signal.index, dtype=float, name="signal_streak")
    run_length = 0
    previous_label: str | None = None

    for ts, raw_value in signal.items():
        if pd.isna(raw_value):
            run_length = 0
            previous_label = None
            continue

        label = str(raw_value)
        if label == previous_label:
            run_length += 1
        else:
            run_length = 1
            previous_label = label
        streak.loc[ts] = float(run_length)

    return streak


def _trailing_health_features(backtest: pd.DataFrame, signal: pd.Series, trailing_window: int) -> pd.DataFrame:
    if trailing_window < 1:
        raise ValueError("trailing_window must be >= 1.")

    # Shift by one row so each decision-date feature only sees outcomes that were
    # already realized before that decision. The current row's t->t+1 outcome is
    # reserved for meta_target_return / meta_label only.
    prior_net_return = pd.to_numeric(backtest["net_return"], errors="coerce").shift(1)
    prior_turnover = pd.to_numeric(backtest["turnover"], errors="coerce").shift(1)
    prior_hit = prior_net_return.gt(0.0).astype(float).where(prior_net_return.notna())

    features = pd.DataFrame(index=backtest.index)
    features[f"trailing_hit_rate_{trailing_window}"] = prior_hit.rolling(
        trailing_window,
        min_periods=1,
    ).mean()
    features[f"trailing_avg_net_return_{trailing_window}"] = prior_net_return.rolling(
        trailing_window,
        min_periods=1,
    ).mean()
    features[f"trailing_vol_net_return_{trailing_window}"] = prior_net_return.rolling(
        trailing_window,
        min_periods=2,
    ).std(ddof=0)
    features[f"trailing_avg_turnover_{trailing_window}"] = prior_turnover.rolling(
        trailing_window,
        min_periods=1,
    ).mean()
    features["signal_streak"] = _signal_streak(signal.reindex(backtest.index))
    return features


def build_secondary_dataset(
    *,
    root: Path = PROJECT_ROOT,
    universe: dict[str, pd.DataFrame] | None = None,
    config: Mapping[str, Any] | None = None,
    include_hold: bool = False,
    trailing_window: int = 12,
    indicator_weights: Mapping[str, float] | None = None,
    use_supplemental: bool = False,
) -> pd.DataFrame:
    """Build a first-pass secondary dataset from the protected primary baseline.

    Each row is a primary decision event at date ``t``. By default, only
    actionable primary signals (``BUY`` / ``SELL``) are retained. The
    ``meta_target_return`` and ``meta_label`` columns come from the realized
    next-period net return produced by the primary-strategy weights chosen at
    ``t``.

    Parameters
    ----------
    use_supplemental:
        When True, load VIX and Liquidity (OAS) supplemental features from
        ``root/data/clean/`` and append them as additional columns. These
        provide regime context for the M2 classifier. Requires the supplemental
        CSVs to exist — run ``prepare-supplemental-data`` first.
    """

    primary_cfg = _primary_settings(config)
    resolved_root = _resolve_within_project(root, "root")
    if universe is None:
        source_universe = _load_secondary_universe(resolved_root)
    else:
        source_universe = {asset: frame.copy(deep=True) for asset, frame in universe.items()}

    adjusted_universe = apply_treasury_total_return(
        source_universe,
        duration=float(primary_cfg["duration"]),
    )
    returns = universe_returns_matrix(adjusted_universe)

    signals = build_primary_signal_variant1(
        adjusted_universe,
        trend_window=int(primary_cfg["trend_window"]),
        relative_window=int(primary_cfg["relative_window"]),
        zscore_min_periods=int(primary_cfg["zscore_min_periods"]),
        indicator_weights=indicator_weights,
        buy_threshold=float(primary_cfg["buy_threshold"]),
        sell_threshold=float(primary_cfg["sell_threshold"]),
    )

    weights = weights_from_primary_signal(
        signal=signals["signal"],
        returns_columns=list(returns.columns),
    )
    weights = weights.reindex(returns.index).ffill()
    equal_weight_row = pd.Series(1.0 / len(returns.columns), index=returns.columns, dtype=float)
    weights = weights.fillna(equal_weight_row)

    backtest = backtest_from_weights(
        returns=returns,
        weights=weights,
        tcost_bps=float(primary_cfg["tcost_bps"]),
    )

    events = signals.reindex(backtest.index).copy()
    events = events.rename(columns={"signal": "primary_signal"})
    events["realized_date"] = _realized_date_series(returns.index).reindex(events.index)
    events["meta_target_gross_return"] = backtest["gross_return"]
    events["meta_target_return"] = backtest["net_return"]
    events["meta_label"] = (events["meta_target_return"] > 0.0).astype(int)
    events["event_turnover"] = backtest["turnover"]

    weight_features = weights.reindex(backtest.index).add_prefix("weight_")
    trailing_features = _trailing_health_features(
        backtest=backtest,
        signal=signals["signal"],
        trailing_window=trailing_window,
    )
    events = events.join(weight_features).join(trailing_features)

    supplemental_columns: list[str] = []
    if use_supplemental:
        supp = build_supplemental_features(root=resolved_root)
        supp = supp.reindex(events.index)
        events = events.join(supp)
        supplemental_columns = list(supp.columns)

    allowed_signals = _ALL_EVENT_SIGNALS if include_hold else _ACTIONABLE_SIGNALS
    events = events[events["primary_signal"].isin(allowed_signals)].copy()

    base_columns = [
        "realized_date",
        "primary_signal",
        "composite_score",
        "meta_target_gross_return",
        "meta_target_return",
        "meta_label",
        "event_turnover",
    ]
    indicator_columns = [
        col for col in signals.columns if col not in {"signal", "composite_score"}
    ]
    weight_columns = list(weight_features.columns)
    trailing_columns = list(trailing_features.columns)

    events.index.name = "date"
    ordered = base_columns + indicator_columns + weight_columns + trailing_columns + supplemental_columns
    dataset = events[ordered].reset_index()
    return dataset


def save_secondary_dataset(dataset: pd.DataFrame, path: Path) -> Path:
    """Save a built secondary dataset to a CSV path inside the project root."""

    output_path = _resolve_within_project(path, "path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    return output_path

