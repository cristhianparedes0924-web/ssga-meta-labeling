"""Supplemental feature engineering for the secondary meta-labeling dataset.

VIX and Liquidity (OAS) features provide regime context that the primary model
does not capture. These are used exclusively as M2 input features — they are
never added to primary signals or portfolio weights.

All z-scores use the same expanding approach as the primary model:
z-score at time t uses only data through t-1, so there is no look-ahead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from metalabel import PROJECT_ROOT
from metalabel.data import SUPPLEMENTAL_ASSETS, load_supplemental_level_csv
from metalabel.primary.signals import expanding_zscore


def _expanding_regime(series: pd.Series, min_periods: int = 12) -> pd.Series:
    """Return 1 where series is above its expanding median, 0 otherwise.

    The expanding median at time t is computed from data through t-1 only,
    matching the no-look-ahead convention used throughout the project.
    """
    prior = series.shift(1)
    median = prior.expanding(min_periods=min_periods).median()
    regime = (series > median).astype(float)
    regime[series.isna() | median.isna()] = np.nan
    return regime


def build_vix_features(
    vix: pd.Series,
    trend_window: int = 6,
    zscore_min_periods: int = 12,
) -> pd.DataFrame:
    """Build VIX-derived features for the secondary dataset.

    Parameters
    ----------
    vix:
        Date-indexed series of monthly VIX closing levels.
    trend_window:
        Rolling window (months) for the short-term trend indicator.
    zscore_min_periods:
        Minimum history required before z-scores are computed.

    Returns
    -------
    DataFrame with columns:
        vix_level_z       : expanding z-score of VIX level
        vix_change        : month-over-month change in VIX
        vix_change_z      : expanding z-score of VIX change
        vix_trend         : VIX level vs rolling mean (normalised deviation)
        vix_high_regime   : 1 if VIX above expanding median, else 0
        vix_rising        : 1 if VIX change > 0, else 0
    """
    vix = pd.to_numeric(vix, errors="coerce")

    level_z = expanding_zscore(vix, min_periods=zscore_min_periods).rename("vix_level_z")

    change = vix.diff().rename("vix_change")
    change_z = expanding_zscore(change, min_periods=zscore_min_periods).rename("vix_change_z")

    rolling_mean = vix.rolling(trend_window, min_periods=trend_window).mean()
    trend = ((vix / rolling_mean) - 1.0).rename("vix_trend")

    high_regime = _expanding_regime(vix, min_periods=zscore_min_periods).rename("vix_high_regime")
    rising = (change > 0.0).astype(float).where(change.notna()).rename("vix_rising")

    return pd.concat([level_z, change, change_z, trend, high_regime, rising], axis=1)


def build_liquidity_features(
    oas: pd.Series,
    trend_window: int = 6,
    zscore_min_periods: int = 12,
) -> pd.DataFrame:
    """Build liquidity/OAS-derived features for the secondary dataset.

    ``oas`` is the US Investment-Grade Corporate Option-Adjusted Spread (LUACOAS).
    A higher OAS means wider credit spreads and more credit stress / illiquidity.
    A widening OAS (positive change) signals deteriorating credit conditions.

    Parameters
    ----------
    oas:
        Date-indexed series of monthly OAS levels (in percent, e.g. 0.78).
    trend_window:
        Rolling window (months) for the short-term trend indicator.
    zscore_min_periods:
        Minimum history required before z-scores are computed.

    Returns
    -------
    DataFrame with columns:
        oas_level_z       : expanding z-score of OAS level (high = credit stress)
        oas_change        : month-over-month change in OAS (positive = widening)
        oas_change_z      : expanding z-score of OAS change
        oas_trend         : OAS vs rolling mean (normalised deviation)
        oas_wide_regime   : 1 if OAS above expanding median (stress regime), else 0
        oas_widening      : 1 if OAS change > 0 (spreads widening), else 0
    """
    oas = pd.to_numeric(oas, errors="coerce")

    level_z = expanding_zscore(oas, min_periods=zscore_min_periods).rename("oas_level_z")

    change = oas.diff().rename("oas_change")
    change_z = expanding_zscore(change, min_periods=zscore_min_periods).rename("oas_change_z")

    rolling_mean = oas.rolling(trend_window, min_periods=trend_window).mean()
    trend = ((oas / rolling_mean) - 1.0).rename("oas_trend")

    wide_regime = _expanding_regime(oas, min_periods=zscore_min_periods).rename("oas_wide_regime")
    widening = (change > 0.0).astype(float).where(change.notna()).rename("oas_widening")

    return pd.concat([level_z, change, change_z, trend, wide_regime, widening], axis=1)


def build_supplemental_features(
    root: Path = PROJECT_ROOT,
    trend_window: int = 6,
    zscore_min_periods: int = 12,
) -> pd.DataFrame:
    """Load VIX and Liquidity CSVs and build all supplemental features.

    Expects ``root/data/clean/vix.csv`` and ``root/data/clean/liquidity.csv``
    to exist. Run ``prepare-supplemental-data`` CLI command to generate them
    from the raw Excel files if they are missing.

    Returns a Date-indexed DataFrame with 12 columns (6 VIX + 6 OAS).
    """
    clean_dir = root / "data" / "clean"

    missing = [s for s in SUPPLEMENTAL_ASSETS if not (clean_dir / f"{s}.csv").exists()]
    if missing:
        raise FileNotFoundError(
            f"Supplemental clean CSVs not found: {missing}. "
            f"Run 'python -m metalabel.cli prepare-supplemental-data' first."
        )

    vix = load_supplemental_level_csv(clean_dir / "vix.csv")
    oas = load_supplemental_level_csv(clean_dir / "liquidity.csv")

    vix_features = build_vix_features(vix, trend_window=trend_window, zscore_min_periods=zscore_min_periods)
    oas_features = build_liquidity_features(oas, trend_window=trend_window, zscore_min_periods=zscore_min_periods)

    combined = pd.concat([vix_features, oas_features], axis=1).sort_index()
    return combined
