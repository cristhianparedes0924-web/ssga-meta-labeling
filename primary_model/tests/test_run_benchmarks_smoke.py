from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from primary_model.data import universe_returns_matrix  # noqa: E402
from primary_model.signals import build_primary_signal_variant1  # noqa: E402
from primary_model.strategy import weights_from_primary_signal  # noqa: E402


def _asset_df(prices: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"Price": prices, "Return": prices.pct_change()})


def test_primary_v1_weights_shape_and_constraints() -> None:
    dates = pd.date_range("2022-01-31", periods=10, freq="ME")
    universe = {
        "spx": _asset_df(pd.Series(np.linspace(100.0, 115.0, len(dates)), index=dates)),
        "bcom": _asset_df(pd.Series(np.linspace(80.0, 90.0, len(dates)), index=dates)),
        "treasury_10y": _asset_df(pd.Series(np.linspace(2.0, 2.6, len(dates)), index=dates)),
        "corp_bonds": _asset_df(pd.Series(np.linspace(95.0, 102.0, len(dates)), index=dates)),
    }

    returns = universe_returns_matrix(universe)
    signals = build_primary_signal_variant1(
        universe,
        trend_window=2,
        relative_window=1,
        zscore_min_periods=2,
    )
    weights = weights_from_primary_signal(
        signal=signals["signal"].reindex(returns.index),
        returns_columns=list(returns.columns),
    )

    assert list(weights.columns) == list(returns.columns)
    assert np.allclose(weights.sum(axis=1).values, np.ones(len(weights)))
    assert (weights >= 0.0).all().all()
