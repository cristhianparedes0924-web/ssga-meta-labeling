from __future__ import annotations

import numpy as np
import pandas as pd

from metalabel.primary.portfolio import weights_equal_weight, weights_from_primary_signal


def test_weights_from_primary_signal_carries_hold_and_nan() -> None:
    idx = pd.date_range("2001-01-31", periods=5, freq="ME")
    sig = pd.Series(["BUY", "HOLD", "SELL", "HOLD", np.nan], index=idx)
    cols = ["spx", "bcom", "treasury_10y", "corp_bonds"]
    weights = weights_from_primary_signal(sig, returns_columns=cols)

    assert weights.loc[idx[0], cols].to_list() == [0.40, 0.15, 0.0, 0.45]
    assert weights.loc[idx[1], cols].to_list() == [0.40, 0.15, 0.0, 0.45]
    assert weights.loc[idx[2], cols].to_list() == [0.05, 0.0, 0.60, 0.35]
    assert weights.loc[idx[3], cols].to_list() == [0.05, 0.0, 0.60, 0.35]
    assert weights.loc[idx[4], cols].to_list() == [0.05, 0.0, 0.60, 0.35]


def test_weights_equal_weight_sums_to_one() -> None:
    returns = pd.DataFrame(
        {"spx": [0.01, 0.02], "bcom": [0.0, 0.01], "treasury_10y": [0.0, 0.0]},
        index=pd.date_range("2020-01-31", periods=2, freq="ME"),
    )
    weights = weights_equal_weight(returns)
    assert np.allclose(weights.sum(axis=1).to_numpy(), 1.0)
