from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from primary_model.backtest import backtest_from_weights  # noqa: E402


def test_backtest_alignment_t_plus_1_application() -> None:
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])

    returns = pd.DataFrame(
        {
            "spx": [0.0, 0.10],
            "bcom": [0.0, 0.0],
        },
        index=dates,
    )

    weights = pd.DataFrame(
        {
            "spx": [1.0, 0.0],
            "bcom": [0.0, 1.0],
        },
        index=dates,
    )

    out = backtest_from_weights(returns=returns, weights=weights, tcost_bps=0.0)

    assert len(out) == 1
    assert out.index[0] == dates[0]
    assert out.iloc[0]["gross_return"] == pytest.approx(0.10)
    assert out.iloc[0]["net_return"] == pytest.approx(0.10)
    assert out.iloc[0]["equity_gross"] == pytest.approx(1.10)
    assert out.iloc[0]["equity_net"] == pytest.approx(1.10)
