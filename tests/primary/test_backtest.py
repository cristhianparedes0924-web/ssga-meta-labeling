from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metalabel.primary.backtest import backtest_from_weights


def test_backtest_applies_weights_to_next_period_returns() -> None:
    idx = pd.date_range("2002-01-31", periods=3, freq="ME")
    returns = pd.DataFrame({"spx": [0.10, 0.20, 0.30]}, index=idx)
    weights = pd.DataFrame({"spx": [1.0, 1.0, 1.0]}, index=idx)
    backtest = backtest_from_weights(returns=returns, weights=weights, tcost_bps=0.0)

    assert len(backtest) == 2
    assert np.allclose(backtest["net_return"].to_numpy(), [0.20, 0.30])
    assert np.allclose(backtest["equity_net"].to_numpy(), [1.2, 1.56])
    assert np.allclose(backtest["turnover"].to_numpy(), [0.0, 0.0])


def test_backtest_rejects_missing_overlap() -> None:
    returns = pd.DataFrame({"spx": [0.10]}, index=pd.to_datetime(["2020-01-31"]))
    weights = pd.DataFrame({"spx": [1.0]}, index=pd.to_datetime(["2020-02-29"]))
    with pytest.raises(ValueError):
        backtest_from_weights(returns=returns, weights=weights)
