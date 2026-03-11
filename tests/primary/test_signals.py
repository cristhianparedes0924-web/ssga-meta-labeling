from __future__ import annotations

import numpy as np
import pandas as pd

from metalabel.primary.signals import build_primary_signal_variant1, score_to_signal


def _sample_universe(n: int = 36) -> dict[str, pd.DataFrame]:
    idx = pd.date_range("1999-01-31", periods=n, freq="ME")
    spx_price = pd.Series(np.linspace(100.0, 140.0, n), index=idx)
    bcom_price = pd.Series(np.linspace(80.0, 90.0, n), index=idx)
    ust_yield = pd.Series(np.linspace(3.5, 4.0, n), index=idx)
    corp_price = pd.Series(np.linspace(95.0, 105.0, n), index=idx)

    return {
        "spx": pd.DataFrame({"Price": spx_price, "Return": spx_price.pct_change().fillna(0.0)}, index=idx),
        "bcom": pd.DataFrame({"Price": bcom_price, "Return": bcom_price.pct_change().fillna(0.0)}, index=idx),
        "treasury_10y": pd.DataFrame({"Price": ust_yield, "Return": pd.Series(np.linspace(0.001, 0.002, n), index=idx)}, index=idx),
        "corp_bonds": pd.DataFrame({"Price": corp_price, "Return": corp_price.pct_change().fillna(0.0)}, index=idx),
    }


def test_score_to_signal_respects_boundaries() -> None:
    score = pd.Series([-0.50, -0.31, 0.0, 0.31, 0.50], index=pd.date_range("2000-01-31", periods=5, freq="ME"))
    signal = score_to_signal(score, buy_threshold=0.31, sell_threshold=-0.31)
    assert signal.tolist() == ["SELL", "HOLD", "HOLD", "HOLD", "BUY"]


def test_build_primary_signal_variant1_emits_expected_columns() -> None:
    signals = build_primary_signal_variant1(_sample_universe())
    expected = {
        "spx_trend",
        "bcom_trend",
        "credit_vs_rates",
        "risk_breadth",
        "bcom_accel",
        "yield_mom",
        "spx_trend_z",
        "bcom_trend_z",
        "credit_vs_rates_z",
        "risk_breadth_z",
        "bcom_accel_z",
        "yield_mom_z",
        "composite_score",
        "signal",
    }
    assert expected.issubset(signals.columns)
