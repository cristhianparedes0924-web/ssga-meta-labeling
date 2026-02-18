from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from primary_model.treasury import compute_bond_total_return_from_yield  # noqa: E402


def test_treasury_total_return_formula() -> None:
    index = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    yields = pd.Series([5.0, 5.2, 5.1], index=index)

    out = compute_bond_total_return_from_yield(
        yields, duration=8.5, periods_per_year=12, include_carry=True
    )

    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == pytest.approx(-0.01283333333333334)
    assert out.iloc[2] == pytest.approx(0.01283333333333333)
