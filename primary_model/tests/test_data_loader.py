from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from primary_model.data import (  # noqa: E402
    load_clean_asset_csv,
    load_universe,
    universe_returns_matrix,
)


def test_data_loader_derives_return(tmp_path: Path) -> None:
    asset_path = tmp_path / "spx.csv"
    pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-01", "2024-01-03"],
            "Price": [110.0, 100.0, 121.0],
        }
    ).to_csv(asset_path, index=False)

    loaded = load_clean_asset_csv(asset_path)

    assert loaded.index.name == "Date"
    assert loaded.index.is_monotonic_increasing
    assert list(loaded.columns) == ["Price", "Return"]
    assert np.isnan(loaded.iloc[0]["Return"])
    assert loaded.iloc[1]["Return"] == pytest.approx(0.10)


def test_universe_returns_alignment(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "Price": [100.0, 110.0, 121.0],
        }
    ).to_csv(tmp_path / "spx.csv", index=False)

    pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "Price": [200.0, 220.0, 242.0],
        }
    ).to_csv(tmp_path / "bcom.csv", index=False)

    assets = ["spx", "bcom"]
    universe = load_universe(tmp_path, assets)
    returns = universe_returns_matrix(universe)

    expected_index = pd.DatetimeIndex([pd.Timestamp("2024-01-03")], name="Date")
    assert returns.index.equals(expected_index)
    assert list(returns.columns) == assets
