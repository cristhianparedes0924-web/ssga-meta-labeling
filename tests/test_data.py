from __future__ import annotations

import pandas as pd
import pytest

from metalabel.data import CANONICAL_COLUMNS, find_header_row, load_clean_asset_csv, validate_canonical_dataframe


def test_find_header_row_detects_embedded_header() -> None:
    raw = pd.DataFrame(
        [
            ["metadata", None, None],
            ["DATE", "PX_LAST", "CHG_PCT_1D"],
            ["2020-01-31", "100", "1.0%"],
        ]
    )
    assert find_header_row(raw) == 1


def test_validate_canonical_dataframe_rejects_wrong_columns() -> None:
    frame = pd.DataFrame({"Date": pd.to_datetime(["2020-01-31"]), "Price": [100.0], "Bad": [0.01]})
    with pytest.raises(ValueError):
        validate_canonical_dataframe(frame)


def test_load_clean_asset_csv_infers_return_when_missing(tmp_path) -> None:
    path = tmp_path / "asset.csv"
    pd.DataFrame(
        {
            "Date": ["2020-01-31", "2020-02-29", "2020-03-31"],
            "Price": [100.0, 110.0, 121.0],
        }
    ).to_csv(path, index=False)

    loaded = load_clean_asset_csv(path)
    assert list(loaded.columns) == CANONICAL_COLUMNS[1:]
    assert loaded["Return"].iloc[1] == pytest.approx(0.10)
