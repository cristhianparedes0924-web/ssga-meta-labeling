from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.data_loader import DATA_FILES, EXPECTED_OUTPUT_COLUMNS, load_data


def _write_excel(path: Path, skiprows: int, frame: pd.DataFrame) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, startrow=skiprows)


class DataLoaderTests(unittest.TestCase):
    def _make_valid_dataset(self, data_dir: Path) -> pd.DatetimeIndex:
        dates = pd.date_range("2020-01-31", periods=6, freq="ME")
        for series_index, (series_name, (filename, skiprows)) in enumerate(DATA_FILES.items()):
            values = [100.0 + series_index + idx for idx in range(len(dates))]
            frame = pd.DataFrame(
                {
                    "Date": dates,
                    "PX_LAST": values,
                    "CHG_PCT_1D": [0.0] * len(dates),
                }
            )
            _write_excel(data_dir / filename, skiprows=skiprows, frame=frame)
        return dates

    def test_load_data_success(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            self._make_valid_dataset(data_dir)

            result = load_data(data_dir)

            self.assertEqual(list(result.columns), EXPECTED_OUTPUT_COLUMNS)
            self.assertEqual(len(result), 6)
            self.assertFalse(result.isna().any().any())

    def test_load_data_missing_file_raises(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            self._make_valid_dataset(data_dir)
            missing_filename = DATA_FILES["SPX"][0]
            (data_dir / missing_filename).unlink()

            with self.assertRaises(FileNotFoundError):
                load_data(data_dir)

    def test_load_data_missing_required_column_raises(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            dates = self._make_valid_dataset(data_dir)

            filename, skiprows = DATA_FILES["Treasury10Y"]
            bad_frame = pd.DataFrame({"Date": dates, "PX_CLOSE": [4.0] * len(dates)})
            _write_excel(data_dir / filename, skiprows=skiprows, frame=bad_frame)

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_data(data_dir)

    def test_load_data_duplicate_dates_raises(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            self._make_valid_dataset(data_dir)

            filename, skiprows = DATA_FILES["BCOM"]
            duplicate_dates = pd.to_datetime(
                ["2020-01-31", "2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30", "2020-05-31"]
            )
            duplicate_frame = pd.DataFrame(
                {
                    "Date": duplicate_dates,
                    "PX_LAST": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                }
            )
            _write_excel(data_dir / filename, skiprows=skiprows, frame=duplicate_frame)

            with self.assertRaisesRegex(ValueError, "duplicate dates"):
                load_data(data_dir)

    def test_load_data_misaligned_dates_raises(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            dates = self._make_valid_dataset(data_dir)

            filename, skiprows = DATA_FILES["IG_Corp"]
            shifted_frame = pd.DataFrame(
                {
                    "Date": dates + pd.offsets.MonthEnd(1),
                    "PX_LAST": [120.0 + idx for idx in range(len(dates))],
                }
            )
            _write_excel(data_dir / filename, skiprows=skiprows, frame=shifted_frame)

            with self.assertRaisesRegex(ValueError, "Merged data contains missing values"):
                load_data(data_dir)


if __name__ == "__main__":
    unittest.main()
