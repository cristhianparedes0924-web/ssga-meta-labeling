from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


ASSETS = {
    "spx": "spx.xlsx",
    "bcom": "bcom.xlsx",
    "treasury_10y": "treasury_10y.xlsx",
    "corp_bonds": "corp_bonds.xlsx",
}


def _write_raw_excel(path: Path, dates: pd.DatetimeIndex, px_last: np.ndarray, ret: np.ndarray) -> None:
    rows: list[list[object]] = [["meta", "", ""], ["Date", "PX_LAST", "CHG_PCT_1D"]]
    for dt, price, r in zip(dates, px_last, ret, strict=True):
        rows.append([dt.date().isoformat(), float(price), f"{r * 100:.4f}%"])

    frame = pd.DataFrame(rows)
    frame.to_excel(path, index=False, header=False, engine="openpyxl")


def _build_mock_raw_data(root: Path) -> None:
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2018-01-31", periods=60, freq="ME")
    t = np.arange(len(dates), dtype=float)

    spx_ret = 0.006 + 0.01 * np.sin(t / 6.0)
    bcom_ret = 0.003 + 0.012 * np.sin(t / 5.0 + 0.4)
    corp_ret = 0.002 + 0.006 * np.cos(t / 7.0)

    spx_px = 100.0 * np.cumprod(1.0 + spx_ret)
    bcom_px = 90.0 * np.cumprod(1.0 + bcom_ret)
    corp_px = 105.0 * np.cumprod(1.0 + corp_ret)

    # Treasury is yield-like in the unified model; returns are replaced downstream.
    ust_yield = 2.0 + 0.8 * np.sin(t / 9.0) + 0.4 * (t / len(t))
    ust_ret_stub = np.full(len(dates), 0.001)

    _write_raw_excel(raw_dir / ASSETS["spx"], dates, spx_px, spx_ret)
    _write_raw_excel(raw_dir / ASSETS["bcom"], dates, bcom_px, bcom_ret)
    _write_raw_excel(raw_dir / ASSETS["corp_bonds"], dates, corp_px, corp_ret)
    _write_raw_excel(raw_dir / ASSETS["treasury_10y"], dates, ust_yield, ust_ret_stub)


def test_run_all_end_to_end(tmp_path: Path) -> None:
    run_root = tmp_path / "sample_project"
    _build_mock_raw_data(run_root)

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    cmd = [sys.executable, "cli.py", "run-all", "--root", str(run_root)]
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = completed.stdout
    assert "Preparing clean data files..." in stdout
    assert "Benchmark performance table (net):" in stdout

    expected_files = [
        run_root / "data" / "clean" / "spx.csv",
        run_root / "data" / "clean" / "bcom.csv",
        run_root / "data" / "clean" / "treasury_10y.csv",
        run_root / "data" / "clean" / "corp_bonds.csv",
        run_root / "reports" / "data_qc.html",
        run_root / "reports" / "primary_v1_backtest.csv",
        run_root / "reports" / "primary_v1_summary.csv",
        run_root / "reports" / "benchmarks_summary.csv",
        run_root / "reports" / "benchmarks_summary.html",
        run_root / "reports" / "primary_v1_signal.csv",
        run_root / "reports" / "primary_v1_weights.csv",
        run_root / "reports" / "assets" / "equity_curves.png",
        run_root / "reports" / "assets" / "drawdowns.png",
        run_root / "reports" / "assets" / "rolling_sharpe.png",
    ]

    for file_path in expected_files:
        assert file_path.exists(), f"Missing expected output: {file_path}"

    summary = pd.read_csv(run_root / "reports" / "benchmarks_summary.csv")
    assert not summary.empty
