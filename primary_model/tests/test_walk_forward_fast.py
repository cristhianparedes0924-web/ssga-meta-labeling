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


def _write_raw_excel(
    path: Path, dates: pd.DatetimeIndex, px_last: np.ndarray, ret: np.ndarray
) -> None:
    rows: list[list[object]] = [["meta", "", ""], ["Date", "PX_LAST", "CHG_PCT_1D"]]
    for dt, price, r in zip(dates, px_last, ret, strict=True):
        rows.append([dt.date().isoformat(), float(price), f"{r * 100:.4f}%"])
    pd.DataFrame(rows).to_excel(path, index=False, header=False, engine="openpyxl")


def _build_mock_raw_data(root: Path) -> None:
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2012-01-31", periods=96, freq="ME")
    t = np.arange(len(dates), dtype=float)

    spx_ret = 0.006 + 0.010 * np.sin(t / 8.0)
    bcom_ret = 0.003 + 0.011 * np.sin(t / 7.0 + 0.2)
    corp_ret = 0.002 + 0.006 * np.cos(t / 9.0)

    spx_px = 100.0 * np.cumprod(1.0 + spx_ret)
    bcom_px = 90.0 * np.cumprod(1.0 + bcom_ret)
    corp_px = 105.0 * np.cumprod(1.0 + corp_ret)

    ust_yield = 2.0 + 0.6 * np.sin(t / 10.0) + 0.25 * (t / len(t))
    ust_stub = np.full(len(dates), 0.001)

    _write_raw_excel(raw_dir / ASSETS["spx"], dates, spx_px, spx_ret)
    _write_raw_excel(raw_dir / ASSETS["bcom"], dates, bcom_px, bcom_ret)
    _write_raw_excel(raw_dir / ASSETS["corp_bonds"], dates, corp_px, corp_ret)
    _write_raw_excel(raw_dir / ASSETS["treasury_10y"], dates, ust_yield, ust_stub)


def test_run_walk_forward_cached_engine_end_to_end(tmp_path: Path) -> None:
    run_root = tmp_path / "walk_forward_cached_project"
    _build_mock_raw_data(run_root)

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    prep_cmd = [sys.executable, "cli.py", "prepare-data", "--root", str(run_root)]
    subprocess.run(
        prep_cmd,
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    cmd = [
        sys.executable,
        "cli.py",
        "run-walk-forward",
        "--config",
        "configs/experiments/walk_forward_fast.yaml",
        "--root",
        str(run_root),
        "--engine-mode",
        "cached_causal",
        "--min-train-periods",
        "36",
    ]
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "walk_forward_summary.csv" in completed.stdout

    out_dir = run_root / "reports" / "walk_forward"
    assert (out_dir / "walk_forward_backtest.csv").exists()
    assert (out_dir / "walk_forward_summary.csv").exists()
    protocol = (out_dir / "walk_forward_protocol.md").read_text(encoding="utf-8")
    assert "cached_causal" in protocol

