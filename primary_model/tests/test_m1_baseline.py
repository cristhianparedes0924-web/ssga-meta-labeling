from __future__ import annotations

import json
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

    dates = pd.date_range("2016-01-31", periods=72, freq="ME")
    t = np.arange(len(dates), dtype=float)

    spx_ret = 0.005 + 0.010 * np.sin(t / 6.0)
    bcom_ret = 0.003 + 0.011 * np.sin(t / 5.0 + 0.4)
    corp_ret = 0.002 + 0.007 * np.cos(t / 7.0)

    spx_px = 100.0 * np.cumprod(1.0 + spx_ret)
    bcom_px = 90.0 * np.cumprod(1.0 + bcom_ret)
    corp_px = 105.0 * np.cumprod(1.0 + corp_ret)

    ust_yield = 2.0 + 0.6 * np.sin(t / 9.0) + 0.3 * (t / len(t))
    ust_ret_stub = np.full(len(dates), 0.001)

    _write_raw_excel(raw_dir / ASSETS["spx"], dates, spx_px, spx_ret)
    _write_raw_excel(raw_dir / ASSETS["bcom"], dates, bcom_px, bcom_ret)
    _write_raw_excel(raw_dir / ASSETS["corp_bonds"], dates, corp_px, corp_ret)
    _write_raw_excel(raw_dir / ASSETS["treasury_10y"], dates, ust_yield, ust_ret_stub)


def test_run_m1_baseline_snapshot(tmp_path: Path) -> None:
    run_root = tmp_path / "m1_baseline_project"
    _build_mock_raw_data(run_root)

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    cmd = [
        sys.executable,
        "cli.py",
        "run-m1-baseline",
        "--config",
        "configs/experiments/m1_canonical_v1_1.yaml",
        "--root",
        str(run_root),
    ]
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "m1_baseline_snapshot.json" in completed.stdout

    snapshot_json = run_root / "reports" / "reproducibility" / "m1_baseline_snapshot.json"
    snapshot_md = run_root / "reports" / "reproducibility" / "m1_baseline_snapshot.md"

    assert snapshot_json.exists()
    assert snapshot_md.exists()

    payload = json.loads(snapshot_json.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["spec"]["id"] == "PrimaryV1.1-baseline"
    assert payload["spec"]["version"] == "1.1"
    assert payload["determinism"]["enabled"] is True
    assert payload["determinism"]["runs"] == 2
    assert payload["determinism"]["deterministic"] is True
    assert "ann_return" in payload["key_metrics"]["performance"]
    assert len(payload["artifact_hashes"]) >= 6
