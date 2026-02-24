from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from primary_model.research.ablations import build_variant_specs


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

    dates = pd.date_range("2014-01-31", periods=108, freq="ME")
    t = np.arange(len(dates), dtype=float)

    spx_ret = 0.006 + 0.010 * np.sin(t / 7.0)
    bcom_ret = 0.003 + 0.011 * np.sin(t / 6.0 + 0.3)
    corp_ret = 0.002 + 0.006 * np.cos(t / 8.0)

    spx_px = 100.0 * np.cumprod(1.0 + spx_ret)
    bcom_px = 90.0 * np.cumprod(1.0 + bcom_ret)
    corp_px = 105.0 * np.cumprod(1.0 + corp_ret)

    ust_yield = 2.0 + 0.7 * np.sin(t / 9.0) + 0.3 * (t / len(t))
    ust_ret_stub = np.full(len(t), 0.001)

    _write_raw_excel(raw_dir / ASSETS["spx"], dates, spx_px, spx_ret)
    _write_raw_excel(raw_dir / ASSETS["bcom"], dates, bcom_px, bcom_ret)
    _write_raw_excel(raw_dir / ASSETS["corp_bonds"], dates, corp_px, corp_ret)
    _write_raw_excel(raw_dir / ASSETS["treasury_10y"], dates, ust_yield, ust_ret_stub)


def test_build_variant_specs_shape() -> None:
    specs = build_variant_specs()
    assert len(specs) == 10
    group_counts = pd.Series([spec.group for spec in specs]).value_counts().to_dict()
    assert group_counts["baseline"] == 1
    assert group_counts["aggregation_comparison"] == 1
    assert group_counts["leave_one_out"] == 4
    assert group_counts["single_indicator"] == 4


def test_run_ablations_end_to_end(tmp_path: Path) -> None:
    run_root = tmp_path / "ablations_project"
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
        "run-ablations",
        "--config",
        "configs/experiments/ablations.yaml",
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
    assert "ablations_assessment.json" in completed.stdout

    out_dir = run_root / "reports" / "ablations"
    expected = [
        out_dir / "ablations_variant_summary.csv",
        out_dir / "ablations_leave_one_out.csv",
        out_dir / "ablations_single_indicator.csv",
        out_dir / "ablations_ranked_by_sharpe.csv",
        out_dir / "ablations_assessment.json",
        out_dir / "ablations_summary.md",
    ]
    for path in expected:
        assert path.exists(), f"Missing expected output: {path}"

    summary = pd.read_csv(out_dir / "ablations_variant_summary.csv")
    assert len(summary) == 10
    assert "baseline_dynamic_all" in summary["variant"].values
    assert "full_equal_weight_aggregation" in summary["variant"].values

    assessment = json.loads((out_dir / "ablations_assessment.json").read_text())
    assert "passed" in assessment
    assert "checks" in assessment
