from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from primary_model.research.signal_validation import (
    build_forward_relative_target,
    build_monotonicity_table,
    compute_predictive_statistics,
)


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

    dates = pd.date_range("2015-01-31", periods=96, freq="ME")
    t = np.arange(len(dates), dtype=float)

    spx_ret = 0.004 + 0.010 * np.sin(t / 8.0)
    bcom_ret = 0.002 + 0.009 * np.sin(t / 7.0 + 0.3)
    corp_ret = 0.003 + 0.007 * np.cos(t / 6.0)
    spx_px = 100.0 * np.cumprod(1.0 + spx_ret)
    bcom_px = 90.0 * np.cumprod(1.0 + bcom_ret)
    corp_px = 105.0 * np.cumprod(1.0 + corp_ret)

    ust_yield = 2.0 + 0.6 * np.sin(t / 9.0) + 0.2 * (t / len(t))
    ust_stub = np.full(len(t), 0.001)

    _write_raw_excel(raw_dir / ASSETS["spx"], dates, spx_px, spx_ret)
    _write_raw_excel(raw_dir / ASSETS["bcom"], dates, bcom_px, bcom_ret)
    _write_raw_excel(raw_dir / ASSETS["corp_bonds"], dates, corp_px, corp_ret)
    _write_raw_excel(raw_dir / ASSETS["treasury_10y"], dates, ust_yield, ust_stub)


def test_build_forward_relative_target_shape() -> None:
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    returns = pd.DataFrame(
        {
            "spx": [0.01, 0.02, 0.03, 0.04, 0.05],
            "bcom": [0.00, 0.01, 0.02, 0.03, 0.04],
            "corp_bonds": [0.01, 0.01, 0.01, 0.01, 0.01],
            "treasury_10y": [0.001, 0.001, 0.001, 0.001, 0.001],
        },
        index=dates,
    )
    out = build_forward_relative_target(returns)
    assert out.index.equals(returns.index)
    assert np.isnan(out.iloc[-1])
    assert out.notna().sum() == len(returns) - 1


def test_compute_predictive_statistics_detects_positive_association() -> None:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2010-01-31", periods=240, freq="ME")
    factor = pd.Series(rng.normal(size=len(idx)), index=idx)
    noise = pd.Series(rng.normal(scale=0.2, size=len(idx)), index=idx)
    target = 0.01 * factor + noise

    stats = compute_predictive_statistics(
        factor=factor,
        target_next=target,
        bootstrap_samples=500,
        bootstrap_seed=11,
        min_pairs=50,
    )
    assert stats["n_obs"] == len(idx)
    assert stats["spearman_corr"] > 0.0
    assert stats["mean_spread_top_minus_bottom"] > 0.0


def test_build_monotonicity_table_top_bin_above_bottom() -> None:
    rng = np.random.default_rng(13)
    idx = pd.date_range("2012-01-31", periods=180, freq="ME")
    factor = pd.Series(np.linspace(-2.0, 2.0, len(idx)), index=idx)
    target = pd.Series(np.linspace(-0.03, 0.03, len(idx)), index=idx) + pd.Series(
        rng.normal(scale=0.003, size=len(idx)), index=idx
    )

    table, summary = build_monotonicity_table(factor=factor, target_next=target, bins=10)
    assert not table.empty
    assert summary["bins_realized"] >= 8
    assert summary["top_bottom_gap"] > 0.0
    assert summary["bin_rank_spearman"] > 0.0
    assert summary["monotonic_up"] is True


def test_run_signal_validation_end_to_end(tmp_path: Path) -> None:
    run_root = tmp_path / "signal_validation_project"
    _build_mock_raw_data(run_root)

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    prep_cmd = [
        sys.executable,
        "cli.py",
        "prepare-data",
        "--root",
        str(run_root),
    ]
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
        "run-signal-validation",
        "--config",
        "configs/experiments/signal_validation.yaml",
        "--root",
        str(run_root),
        "--bootstrap-samples",
        "300",
    ]
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = completed.stdout
    assert "signal_validation_assessment.json" in stdout

    out_dir = run_root / "reports" / "signal_validation"
    expected = [
        out_dir / "signal_validation_fullsample.csv",
        out_dir / "signal_validation_subperiods.csv",
        out_dir / "signal_validation_monotonicity_summary.csv",
        out_dir / "signal_validation_monotonicity_bins.csv",
        out_dir / "signal_validation_assessment.json",
        out_dir / "signal_validation_summary.md",
    ]
    for path in expected:
        assert path.exists(), f"Missing expected output: {path}"

    assessment = json.loads((out_dir / "signal_validation_assessment.json").read_text())
    assert "passed" in assessment
    assert "checks" in assessment
