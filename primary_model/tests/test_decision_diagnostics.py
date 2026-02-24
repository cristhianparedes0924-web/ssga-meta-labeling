from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from primary_model.research.decision_diagnostics import (
    build_score_zone_tables,
    compute_transition_events,
    compute_whipsaw_events,
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

    dates = pd.date_range("2013-01-31", periods=120, freq="ME")
    t = np.arange(len(dates), dtype=float)

    spx_ret = 0.005 + 0.011 * np.sin(t / 7.0)
    bcom_ret = 0.003 + 0.010 * np.sin(t / 5.5 + 0.3)
    corp_ret = 0.002 + 0.007 * np.cos(t / 8.0)

    spx_px = 100.0 * np.cumprod(1.0 + spx_ret)
    bcom_px = 90.0 * np.cumprod(1.0 + bcom_ret)
    corp_px = 108.0 * np.cumprod(1.0 + corp_ret)

    ust_yield = 2.0 + 0.6 * np.sin(t / 9.0) + 0.3 * (t / len(t))
    ust_ret_stub = np.full(len(t), 0.001)

    _write_raw_excel(raw_dir / ASSETS["spx"], dates, spx_px, spx_ret)
    _write_raw_excel(raw_dir / ASSETS["bcom"], dates, bcom_px, bcom_ret)
    _write_raw_excel(raw_dir / ASSETS["corp_bonds"], dates, corp_px, corp_ret)
    _write_raw_excel(raw_dir / ASSETS["treasury_10y"], dates, ust_yield, ust_ret_stub)


def test_compute_transition_events_basic() -> None:
    idx = pd.date_range("2020-01-31", periods=5, freq="ME")
    signal = pd.Series(["BUY", "SELL", "BUY", "BUY", "SELL"], index=idx, dtype=object)
    weights = pd.DataFrame(
        {
            "a": [1.0, 0.0, 1.0, 1.0, 0.0],
            "b": [0.0, 1.0, 0.0, 0.0, 1.0],
        },
        index=idx,
    )
    backtest = pd.DataFrame(
        {
            "net_return": [0.01, -0.02, 0.03, 0.02, -0.01],
            "turnover": [0.0, 1.0, 1.0, 0.0, 1.0],
        },
        index=idx,
    )
    next_rets = pd.DataFrame(
        {
            "a": [0.01, -0.01, 0.02, 0.02, -0.01],
            "b": [-0.01, 0.01, -0.02, -0.02, 0.01],
        },
        index=idx,
    )

    events = compute_transition_events(
        variant="v1",
        signal=signal,
        weights=weights,
        backtest=backtest,
        next_rets=next_rets,
    )
    assert len(events) == 3
    assert set(events["transition"]) == {"BUY->SELL", "SELL->BUY", "BUY->SELL"}
    assert events["switch_value"].notna().all()


def test_compute_whipsaw_events_detects_reversal() -> None:
    idx = pd.date_range("2021-01-31", periods=6, freq="ME")
    signal = pd.Series(["BUY", "SELL", "BUY", "BUY", "SELL", "BUY"], index=idx, dtype=object)
    transition_events = pd.DataFrame(
        {
            "variant": ["v1", "v1", "v1", "v1", "v1"],
            "date": idx[1:],
            "transition": ["BUY->SELL", "SELL->BUY", "BUY->BUY", "BUY->SELL", "SELL->BUY"],
            "from_signal": ["BUY", "SELL", "BUY", "BUY", "SELL"],
            "to_signal": ["SELL", "BUY", "BUY", "SELL", "BUY"],
            "switch_value": [0.01, 0.02, 0.0, -0.01, 0.03],
        }
    )
    backtest = pd.DataFrame(
        {"net_return": [0.01, -0.02, 0.01, 0.02, -0.03, 0.04]},
        index=idx,
    )

    whipsaw = compute_whipsaw_events(
        variant="v1",
        signal=signal,
        backtest=backtest,
        transition_events=transition_events,
    )
    assert not whipsaw.empty
    assert whipsaw["reversed_within_3m"].any()
    assert whipsaw["false_flip"].isin([True, False]).all()


def test_build_score_zone_tables_outputs_bins() -> None:
    idx = pd.date_range("2010-01-31", periods=120, freq="ME")
    score = pd.Series(np.linspace(-2.0, 2.0, len(idx)), index=idx)
    target = score * 0.01 + pd.Series(np.random.default_rng(12).normal(0.0, 0.01, len(idx)), index=idx)

    bins_df, zones_df = build_score_zone_tables(
        variant="dynamic_all",
        score=score,
        target_next=target,
        bins=10,
        buy_threshold=0.0001,
        sell_threshold=-0.0001,
        min_pairs=36,
    )
    assert len(bins_df) >= 8
    assert {"BUY_ZONE", "HOLD_ZONE", "SELL_ZONE"}.issuperset(set(zones_df["zone"]))


def test_run_decision_diagnostics_end_to_end(tmp_path: Path) -> None:
    run_root = tmp_path / "decision_diagnostics_project"
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
        "run-decision-diagnostics",
        "--config",
        "configs/experiments/decision_diagnostics.yaml",
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
    assert "decision_assessment.json" in completed.stdout

    out_dir = run_root / "reports" / "decision_diagnostics"
    expected = [
        out_dir / "decision_transition_events.csv",
        out_dir / "decision_transition_summary.csv",
        out_dir / "decision_whipsaw_events.csv",
        out_dir / "decision_whipsaw_summary.csv",
        out_dir / "decision_dwell_runs.csv",
        out_dir / "decision_dwell_summary.csv",
        out_dir / "decision_dwell_duration_buckets.csv",
        out_dir / "decision_score_zone_bins.csv",
        out_dir / "decision_threshold_zone_summary.csv",
        out_dir / "decision_assessment.json",
        out_dir / "decision_summary.md",
    ]
    for path in expected:
        assert path.exists(), f"Missing expected output: {path}"

    assessment = json.loads((out_dir / "decision_assessment.json").read_text(encoding="utf-8"))
    assert "passed" in assessment
    assert "checks" in assessment
