from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from primary_model.research.m1_readiness import M1ReadinessRunConfig, evaluate_m1_readiness


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_stage_artifacts(root: Path) -> None:
    signal_dir = root / "reports" / "signal_validation"
    ablation_dir = root / "reports" / "ablations"
    decision_dir = root / "reports" / "decision_diagnostics"

    signal_dir.mkdir(parents=True, exist_ok=True)
    ablation_dir.mkdir(parents=True, exist_ok=True)
    decision_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "factor": "composite_score",
                "n_obs": 120,
                "spearman_corr": 0.08,
                "spearman_pvalue_approx": 0.12,
                "spread_pvalue_approx": 0.18,
            }
        ]
    ).to_csv(signal_dir / "signal_validation_fullsample.csv", index=False)
    pd.DataFrame(
        [{"factor": "composite_score", "top_bottom_gap": 0.01}]
    ).to_csv(signal_dir / "signal_validation_monotonicity_summary.csv", index=False)
    pd.DataFrame(
        [
            {"factor": "composite_score", "subperiod": "1996-2003", "spearman_corr": 0.03},
            {"factor": "composite_score", "subperiod": "2004-2012", "spearman_corr": 0.04},
            {"factor": "composite_score", "subperiod": "2013-2019", "spearman_corr": -0.01},
            {"factor": "composite_score", "subperiod": "2020-2025", "spearman_corr": 0.02},
        ]
    ).to_csv(signal_dir / "signal_validation_subperiods.csv", index=False)
    (signal_dir / "signal_validation_assessment.json").write_text(
        json.dumps({"passed": True, "checks": {}, "details": {}}),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "variant": "baseline_dynamic_all",
                "group": "baseline",
                "sharpe": 0.80,
            },
            {
                "variant": "full_equal_weight_aggregation",
                "group": "aggregation_comparison",
                "sharpe": 0.82,
            },
            {
                "variant": "leave_one_out_spx_trend",
                "group": "leave_one_out",
                "sharpe": 0.75,
            },
            {
                "variant": "leave_one_out_bcom_trend",
                "group": "leave_one_out",
                "sharpe": 0.70,
            },
            {
                "variant": "leave_one_out_credit_vs_rates",
                "group": "leave_one_out",
                "sharpe": 0.76,
            },
            {
                "variant": "leave_one_out_risk_breadth",
                "group": "leave_one_out",
                "sharpe": 0.74,
            },
            {
                "variant": "single_indicator_spx_trend",
                "group": "single_indicator",
                "sharpe": 0.84,
            },
            {
                "variant": "single_indicator_bcom_trend",
                "group": "single_indicator",
                "sharpe": 0.60,
            },
            {
                "variant": "single_indicator_credit_vs_rates",
                "group": "single_indicator",
                "sharpe": 0.65,
            },
            {
                "variant": "single_indicator_risk_breadth",
                "group": "single_indicator",
                "sharpe": 0.62,
            },
        ]
    ).to_csv(ablation_dir / "ablations_variant_summary.csv", index=False)
    (ablation_dir / "ablations_assessment.json").write_text(
        json.dumps(
            {
                "passed": True,
                "checks": {},
                "details": {"complete_indicator_coverage": True},
            }
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "variant": "dynamic_all",
                "transition": "BUY->SELL",
                "avg_switch_value": 0.0010,
                "total_switch_value": 0.05,
            },
            {
                "variant": "dynamic_all",
                "transition": "SELL->BUY",
                "avg_switch_value": 0.0008,
                "total_switch_value": 0.03,
            },
        ]
    ).to_csv(decision_dir / "decision_transition_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "variant": "dynamic_all",
                "false_flip_rate": 0.40,
                "false_flip_total_switch_value": -0.02,
            }
        ]
    ).to_csv(decision_dir / "decision_whipsaw_summary.csv", index=False)
    (decision_dir / "decision_assessment.json").write_text(
        json.dumps({"passed": True, "checks": {}, "details": {}}),
        encoding="utf-8",
    )


def _build_run_config(tmp_path: Path) -> M1ReadinessRunConfig:
    root = tmp_path / "root"
    signal_dir = root / "reports" / "signal_validation"
    ablation_dir = root / "reports" / "ablations"
    decision_dir = root / "reports" / "decision_diagnostics"
    return M1ReadinessRunConfig(
        root=root,
        out_dir=root / "reports" / "readiness",
        criteria_version="m1_readiness_v1",
        decision_variant="dynamic_all",
        min_composite_spearman=0.0,
        max_composite_pvalue=0.25,
        min_monotonic_top_bottom_gap=0.0,
        min_positive_subperiod_fraction=0.75,
        max_single_indicator_dominance_ratio=1.15,
        max_loo_sharpe_drop=0.25,
        min_transition_avg_switch_value=0.0,
        max_false_flip_rate=0.60,
        min_net_transition_after_whipsaw=0.0,
        signal_assessment_path=signal_dir / "signal_validation_assessment.json",
        signal_fullsample_path=signal_dir / "signal_validation_fullsample.csv",
        signal_monotonicity_path=signal_dir / "signal_validation_monotonicity_summary.csv",
        signal_subperiods_path=signal_dir / "signal_validation_subperiods.csv",
        ablation_assessment_path=ablation_dir / "ablations_assessment.json",
        ablation_summary_path=ablation_dir / "ablations_variant_summary.csv",
        decision_assessment_path=decision_dir / "decision_assessment.json",
        decision_transition_summary_path=decision_dir / "decision_transition_summary.csv",
        decision_whipsaw_summary_path=decision_dir / "decision_whipsaw_summary.csv",
    )


def test_evaluate_m1_readiness_flags_negative_net_transition(tmp_path: Path) -> None:
    run_cfg = _build_run_config(tmp_path)

    signal_assessment = {"passed": True}
    signal_fullsample = pd.DataFrame(
        [
            {
                "factor": "composite_score",
                "spearman_corr": 0.05,
                "spearman_pvalue_approx": 0.10,
                "spread_pvalue_approx": 0.15,
            }
        ]
    )
    signal_monotonicity = pd.DataFrame(
        [{"factor": "composite_score", "top_bottom_gap": 0.01}]
    )
    signal_subperiods = pd.DataFrame(
        [
            {"factor": "composite_score", "spearman_corr": 0.01},
            {"factor": "composite_score", "spearman_corr": 0.02},
            {"factor": "composite_score", "spearman_corr": -0.01},
            {"factor": "composite_score", "spearman_corr": 0.03},
        ]
    )

    ablation_assessment = {"passed": True, "details": {"complete_indicator_coverage": True}}
    ablation_summary = pd.DataFrame(
        [
            {"variant": "baseline_dynamic_all", "group": "baseline", "sharpe": 0.8},
            {"variant": "leave_one_out_spx_trend", "group": "leave_one_out", "sharpe": 0.7},
            {"variant": "single_indicator_spx_trend", "group": "single_indicator", "sharpe": 0.85},
        ]
    )

    decision_assessment = {"passed": True}
    transition_summary = pd.DataFrame(
        [
            {
                "variant": "dynamic_all",
                "transition": "BUY->SELL",
                "avg_switch_value": 0.001,
                "total_switch_value": -0.02,
            },
            {
                "variant": "dynamic_all",
                "transition": "SELL->BUY",
                "avg_switch_value": 0.001,
                "total_switch_value": -0.01,
            },
        ]
    )
    whipsaw_summary = pd.DataFrame(
        [
            {
                "variant": "dynamic_all",
                "false_flip_rate": 0.50,
                "false_flip_total_switch_value": -0.05,
            }
        ]
    )

    report = evaluate_m1_readiness(
        run_config=run_cfg,
        signal_assessment=signal_assessment,
        signal_fullsample=signal_fullsample,
        signal_monotonicity=signal_monotonicity,
        signal_subperiods=signal_subperiods,
        ablation_assessment=ablation_assessment,
        ablation_summary=ablation_summary,
        decision_assessment=decision_assessment,
        decision_transition_summary=transition_summary,
        decision_whipsaw_summary=whipsaw_summary,
    )
    assert report["passed"] is False
    assert "decision_transitions_add_value_after_whipsaw_costs" in report["failed_checks"]


def test_run_m1_readiness_end_to_end(tmp_path: Path) -> None:
    run_root = tmp_path / "m1_readiness_project"
    _write_stage_artifacts(run_root)

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    cmd = [
        sys.executable,
        "cli.py",
        "run-m1-readiness",
        "--config",
        "configs/experiments/m1_readiness.yaml",
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
    assert "m1_readiness_checklist.json" in completed.stdout

    out_dir = run_root / "reports" / "readiness"
    expected = [
        out_dir / "m1_readiness_checklist.json",
        out_dir / "m1_readiness_summary.md",
    ]
    for path in expected:
        assert path.exists(), f"Missing expected output: {path}"

    payload = json.loads((out_dir / "m1_readiness_checklist.json").read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert "readiness_passed" in payload
    assert "criteria_version" in payload
    assert len(payload["checklist"]) == 4
