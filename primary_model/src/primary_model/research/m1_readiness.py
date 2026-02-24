"""Stage 5 M1 readiness gate built from Stage 2-4 research artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from primary_model.utils.artifacts import write_markdown_protocol


@dataclass(frozen=True)
class M1ReadinessRunConfig:
    root: Path
    out_dir: Path
    criteria_version: str
    decision_variant: str
    min_composite_spearman: float
    max_composite_pvalue: float
    min_monotonic_top_bottom_gap: float
    min_positive_subperiod_fraction: float
    max_single_indicator_dominance_ratio: float
    max_loo_sharpe_drop: float
    min_transition_avg_switch_value: float
    max_false_flip_rate: float
    min_net_transition_after_whipsaw: float
    signal_assessment_path: Path
    signal_fullsample_path: Path
    signal_monotonicity_path: Path
    signal_subperiods_path: Path
    ablation_assessment_path: Path
    ablation_summary_path: Path
    decision_assessment_path: Path
    decision_transition_summary_path: Path
    decision_whipsaw_summary_path: Path


def _resolve_path(root: Path, value: Any, default_rel: str) -> Path:
    candidate = Path(value) if value is not None else Path(default_rel)
    if candidate.is_absolute():
        return candidate.resolve()
    return (root / candidate).resolve()


def _must_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required readiness input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _must_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required readiness input: {path}")
    return pd.read_csv(path)


def _resolve_run_config(
    config: Mapping[str, Any],
    cli_args: Any | None = None,
) -> M1ReadinessRunConfig:
    paths_cfg = config.get("paths", {})
    run_cfg = config.get("run", {})

    root_override = getattr(cli_args, "root", None) if cli_args else None
    out_dir_override = getattr(cli_args, "out_dir", None) if cli_args else None

    root = Path(root_override or paths_cfg.get("root", "artifacts")).resolve()
    out_dir = _resolve_path(root, out_dir_override or run_cfg.get("out_dir"), "reports/readiness")

    signal_dir = _resolve_path(
        root,
        run_cfg.get("signal_validation_dir"),
        "reports/signal_validation",
    )
    ablation_dir = _resolve_path(
        root,
        run_cfg.get("ablations_dir"),
        "reports/ablations",
    )
    decision_dir = _resolve_path(
        root,
        run_cfg.get("decision_diagnostics_dir"),
        "reports/decision_diagnostics",
    )

    return M1ReadinessRunConfig(
        root=root,
        out_dir=out_dir,
        criteria_version=str(run_cfg.get("criteria_version", "m1_readiness_v1")),
        decision_variant=str(run_cfg.get("decision_variant", "dynamic_all")),
        min_composite_spearman=float(run_cfg.get("min_composite_spearman", 0.0)),
        max_composite_pvalue=float(run_cfg.get("max_composite_pvalue", 0.25)),
        min_monotonic_top_bottom_gap=float(run_cfg.get("min_monotonic_top_bottom_gap", 0.0)),
        min_positive_subperiod_fraction=float(run_cfg.get("min_positive_subperiod_fraction", 0.75)),
        max_single_indicator_dominance_ratio=float(
            run_cfg.get("max_single_indicator_dominance_ratio", 1.15)
        ),
        max_loo_sharpe_drop=float(run_cfg.get("max_loo_sharpe_drop", 0.25)),
        min_transition_avg_switch_value=float(
            run_cfg.get("min_transition_avg_switch_value", 0.0)
        ),
        max_false_flip_rate=float(run_cfg.get("max_false_flip_rate", 0.60)),
        min_net_transition_after_whipsaw=float(
            run_cfg.get("min_net_transition_after_whipsaw", 0.0)
        ),
        signal_assessment_path=signal_dir
        / str(run_cfg.get("signal_assessment_file", "signal_validation_assessment.json")),
        signal_fullsample_path=signal_dir
        / str(run_cfg.get("signal_fullsample_file", "signal_validation_fullsample.csv")),
        signal_monotonicity_path=signal_dir
        / str(
            run_cfg.get(
                "signal_monotonicity_file",
                "signal_validation_monotonicity_summary.csv",
            )
        ),
        signal_subperiods_path=signal_dir
        / str(run_cfg.get("signal_subperiods_file", "signal_validation_subperiods.csv")),
        ablation_assessment_path=ablation_dir
        / str(run_cfg.get("ablation_assessment_file", "ablations_assessment.json")),
        ablation_summary_path=ablation_dir
        / str(run_cfg.get("ablation_summary_file", "ablations_variant_summary.csv")),
        decision_assessment_path=decision_dir
        / str(run_cfg.get("decision_assessment_file", "decision_assessment.json")),
        decision_transition_summary_path=decision_dir
        / str(run_cfg.get("decision_transition_file", "decision_transition_summary.csv")),
        decision_whipsaw_summary_path=decision_dir
        / str(run_cfg.get("decision_whipsaw_file", "decision_whipsaw_summary.csv")),
    )


def _check_entry(
    *,
    check_id: str,
    passed: bool,
    actual: Any,
    threshold: str,
    remediation: str,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "passed": bool(passed),
        "actual": actual,
        "threshold": threshold,
        "remediation": remediation,
    }


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def evaluate_m1_readiness(
    *,
    run_config: M1ReadinessRunConfig,
    signal_assessment: Mapping[str, Any],
    signal_fullsample: pd.DataFrame,
    signal_monotonicity: pd.DataFrame,
    signal_subperiods: pd.DataFrame,
    ablation_assessment: Mapping[str, Any],
    ablation_summary: pd.DataFrame,
    decision_assessment: Mapping[str, Any],
    decision_transition_summary: pd.DataFrame,
    decision_whipsaw_summary: pd.DataFrame,
) -> dict[str, Any]:
    composite = signal_fullsample.loc[signal_fullsample["factor"] == "composite_score"]
    if composite.empty:
        raise ValueError("signal_validation_fullsample.csv missing `composite_score` row.")
    composite_row = composite.iloc[0]

    mono = signal_monotonicity.loc[signal_monotonicity["factor"] == "composite_score"]
    if mono.empty:
        raise ValueError("signal_validation_monotonicity_summary.csv missing `composite_score` row.")
    mono_row = mono.iloc[0]

    sub = signal_subperiods.loc[signal_subperiods["factor"] == "composite_score"].copy()
    if sub.empty:
        raise ValueError("signal_validation_subperiods.csv missing composite_score rows.")

    composite_spearman = _safe_float(composite_row.get("spearman_corr"))
    spearman_p = _safe_float(composite_row.get("spearman_pvalue_approx"))
    spread_p = _safe_float(composite_row.get("spread_pvalue_approx"))
    top_bottom_gap = _safe_float(mono_row.get("top_bottom_gap"))
    positive_subperiods = int((pd.to_numeric(sub["spearman_corr"], errors="coerce") > 0.0).sum())
    total_subperiods = int(len(sub))
    positive_fraction = (
        float(positive_subperiods / total_subperiods)
        if total_subperiods > 0
        else float("nan")
    )

    pvalue_candidates = [value for value in (spearman_p, spread_p) if not math.isnan(value)]
    pvalue_min = min(pvalue_candidates) if pvalue_candidates else float("nan")

    stage2_pass = _is_true(signal_assessment.get("passed"))
    signal_direction_ok = composite_spearman > run_config.min_composite_spearman
    signal_significance_ok = (
        not math.isnan(pvalue_min) and pvalue_min <= run_config.max_composite_pvalue
    )
    signal_monotonic_ok = top_bottom_gap > run_config.min_monotonic_top_bottom_gap
    subperiod_stability_ok = (
        not math.isnan(positive_fraction)
        and positive_fraction >= run_config.min_positive_subperiod_fraction
    )
    signal_validity_ok = (
        stage2_pass
        and signal_direction_ok
        and signal_significance_ok
        and signal_monotonic_ok
    )

    baseline = ablation_summary.loc[ablation_summary["variant"] == "baseline_dynamic_all"]
    if baseline.empty:
        raise ValueError("ablations_variant_summary.csv missing `baseline_dynamic_all` row.")
    baseline_sharpe = _safe_float(baseline.iloc[0].get("sharpe"))

    single = ablation_summary.loc[ablation_summary["group"] == "single_indicator"].copy()
    loo = ablation_summary.loc[ablation_summary["group"] == "leave_one_out"].copy()
    if single.empty or loo.empty:
        raise ValueError(
            "ablations_variant_summary.csv missing single_indicator/leave_one_out rows."
        )

    best_single_sharpe = _safe_float(pd.to_numeric(single["sharpe"], errors="coerce").max())
    worst_loo_sharpe = _safe_float(pd.to_numeric(loo["sharpe"], errors="coerce").min())
    dominance_ratio = (
        float(best_single_sharpe / baseline_sharpe)
        if not math.isnan(baseline_sharpe) and baseline_sharpe != 0.0
        else float("nan")
    )
    largest_loo_sharpe_drop = baseline_sharpe - worst_loo_sharpe

    stage3_pass = _is_true(ablation_assessment.get("passed"))
    coverage_ok = _is_true(
        ablation_assessment.get("details", {}).get("complete_indicator_coverage", False)
    )
    no_single_fragility = (
        not math.isnan(dominance_ratio)
        and dominance_ratio <= run_config.max_single_indicator_dominance_ratio
    )
    loo_stability_ok = largest_loo_sharpe_drop <= run_config.max_loo_sharpe_drop
    ablation_robustness_ok = stage3_pass and coverage_ok and no_single_fragility and loo_stability_ok

    stage4_pass = _is_true(decision_assessment.get("passed"))
    transition_rows = decision_transition_summary.loc[
        (decision_transition_summary["variant"] == run_config.decision_variant)
        & (decision_transition_summary["transition"].isin(["BUY->SELL", "SELL->BUY"]))
    ].copy()
    if transition_rows.empty:
        raise ValueError(
            f"decision_transition_summary.csv missing BUY/SELL transitions for `{run_config.decision_variant}`."
        )
    whipsaw_rows = decision_whipsaw_summary.loc[
        decision_whipsaw_summary["variant"] == run_config.decision_variant
    ].copy()
    if whipsaw_rows.empty:
        raise ValueError(
            f"decision_whipsaw_summary.csv missing `{run_config.decision_variant}` row."
        )
    whipsaw_row = whipsaw_rows.iloc[0]

    transition_avg = _safe_float(pd.to_numeric(transition_rows["avg_switch_value"]).min())
    transition_total = _safe_float(pd.to_numeric(transition_rows["total_switch_value"]).sum())
    false_flip_total = _safe_float(whipsaw_row.get("false_flip_total_switch_value"))
    false_flip_rate = _safe_float(whipsaw_row.get("false_flip_rate"))
    # Transition totals already include false-flip outcomes, so this is the net value after whipsaws.
    net_after_whipsaw = transition_total

    transitions_value_ok = transition_avg >= run_config.min_transition_avg_switch_value
    whipsaw_rate_ok = (
        not math.isnan(false_flip_rate) and false_flip_rate <= run_config.max_false_flip_rate
    )
    net_after_whipsaw_ok = net_after_whipsaw >= run_config.min_net_transition_after_whipsaw
    decision_quality_ok = (
        stage4_pass and transitions_value_ok and whipsaw_rate_ok and net_after_whipsaw_ok
    )

    checklist = [
        _check_entry(
            check_id="signal_validity_direction_significance_monotonicity",
            passed=signal_validity_ok,
            actual={
                "stage2_passed": stage2_pass,
                "composite_spearman_corr": composite_spearman,
                "composite_min_pvalue": pvalue_min,
                "composite_top_bottom_gap": top_bottom_gap,
            },
            threshold=(
                f"stage2_pass=true, spearman>{run_config.min_composite_spearman}, "
                f"min_pvalue<={run_config.max_composite_pvalue}, "
                f"top_bottom_gap>{run_config.min_monotonic_top_bottom_gap}"
            ),
            remediation=(
                "Recalibrate Stage 2 signal construction/thresholds until composite direction, "
                "significance, and monotonicity all pass."
            ),
        ),
        _check_entry(
            check_id="subperiod_stability_majority_eras",
            passed=subperiod_stability_ok,
            actual={
                "positive_subperiods": positive_subperiods,
                "total_subperiods": total_subperiods,
                "positive_fraction": positive_fraction,
            },
            threshold=f"positive_fraction>={run_config.min_positive_subperiod_fraction}",
            remediation=(
                "Target weaker eras with robust threshold/feature tuning and rerun Stage 2 "
                "until majority-era sign consistency holds."
            ),
        ),
        _check_entry(
            check_id="ablation_no_single_feature_fragility",
            passed=ablation_robustness_ok,
            actual={
                "stage3_passed": stage3_pass,
                "indicator_coverage_complete": coverage_ok,
                "single_indicator_dominance_ratio": dominance_ratio,
                "largest_loo_sharpe_drop": largest_loo_sharpe_drop,
            },
            threshold=(
                f"stage3_pass=true, coverage=true, dominance_ratio<={run_config.max_single_indicator_dominance_ratio}, "
                f"largest_loo_sharpe_drop<={run_config.max_loo_sharpe_drop}"
            ),
            remediation=(
                "Reduce indicator dependence by revisiting aggregation and feature scaling, then rerun Stage 3."
            ),
        ),
        _check_entry(
            check_id="decision_transitions_add_value_after_whipsaw_costs",
            passed=decision_quality_ok,
            actual={
                "stage4_passed": stage4_pass,
                "decision_variant": run_config.decision_variant,
                "min_transition_avg_switch_value": transition_avg,
                "false_flip_rate": false_flip_rate,
                "transition_total_switch_value": transition_total,
                "false_flip_total_switch_value": false_flip_total,
                "net_transition_after_whipsaw": net_after_whipsaw,
            },
            threshold=(
                f"stage4_pass=true, min_transition_avg_switch_value>={run_config.min_transition_avg_switch_value}, "
                f"false_flip_rate<={run_config.max_false_flip_rate}, "
                f"net_transition_after_whipsaw>={run_config.min_net_transition_after_whipsaw}"
            ),
            remediation=(
                "Improve Stage 4 transition policy to cut false flips and raise net switch value "
                "before advancing to M2."
            ),
        ),
    ]

    failed = [entry for entry in checklist if not entry["passed"]]
    return {
        "criteria_version": run_config.criteria_version,
        "passed": len(failed) == 0,
        "checklist": checklist,
        "failed_checks": [entry["id"] for entry in failed],
    }


def run_experiment(
    config: dict[str, Any],
    cli_args: Any = None,
) -> dict[str, Any]:
    """Evaluate Stage 5 M1 readiness based on Stage 2-4 artifacts."""
    run_config = _resolve_run_config(config=config, cli_args=cli_args)
    run_config.out_dir.mkdir(parents=True, exist_ok=True)

    signal_assessment = _must_read_json(run_config.signal_assessment_path)
    ablation_assessment = _must_read_json(run_config.ablation_assessment_path)
    decision_assessment = _must_read_json(run_config.decision_assessment_path)

    signal_fullsample = _must_read_csv(run_config.signal_fullsample_path)
    signal_monotonicity = _must_read_csv(run_config.signal_monotonicity_path)
    signal_subperiods = _must_read_csv(run_config.signal_subperiods_path)
    ablation_summary = _must_read_csv(run_config.ablation_summary_path)
    decision_transition_summary = _must_read_csv(run_config.decision_transition_summary_path)
    decision_whipsaw_summary = _must_read_csv(run_config.decision_whipsaw_summary_path)

    checklist = evaluate_m1_readiness(
        run_config=run_config,
        signal_assessment=signal_assessment,
        signal_fullsample=signal_fullsample,
        signal_monotonicity=signal_monotonicity,
        signal_subperiods=signal_subperiods,
        ablation_assessment=ablation_assessment,
        ablation_summary=ablation_summary,
        decision_assessment=decision_assessment,
        decision_transition_summary=decision_transition_summary,
        decision_whipsaw_summary=decision_whipsaw_summary,
    )

    output = {
        "status": "ok",
        "criteria_version": checklist["criteria_version"],
        "readiness_passed": bool(checklist["passed"]),
        "checklist": checklist["checklist"],
        "failed_checks": checklist["failed_checks"],
        "inputs": {
            "signal_assessment": str(run_config.signal_assessment_path),
            "signal_fullsample": str(run_config.signal_fullsample_path),
            "signal_monotonicity": str(run_config.signal_monotonicity_path),
            "signal_subperiods": str(run_config.signal_subperiods_path),
            "ablation_assessment": str(run_config.ablation_assessment_path),
            "ablation_summary": str(run_config.ablation_summary_path),
            "decision_assessment": str(run_config.decision_assessment_path),
            "decision_transition_summary": str(run_config.decision_transition_summary_path),
            "decision_whipsaw_summary": str(run_config.decision_whipsaw_summary_path),
        },
    }

    checklist_path = run_config.out_dir / "m1_readiness_checklist.json"
    summary_path = run_config.out_dir / "m1_readiness_summary.md"
    checklist_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved: {checklist_path}")

    lines = [
        "# M1 Readiness Summary",
        "",
        f"- Criteria version: `{output['criteria_version']}`",
        f"- Ready for M2: `{output['readiness_passed']}`",
        f"- Failed checks: `{len(output['failed_checks'])}`",
        "",
        "## Checklist",
    ]
    for item in output["checklist"]:
        lines.append(f"- `{item['id']}`: `{item['passed']}`")

    if output["failed_checks"]:
        lines.extend(["", "## Remediation Targets"])
        for item in output["checklist"]:
            if not item["passed"]:
                lines.append(f"- `{item['id']}`: {item['remediation']}")

    lines.extend(
        [
            "",
            "## Outputs",
            f"- `{checklist_path}`",
            f"- `{summary_path}`",
        ]
    )
    write_markdown_protocol(lines, summary_path)

    return {
        "status": "ok",
        "readiness_passed": bool(output["readiness_passed"]),
        "criteria_version": output["criteria_version"],
        "out_dir": str(run_config.out_dir),
        "checklist_json": str(checklist_path),
        "summary_md": str(summary_path),
    }


__all__ = [
    "M1ReadinessRunConfig",
    "evaluate_m1_readiness",
    "run_experiment",
]
