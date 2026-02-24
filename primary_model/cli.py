#!/usr/bin/env python3
"""Canonical CLI for primary model workflows and experiments."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from primary_model.backtest.reporting import run_primary_variant1
from primary_model.benchmarks.evaluate import run_experiment as run_benchmarks
from primary_model.data.cleaner import prepare_data
from primary_model.qc.reports import run_data_qc
from primary_model.research.ablations import run_experiment as run_ablations
from primary_model.research.decision_diagnostics import run_experiment as run_decision_diagnostics
from primary_model.research.m1_baseline import run_experiment as run_m1_baseline
from primary_model.research.m1_readiness import run_experiment as run_m1_readiness
from primary_model.research.robustness import run_experiment as run_robustness
from primary_model.research.signal_validation import run_experiment as run_signal_validation
from primary_model.research.validation_suite import run_experiment as run_validation_suite
from primary_model.research.walk_forward import run_experiment as run_walk_forward
from primary_model.utils.config_loader import load_merged_config
from primary_model.utils.paths import get_artifacts_root

ROBUSTNESS_CONFIG = Path("configs/experiments/robustness.yaml")
WALK_FORWARD_CONFIG = Path("configs/experiments/walk_forward.yaml")
WALK_FORWARD_FAST_CONFIG = Path("configs/experiments/walk_forward_fast.yaml")
VALIDATION_CONFIG = Path("configs/experiments/validation_suite.yaml")
VALIDATION_FAST_CONFIG = Path("configs/experiments/validation_suite_fast.yaml")
M1_BASELINE_CONFIG = Path("configs/experiments/m1_canonical_v1_1.yaml")
SIGNAL_VALIDATION_CONFIG = Path("configs/experiments/signal_validation.yaml")
SIGNAL_VALIDATION_FAST_CONFIG = Path("configs/experiments/signal_validation_fast.yaml")
ABLATIONS_CONFIG = Path("configs/experiments/ablations.yaml")
DECISION_DIAGNOSTICS_CONFIG = Path("configs/experiments/decision_diagnostics.yaml")
M1_READINESS_CONFIG = Path("configs/experiments/m1_readiness.yaml")
ROBUSTNESS_FAST_CONFIG = Path("configs/experiments/robustness_fast.yaml")


def _add_root_arg(
    parser: argparse.ArgumentParser,
    default: Path | None,
    *,
    from_config_fallback: bool = False,
) -> None:
    root_help = "Root containing `data/` and `reports/`."
    if from_config_fallback:
        root_help += " If omitted, uses `paths.root` from the merged config."
    parser.add_argument(
        "--root",
        type=Path,
        default=default,
        help=root_help,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Primary model workflow and experiment runner.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python cli.py run-all --root artifacts\n"
            "  python cli.py run-robustness --config configs/experiments/robustness.yaml\n"
            "  python cli.py run-validation-suite --clean-root"
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        title="commands",
        metavar="<command>",
    )

    default_root = get_artifacts_root()
    for command, help_text, description in (
        (
            "prepare-data",
            "Build cleaned datasets from raw files.",
            "Build cleaned datasets from raw Excel files into `<root>/data/clean`.",
        ),
        (
            "data-qc",
            "Generate data quality HTML report.",
            "Validate cleaned inputs and write the QC report into `<root>/reports`.",
        ),
        (
            "run-primary-v1",
            "Run PrimaryV1 strategy backtest.",
            "Run the PrimaryV1 strategy backtest and write strategy artifacts.",
        ),
        (
            "run-benchmarks",
            "Run benchmark suite.",
            "Run benchmark evaluations and write benchmark tables/plots.",
        ),
        (
            "run-all",
            "Run full main pipeline.",
            "Execute `prepare-data`, `data-qc`, `run-primary-v1`, and `run-benchmarks` in order.",
        ),
    ):
        cmd_parser = subparsers.add_parser(
            command,
            help=help_text,
            description=description,
        )
        _add_root_arg(cmd_parser, default=default_root)
        if command in {"run-primary-v1", "run-benchmarks", "run-all"}:
            cmd_parser.add_argument(
                "--aggregation-mode",
                type=str,
                choices=("dynamic", "equal_weight"),
                default="equal_weight",
                help=(
                    "Composite aggregation mode for PrimaryV1 "
                    "(`dynamic` or `equal_weight`)."
                ),
            )

    robustness = subparsers.add_parser(
        "run-robustness",
        help="Run robustness grid.",
        description="Run robustness grid sweeps for thresholds, costs, and duration assumptions.",
    )
    robustness.add_argument(
        "--config",
        type=Path,
        default=ROBUSTNESS_CONFIG,
        help=(
            "Experiment YAML path to merge over `configs/base.yaml` "
            "(default: configs/experiments/robustness.yaml)."
        ),
    )
    _add_root_arg(robustness, default=None, from_config_fallback=True)
    robustness.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for robustness artifacts (default: `<root>/reports/robustness`).",
    )
    robustness.add_argument(
        "--tcost-grid-bps",
        type=str,
        default=None,
        help="Override transaction-cost grid in bps, e.g. `0,5,10,25`.",
    )
    robustness.add_argument(
        "--buy-grid",
        type=str,
        default=None,
        help="Override buy-threshold grid, e.g. `0.0001,0.25,0.50`.",
    )
    robustness.add_argument(
        "--sell-grid",
        type=str,
        default=None,
        help="Override sell-threshold grid, e.g. `-0.0001,-0.25,-0.50`.",
    )
    robustness.add_argument(
        "--duration-grid",
        type=str,
        default=None,
        help="Override treasury-duration grid, e.g. `6.0,8.5,10.0`.",
    )

    walk_forward = subparsers.add_parser(
        "run-walk-forward",
        help="Run strict walk-forward evaluation.",
        description="Run strict expanding-window walk-forward validation for PrimaryV1.",
    )
    walk_forward.add_argument(
        "--config",
        type=Path,
        default=WALK_FORWARD_CONFIG,
        help=(
            "Experiment YAML path to merge over `configs/base.yaml` "
            "(default: configs/experiments/walk_forward.yaml)."
        ),
    )
    _add_root_arg(walk_forward, default=None, from_config_fallback=True)
    walk_forward.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for walk-forward artifacts (default: `<root>/reports/walk_forward`).",
    )
    walk_forward.add_argument(
        "--min-train-periods",
        type=int,
        default=None,
        help="Minimum in-sample periods before first OOS decision (falls back to config).",
    )
    walk_forward.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Treasury duration assumption for return approximation (falls back to config).",
    )
    walk_forward.add_argument(
        "--buy-threshold",
        type=float,
        default=None,
        help="BUY threshold override (falls back to config).",
    )
    walk_forward.add_argument(
        "--sell-threshold",
        type=float,
        default=None,
        help="SELL threshold override (falls back to config).",
    )
    walk_forward.add_argument(
        "--tcost-bps",
        type=float,
        default=None,
        help="Transaction cost override in bps (falls back to config).",
    )
    walk_forward.add_argument(
        "--engine-mode",
        type=str,
        choices=("cached_causal", "recompute_history"),
        default=None,
        help=(
            "Walk-forward engine mode override: "
            "`cached_causal` (fast) or `recompute_history` (strict, slower)."
        ),
    )

    validation = subparsers.add_parser(
        "run-validation-suite",
        help="Run full validation suite with manifest outputs.",
        description=(
            "Run the validation suite with reproducibility logs and manifest output.\n"
            "Unless explicitly overridden, skip flags are sourced from experiment config."
        ),
    )
    validation.add_argument(
        "--config",
        type=Path,
        default=VALIDATION_CONFIG,
        help=(
            "Experiment YAML path to merge over `configs/base.yaml` "
            "(default: configs/experiments/validation_suite.yaml)."
        ),
    )
    _add_root_arg(validation, default=None, from_config_fallback=True)
    validation.add_argument(
        "--clean-root",
        action="store_true",
        help="Clean target root reports and clean-data folders before execution.",
    )
    validation.add_argument(
        "--skip-pytest",
        action="store_true",
        default=None,
        help="Skip unit/integration tests (otherwise uses config value).",
    )
    validation.add_argument(
        "--skip-run-all",
        action="store_true",
        default=None,
        help="Skip main pipeline `run-all` step (otherwise uses config value).",
    )
    validation.add_argument(
        "--skip-robustness",
        action="store_true",
        default=None,
        help="Skip robustness step (otherwise uses config value).",
    )
    validation.add_argument(
        "--skip-walk-forward",
        action="store_true",
        default=None,
        help="Skip walk-forward step (otherwise uses config value).",
    )
    validation.add_argument(
        "--run-m1-readiness",
        action="store_true",
        default=None,
        help="Run Stage 5 readiness hook at end of validation suite (otherwise uses config value).",
    )
    validation.add_argument(
        "--robustness-config",
        type=Path,
        default=None,
        help=(
            "Robustness experiment config override for validation suite "
            "(default from validation config or configs/experiments/robustness.yaml)."
        ),
    )
    validation.add_argument(
        "--walk-forward-config",
        type=Path,
        default=None,
        help=(
            "Walk-forward experiment config override for validation suite "
            "(default from validation config or configs/experiments/walk_forward.yaml)."
        ),
    )
    validation.add_argument(
        "--m1-readiness-config",
        type=Path,
        default=None,
        help=(
            "M1 readiness config override for validation suite readiness hook "
            "(default from validation config or configs/experiments/m1_readiness.yaml)."
        ),
    )

    validation_fast = subparsers.add_parser(
        "run-validation-fast",
        help="Run fast validation profile for development iteration.",
        description=(
            "Run a fast validation profile using lightweight experiment configs and "
            "skipping redundant heavy steps where configured."
        ),
    )
    validation_fast.add_argument(
        "--config",
        type=Path,
        default=VALIDATION_FAST_CONFIG,
        help=(
            "Fast validation suite config path "
            "(default: configs/experiments/validation_suite_fast.yaml)."
        ),
    )
    _add_root_arg(validation_fast, default=None, from_config_fallback=True)
    validation_fast.add_argument(
        "--clean-root",
        action="store_true",
        help="Clean target root reports and clean-data folders before execution.",
    )

    m1_baseline = subparsers.add_parser(
        "run-m1-baseline",
        help="Run frozen M1 baseline and write reproducibility snapshot.",
        description=(
            "Run the canonical PrimaryV1.1 baseline workflow and write a determinism snapshot "
            "with artifact hashes."
        ),
    )
    m1_baseline.add_argument(
        "--config",
        type=Path,
        default=M1_BASELINE_CONFIG,
        help=(
            "Experiment YAML path to merge over `configs/base.yaml` "
            "(default: configs/experiments/m1_canonical_v1_1.yaml)."
        ),
    )
    _add_root_arg(m1_baseline, default=None, from_config_fallback=True)
    m1_baseline.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for Stage 1 snapshot artifacts (default: `<root>/reports/reproducibility`).",
    )
    m1_baseline.add_argument(
        "--determinism-runs",
        type=int,
        default=None,
        help="Number of repeated baseline runs used for determinism check (default from config).",
    )
    m1_baseline.add_argument(
        "--no-determinism-check",
        action="store_true",
        help="Disable repeated-run determinism check and run baseline once.",
    )

    signal_validation = subparsers.add_parser(
        "run-signal-validation",
        help="Run Stage 2 signal validity and monotonicity diagnostics.",
        description=(
            "Validate indicator/composite predictive quality versus forward relative outcomes "
            "and write Stage 2 artifacts."
        ),
    )
    signal_validation.add_argument(
        "--config",
        type=Path,
        default=SIGNAL_VALIDATION_CONFIG,
        help=(
            "Experiment YAML path to merge over `configs/base.yaml` "
            "(default: configs/experiments/signal_validation.yaml)."
        ),
    )
    _add_root_arg(signal_validation, default=None, from_config_fallback=True)
    signal_validation.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for signal validation artifacts (default: `<root>/reports/signal_validation`).",
    )
    signal_validation.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Treasury duration assumption override (falls back to config).",
    )
    signal_validation.add_argument(
        "--buy-threshold",
        type=float,
        default=None,
        help="BUY threshold override (falls back to config).",
    )
    signal_validation.add_argument(
        "--sell-threshold",
        type=float,
        default=None,
        help="SELL threshold override (falls back to config).",
    )
    signal_validation.add_argument(
        "--bins",
        type=int,
        default=None,
        help="Quantile bins for monotonicity analysis (falls back to config).",
    )
    signal_validation.add_argument(
        "--bootstrap-samples",
        type=int,
        default=None,
        help="Bootstrap sample count override (falls back to config).",
    )
    signal_validation.add_argument(
        "--min-pairs",
        type=int,
        default=None,
        help="Minimum valid pairs required for statistical estimation (falls back to config).",
    )

    ablations = subparsers.add_parser(
        "run-ablations",
        help="Run Stage 3 ablation suite.",
        description=(
            "Run leave-one-out, single-indicator, and aggregation comparison studies "
            "for PrimaryV1 signal components."
        ),
    )
    ablations.add_argument(
        "--config",
        type=Path,
        default=ABLATIONS_CONFIG,
        help=(
            "Experiment YAML path to merge over `configs/base.yaml` "
            "(default: configs/experiments/ablations.yaml)."
        ),
    )
    _add_root_arg(ablations, default=None, from_config_fallback=True)
    ablations.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for ablation artifacts (default: `<root>/reports/ablations`).",
    )
    ablations.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Treasury duration assumption override (falls back to config).",
    )
    ablations.add_argument(
        "--buy-threshold",
        type=float,
        default=None,
        help="BUY threshold override (falls back to config).",
    )
    ablations.add_argument(
        "--sell-threshold",
        type=float,
        default=None,
        help="SELL threshold override (falls back to config).",
    )
    ablations.add_argument(
        "--tcost-bps",
        type=float,
        default=None,
        help="Transaction cost override in bps (falls back to config).",
    )

    decision = subparsers.add_parser(
        "run-decision-diagnostics",
        help="Run Stage 4 decision-quality diagnostics.",
        description=(
            "Run transition, whipsaw, dwell-time, and score-zone diagnostics "
            "for selected signal variants."
        ),
    )
    decision.add_argument(
        "--config",
        type=Path,
        default=DECISION_DIAGNOSTICS_CONFIG,
        help=(
            "Experiment YAML path to merge over `configs/base.yaml` "
            "(default: configs/experiments/decision_diagnostics.yaml)."
        ),
    )
    _add_root_arg(decision, default=None, from_config_fallback=True)
    decision.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for decision diagnostics artifacts (default: `<root>/reports/decision_diagnostics`).",
    )
    decision.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Treasury duration assumption override (falls back to config).",
    )
    decision.add_argument(
        "--buy-threshold",
        type=float,
        default=None,
        help="BUY threshold override (falls back to config).",
    )
    decision.add_argument(
        "--sell-threshold",
        type=float,
        default=None,
        help="SELL threshold override (falls back to config).",
    )
    decision.add_argument(
        "--tcost-bps",
        type=float,
        default=None,
        help="Transaction cost override in bps (falls back to config).",
    )
    decision.add_argument(
        "--bins",
        type=int,
        default=None,
        help="Number of score quantile bins (falls back to config).",
    )
    decision.add_argument(
        "--min-pairs",
        type=int,
        default=None,
        help="Minimum valid observations required for zone analysis (falls back to config).",
    )
    decision.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated variant list, e.g. `dynamic_all,equal_weight_all`.",
    )

    readiness = subparsers.add_parser(
        "run-m1-readiness",
        help="Run Stage 5 M1 readiness gate.",
        description=(
            "Build objective M1 go/no-go checklist from Stage 2-4 artifacts "
            "and write readiness reports."
        ),
    )
    readiness.add_argument(
        "--config",
        type=Path,
        default=M1_READINESS_CONFIG,
        help=(
            "Experiment YAML path to merge over `configs/base.yaml` "
            "(default: configs/experiments/m1_readiness.yaml)."
        ),
    )
    _add_root_arg(readiness, default=None, from_config_fallback=True)
    readiness.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for readiness artifacts (default: `<root>/reports/readiness`).",
    )

    dev_loop = subparsers.add_parser(
        "run-dev-loop",
        help="Run optimized development loop (fast configs).",
        description=(
            "Run a fast end-to-end dev loop: main pipeline, signal validation fast, "
            "ablations, decision diagnostics, and readiness."
        ),
    )
    _add_root_arg(dev_loop, default=default_root)
    dev_loop.add_argument(
        "--skip-main-pipeline",
        action="store_true",
        help="Skip `run-all` and reuse existing artifacts root data/reports.",
    )
    dev_loop.add_argument(
        "--skip-ablations",
        action="store_true",
        help="Skip Stage 3 ablations in the dev loop.",
    )
    return parser


def _resolve_validation_bool(value: bool | None, fallback: bool) -> bool:
    if value is None:
        return fallback
    return value


def _warn_about_unc_wsl_path() -> None:
    """Warn when running Windows Python against UNC WSL mounts (slow IO path)."""
    if os.name != "nt":
        return
    cwd = str(Path.cwd())
    if cwd.startswith("\\\\wsl.localhost\\"):
        print(
            "WARNING: Running Windows Python on a \\\\wsl.localhost UNC path can be slow. "
            "For faster IO, run inside WSL with Linux Python."
        )


def main(argv: list[str] | None = None) -> int:
    _warn_about_unc_wsl_path()
    args = build_parser().parse_args(argv)
    command = args.command

    if command == "prepare-data":
        prepare_data(args.root.resolve())
        return 0
    if command == "data-qc":
        run_data_qc(args.root.resolve())
        return 0
    if command == "run-primary-v1":
        run_primary_variant1(
            args.root.resolve(),
            aggregation_mode=args.aggregation_mode,
        )
        return 0
    if command == "run-benchmarks":
        run_benchmarks({"run": {"aggregation_mode": args.aggregation_mode}}, cli_args=args)
        return 0
    if command == "run-all":
        root = args.root.resolve()
        prepare_data(root)
        run_data_qc(root)
        run_primary_variant1(root, aggregation_mode=args.aggregation_mode)
        run_benchmarks({"run": {"aggregation_mode": args.aggregation_mode}}, cli_args=args)
        return 0

    if command == "run-robustness":
        config = load_merged_config("base.yaml", args.config)
        status = run_robustness(config=config, cli_args=args)
        return 0 if status.get("status") == "ok" else 1

    if command == "run-walk-forward":
        config = load_merged_config("base.yaml", args.config)
        status = run_walk_forward(config=config, cli_args=args)
        return 0 if status.get("status") == "ok" else 1

    if command == "run-validation-suite":
        config = load_merged_config("base.yaml", args.config)
        run_cfg = config.get("run", {})
        default_root = Path(config.get("paths", {}).get("root", get_artifacts_root()))
        root = args.root if args.root is not None else default_root
        robustness_cfg = args.robustness_config or Path(
            run_cfg.get("robustness_config", "configs/experiments/robustness.yaml")
        )
        walk_forward_cfg = args.walk_forward_config or Path(
            run_cfg.get("walk_forward_config", "configs/experiments/walk_forward.yaml")
        )
        readiness_cfg = args.m1_readiness_config or Path(
            run_cfg.get("m1_readiness_config", "configs/experiments/m1_readiness.yaml")
        )
        return run_validation_suite(
            project_root=PROJECT_ROOT,
            target_root_relative=root,
            clean_root=args.clean_root,
            skip_pytest=_resolve_validation_bool(
                args.skip_pytest, bool(run_cfg.get("skip_pytest", False))
            ),
            skip_run_all=_resolve_validation_bool(
                args.skip_run_all, bool(run_cfg.get("skip_run_all", False))
            ),
            skip_robustness=_resolve_validation_bool(
                args.skip_robustness, bool(run_cfg.get("skip_robustness", False))
            ),
            skip_walk_forward=_resolve_validation_bool(
                args.skip_walk_forward, bool(run_cfg.get("skip_walk_forward", False))
            ),
            run_m1_readiness=_resolve_validation_bool(
                args.run_m1_readiness, bool(run_cfg.get("run_m1_readiness", False))
            ),
            robustness_config=robustness_cfg,
            walk_forward_config=walk_forward_cfg,
            m1_readiness_config=readiness_cfg,
        )

    if command == "run-validation-fast":
        config = load_merged_config("base.yaml", args.config)
        run_cfg = config.get("run", {})
        default_root = Path(config.get("paths", {}).get("root", get_artifacts_root()))
        root = args.root if args.root is not None else default_root
        return run_validation_suite(
            project_root=PROJECT_ROOT,
            target_root_relative=root,
            clean_root=args.clean_root,
            skip_pytest=bool(run_cfg.get("skip_pytest", True)),
            skip_run_all=bool(run_cfg.get("skip_run_all", False)),
            skip_robustness=bool(run_cfg.get("skip_robustness", False)),
            skip_walk_forward=bool(run_cfg.get("skip_walk_forward", False)),
            run_m1_readiness=bool(run_cfg.get("run_m1_readiness", False)),
            robustness_config=Path(
                run_cfg.get("robustness_config", str(ROBUSTNESS_FAST_CONFIG))
            ),
            walk_forward_config=Path(
                run_cfg.get("walk_forward_config", str(WALK_FORWARD_FAST_CONFIG))
            ),
            m1_readiness_config=Path(
                run_cfg.get("m1_readiness_config", str(M1_READINESS_CONFIG))
            ),
        )

    if command == "run-m1-baseline":
        config = load_merged_config("base.yaml", args.config)
        status = run_m1_baseline(config=config, cli_args=args)
        return 0 if status.get("status") == "ok" else 1

    if command == "run-signal-validation":
        config = load_merged_config("base.yaml", args.config)
        status = run_signal_validation(config=config, cli_args=args)
        return 0 if status.get("status") == "ok" else 1

    if command == "run-ablations":
        config = load_merged_config("base.yaml", args.config)
        status = run_ablations(config=config, cli_args=args)
        return 0 if status.get("status") == "ok" else 1

    if command == "run-decision-diagnostics":
        config = load_merged_config("base.yaml", args.config)
        status = run_decision_diagnostics(config=config, cli_args=args)
        return 0 if status.get("status") == "ok" else 1

    if command == "run-m1-readiness":
        config = load_merged_config("base.yaml", args.config)
        status = run_m1_readiness(config=config, cli_args=args)
        return 0 if status.get("status") == "ok" else 1

    if command == "run-dev-loop":
        root = args.root.resolve()
        if not args.skip_main_pipeline:
            prepare_data(root)
            run_data_qc(root)
            run_primary_variant1(root)
            run_benchmarks({}, cli_args=args)

        signal_cfg = load_merged_config("base.yaml", SIGNAL_VALIDATION_FAST_CONFIG)
        signal_status = run_signal_validation(
            config=signal_cfg,
            cli_args=argparse.Namespace(
                root=root,
                out_dir=None,
                duration=None,
                buy_threshold=None,
                sell_threshold=None,
                bins=None,
                bootstrap_samples=None,
                min_pairs=None,
            ),
        )
        if signal_status.get("status") != "ok":
            return 1

        if not args.skip_ablations:
            ablation_cfg = load_merged_config("base.yaml", ABLATIONS_CONFIG)
            ablation_status = run_ablations(
                config=ablation_cfg,
                cli_args=argparse.Namespace(
                    root=root,
                    out_dir=None,
                    duration=None,
                    buy_threshold=None,
                    sell_threshold=None,
                    tcost_bps=None,
                ),
            )
            if ablation_status.get("status") != "ok":
                return 1

        decision_cfg = load_merged_config("base.yaml", DECISION_DIAGNOSTICS_CONFIG)
        decision_status = run_decision_diagnostics(
            config=decision_cfg,
            cli_args=argparse.Namespace(
                root=root,
                out_dir=None,
                duration=None,
                buy_threshold=None,
                sell_threshold=None,
                tcost_bps=None,
                bins=None,
                min_pairs=None,
                variants=None,
            ),
        )
        if decision_status.get("status") != "ok":
            return 1

        readiness_cfg = load_merged_config("base.yaml", M1_READINESS_CONFIG)
        readiness_status = run_m1_readiness(
            config=readiness_cfg,
            cli_args=argparse.Namespace(
                root=root,
                out_dir=None,
            ),
        )
        return 0 if readiness_status.get("status") == "ok" else 1

    raise ValueError(f"Unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
