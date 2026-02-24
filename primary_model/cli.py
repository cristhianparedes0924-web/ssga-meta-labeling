#!/usr/bin/env python3
"""Canonical CLI for primary model workflows and experiments."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
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
from primary_model.research.robustness import run_experiment as run_robustness
from primary_model.research.validation_suite import run_experiment as run_validation_suite
from primary_model.research.walk_forward import run_experiment as run_walk_forward
from primary_model.utils.config_loader import load_merged_config
from primary_model.utils.paths import get_artifacts_root

ROBUSTNESS_CONFIG = Path("configs/experiments/robustness.yaml")
WALK_FORWARD_CONFIG = Path("configs/experiments/walk_forward.yaml")
VALIDATION_CONFIG = Path("configs/experiments/validation_suite.yaml")


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
    return parser


def _resolve_validation_bool(value: bool | None, fallback: bool) -> bool:
    if value is None:
        return fallback
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command

    if command == "prepare-data":
        prepare_data(args.root.resolve())
        return 0
    if command == "data-qc":
        run_data_qc(args.root.resolve())
        return 0
    if command == "run-primary-v1":
        run_primary_variant1(args.root.resolve())
        return 0
    if command == "run-benchmarks":
        run_benchmarks({}, cli_args=args)
        return 0
    if command == "run-all":
        root = args.root.resolve()
        prepare_data(root)
        run_data_qc(root)
        run_primary_variant1(root)
        run_benchmarks({}, cli_args=args)
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
        return run_validation_suite(
            project_root=PROJECT_ROOT,
            target_root_relative=root,
            clean_root=args.clean_root,
            skip_pytest=_resolve_validation_bool(
                args.skip_pytest, bool(run_cfg.get("skip_pytest", False))
            ),
            skip_robustness=_resolve_validation_bool(
                args.skip_robustness, bool(run_cfg.get("skip_robustness", False))
            ),
            skip_walk_forward=_resolve_validation_bool(
                args.skip_walk_forward, bool(run_cfg.get("skip_walk_forward", False))
            ),
        )

    raise ValueError(f"Unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
