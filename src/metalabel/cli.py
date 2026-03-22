"""Command-line interface for the modular meta-labeling project."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from metalabel import PROJECT_ROOT, load_primary_config
from metalabel.data import _resolve_within_project, prepare_data
from metalabel.primary.pipeline import run_all, run_benchmarks, run_primary_variant1
from metalabel.validation import run_data_qc, run_modes, run_monthly_cv, run_robustness, run_self_tests, run_validation_suite, run_walk_forward, setup_test_root


def build_parser() -> argparse.ArgumentParser:
    config = load_primary_config()
    primary_cfg = config["primary"]
    validation_cfg = config["validation"]
    robustness_cfg = validation_cfg["robustness"]

    parser = argparse.ArgumentParser(
        description="CLI for the meta-labeling primary-model repository."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    core_help = {
        "prepare-data": "Run raw Excel to clean CSV data preparation.",
        "data-qc": "Run data quality report generation.",
        "run-primary-v1": "Run PrimaryV1 strategy and summary output.",
        "run-benchmarks": "Run benchmark suite and results.",
        "run-all": "Run full core pipeline (prepare-data, data-qc, strategy, benchmarks).",
    }
    for cmd, help_text in core_help.items():
        core_parser = subparsers.add_parser(cmd, help=help_text)
        core_parser.add_argument(
            "--root",
            type=Path,
            default=PROJECT_ROOT,
            help="Project root (must stay inside this repository folder).",
        )

    subparsers.add_parser(
        "run-self-tests",
        aliases=["run_self_tests"],
        help="Run native built-in tests from this project.",
    )

    mode_parser = subparsers.add_parser(
        "run-modes",
        aliases=["run_modes"],
        help="Run project workflow, validation workflow, or both.",
    )
    mode_parser.add_argument(
        "--mode",
        choices=["project", "test", "both"],
        required=True,
        help="Which execution mode to run.",
    )

    setup_parser = subparsers.add_parser(
        "setup-test-root",
        aliases=["setup_test_root"],
        help="Bootstrap isolated test root with raw input files.",
    )
    setup_parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("test"),
        help="Target isolated root to populate inside this repository (default: test).",
    )
    setup_parser.add_argument(
        "--source-raw",
        type=Path,
        default=Path("data/raw"),
        help="Raw-data directory (must be this repository's data/raw).",
    )
    setup_parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing target data/clean and reports before setup.",
    )

    robustness_parser = subparsers.add_parser(
        "run-robustness",
        aliases=["run_robustness"],
        help="Run robustness grid for costs, thresholds, and duration assumptions.",
    )
    robustness_parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Root containing data/clean for evaluation (default: repository folder).",
    )
    robustness_parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <root>/reports/results/robustness).",
    )
    robustness_parser.add_argument(
        "--tcost-grid-bps",
        type=str,
        default=str(robustness_cfg["tcost_grid_bps"]),
        help="Comma-separated transaction cost grid in bps.",
    )
    robustness_parser.add_argument(
        "--buy-grid",
        type=str,
        default=str(robustness_cfg["buy_grid"]),
        help="Comma-separated buy-threshold grid.",
    )
    robustness_parser.add_argument(
        "--sell-grid",
        type=str,
        default=str(robustness_cfg["sell_grid"]),
        help="Comma-separated sell-threshold grid.",
    )
    robustness_parser.add_argument(
        "--duration-grid",
        type=str,
        default=str(robustness_cfg["duration_grid"]),
        help="Comma-separated treasury-duration grid.",
    )

    walk_parser = subparsers.add_parser(
        "run-walk-forward",
        aliases=["run_walk_forward"],
        help="Run strict walk-forward validation.",
    )
    walk_parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Root containing data/clean for evaluation (default: repository folder).",
    )
    walk_parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <root>/reports/results/walk_forward).",
    )
    walk_parser.add_argument(
        "--min-train-periods",
        type=int,
        default=int(validation_cfg["min_train_periods"]),
        help="Minimum in-sample history before first OOS decision.",
    )
    walk_parser.add_argument(
        "--duration",
        type=float,
        default=float(primary_cfg["duration"]),
        help="Treasury duration used for return approximation.",
    )
    walk_parser.add_argument(
        "--buy-threshold",
        type=float,
        default=float(primary_cfg["buy_threshold"]),
        help="BUY threshold for signal mapping.",
    )
    walk_parser.add_argument(
        "--sell-threshold",
        type=float,
        default=float(primary_cfg["sell_threshold"]),
        help="SELL threshold for signal mapping.",
    )
    walk_parser.add_argument(
        "--tcost-bps",
        type=float,
        default=float(primary_cfg["tcost_bps"]),
        help="Transaction costs in bps.",
    )

    monthly_cv_parser = subparsers.add_parser(
        "run-monthly-cv",
        aliases=["run_monthly_cv"],
        help="Run expanding month-based cross-validation.",
    )
    monthly_cv_parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Root containing data/clean for evaluation (default: repository folder).",
    )
    monthly_cv_parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <root>/reports/results/monthly_cv).",
    )
    monthly_cv_parser.add_argument(
        "--min-train-periods",
        type=int,
        default=int(validation_cfg["min_train_periods"]),
        help="Minimum in-sample history before first OOS month.",
    )
    monthly_cv_cfg = validation_cfg.get("monthly_cv", {})
    monthly_cv_parser.add_argument(
        "--test-window-months",
        type=int,
        default=int(monthly_cv_cfg.get("test_window_months", 1)),
        help="Number of calendar months in each OOS fold window (default: 1).",
    )
    monthly_cv_parser.add_argument(
        "--duration",
        type=float,
        default=float(primary_cfg["duration"]),
        help="Treasury duration used for return approximation.",
    )
    monthly_cv_parser.add_argument(
        "--buy-threshold",
        type=float,
        default=float(primary_cfg["buy_threshold"]),
        help="BUY threshold for signal mapping.",
    )
    monthly_cv_parser.add_argument(
        "--sell-threshold",
        type=float,
        default=float(primary_cfg["sell_threshold"]),
        help="SELL threshold for signal mapping.",
    )
    monthly_cv_parser.add_argument(
        "--tcost-bps",
        type=float,
        default=float(primary_cfg["tcost_bps"]),
        help="Transaction costs in bps.",
    )

    validation_parser = subparsers.add_parser(
        "run-validation-suite",
        aliases=["run_validation_suite"],
        help="Run full isolated validation suite and reproducibility logging.",
    )
    validation_parser.add_argument(
        "--root",
        type=Path,
        default=Path("test"),
        help="Isolated root for data and reports (default: test).",
    )
    validation_parser.add_argument(
        "--clean-root",
        action="store_true",
        help="Clean target root reports and clean data before running.",
    )
    validation_parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip unit and integration tests.",
    )
    validation_parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Skip robustness grid.",
    )
    validation_parser.add_argument(
        "--skip-walk-forward",
        action="store_true",
        help="Skip strict walk-forward run.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")

    if argv is None:
        argv = sys.argv[1:]
    if len(argv) == 0:
        argv = ["run-all"]

    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command

    if command in {"prepare-data", "data-qc", "run-primary-v1", "run-benchmarks", "run-all"}:
        root = _resolve_within_project(args.root, "root")
        if command == "prepare-data":
            prepare_data(root)
        elif command == "data-qc":
            run_data_qc(root)
        elif command == "run-primary-v1":
            run_primary_variant1(root)
        elif command == "run-benchmarks":
            run_benchmarks(root)
        elif command == "run-all":
            run_all(root)
        return

    if command in {"run-self-tests", "run_self_tests"}:
        run_self_tests()
        return

    if command in {"run-modes", "run_modes"}:
        run_modes(args.mode)
        return

    if command in {"setup-test-root", "setup_test_root"}:
        setup_test_root(
            target_root=args.target_root,
            source_raw=args.source_raw,
            clean=args.clean,
        )
        return

    if command in {"run-robustness", "run_robustness"}:
        run_robustness(
            root=args.root,
            out_dir=args.out_dir,
            tcost_grid_bps=args.tcost_grid_bps,
            buy_grid=args.buy_grid,
            sell_grid=args.sell_grid,
            duration_grid=args.duration_grid,
        )
        return

    if command in {"run-walk-forward", "run_walk_forward"}:
        run_walk_forward(
            root=args.root,
            out_dir=args.out_dir,
            min_train_periods=args.min_train_periods,
            duration=args.duration,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
            tcost_bps=args.tcost_bps,
        )
        return

    if command in {"run-monthly-cv", "run_monthly_cv"}:
        run_monthly_cv(
            root=args.root,
            out_dir=args.out_dir,
            min_train_periods=args.min_train_periods,
            test_window_months=args.test_window_months,
            duration=args.duration,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
            tcost_bps=args.tcost_bps,
        )
        return

    if command in {"run-validation-suite", "run_validation_suite"}:
        run_validation_suite(
            root=args.root,
            clean_root=args.clean_root,
            skip_pytest=args.skip_pytest,
            skip_robustness=args.skip_robustness,
            skip_walk_forward=args.skip_walk_forward,
        )
        return

    parser.error(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
