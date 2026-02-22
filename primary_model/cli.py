#!/usr/bin/env python3
"""Command-line interface for the standalone primary model pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from backtest.reporting import run_benchmarks, run_primary_variant1
from data.cleaner import prepare_data
from qc.reports import run_data_qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runner for the primary_model project.")
    parser.add_argument(
        "command",
        choices=["prepare-data", "data-qc", "run-primary-v1", "run-benchmarks", "run-all"],
        help="Which project workflow to run.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root containing data/, reports/, and package modules.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    if args.command == "prepare-data":
        prepare_data(root)
    elif args.command == "data-qc":
        run_data_qc(root)
    elif args.command == "run-primary-v1":
        run_primary_variant1(root)
    elif args.command == "run-benchmarks":
        run_benchmarks(root)
    elif args.command == "run-all":
        prepare_data(root)
        run_data_qc(root)
        run_primary_variant1(root)
        run_benchmarks(root)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
