#!/usr/bin/env python3
"""Simple runner for project mode, test mode, or both."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run main project pipeline, isolated tests, or both."
    )
    parser.add_argument(
        "--mode",
        choices=["project", "test", "both"],
        required=True,
        help="Which execution mode to run.",
    )
    return parser.parse_args()


def _run(command: list[str]) -> None:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    print(f"$ {' '.join(shlex.quote(tok) for tok in command)}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    args = parse_args()

    project_cmd = [sys.executable, "cli.py", "run-all"]
    test_cmd = [
        sys.executable,
        "scripts/run_validation_suite.py",
        "--root",
        "test",
        "--clean-root",
    ]

    if args.mode == "project":
        _run(project_cmd)
        return
    if args.mode == "test":
        _run(test_cmd)
        return

    _run(project_cmd)
    _run(test_cmd)


if __name__ == "__main__":
    main()
