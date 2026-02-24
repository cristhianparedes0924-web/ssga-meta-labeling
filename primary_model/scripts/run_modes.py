#!/usr/bin/env python3
"""Simple runner for project mode, artifacts mode, or both."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

try:
    from ._bootstrap import ensure_project_on_path
except ImportError:  # pragma: no cover - direct script execution path
    from _bootstrap import ensure_project_on_path


ensure_project_on_path()

from primary_model.utils.paths import get_artifacts_root, get_project_root

PROJECT_ROOT = get_project_root()
ARTIFACTS_ROOT = get_artifacts_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run main project pipeline, isolated tests, or both."
    )
    parser.add_argument(
        "--mode",
        choices=["project", "artifacts", "both"],
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

    project_cmd = [
        sys.executable,
        "cli.py",
        "run-all",
        "--root",
        str(ARTIFACTS_ROOT),
    ]
    test_cmd = [
        sys.executable,
        "cli.py",
        "run-validation-suite",
        "--root",
        str(ARTIFACTS_ROOT),
        "--clean-root",
    ]

    if args.mode == "project":
        _run(project_cmd)
        return
    if args.mode == "artifacts":
        _run(test_cmd)
        return

    _run(project_cmd)
    _run(test_cmd)


if __name__ == "__main__":
    main()
