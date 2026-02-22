#!/usr/bin/env python3
"""Run full isolated validation suite and write reproducibility artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full validation suite in isolated root.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("test"),
        help="Isolated root for data/reports (default: test).",
    )
    parser.add_argument(
        "--clean-root",
        action="store_true",
        help="Clean target root reports/clean data before running.",
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip unit/integration tests.",
    )
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Skip robustness grid.",
    )
    parser.add_argument(
        "--skip-walk-forward",
        action="store_true",
        help="Skip strict walk-forward run.",
    )
    return parser.parse_args()


def _git_commit_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(PROJECT_ROOT.parent), "rev-parse", "HEAD"], text=True
            )
            .strip()
        )
    except Exception:
        return "unknown"


def _run_and_log(command: list[str], log_path: Path) -> None:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        env=env,
    )

    log_lines = [
        f"$ {' '.join(shlex.quote(tok) for tok in command)}",
        "",
        "STDOUT:",
        completed.stdout,
        "",
        "STDERR:",
        completed.stderr,
        "",
        f"exit_code={completed.returncode}",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    print(f"Ran: {' '.join(command)}")
    print(f"Log: {log_path}")
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {command}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    target_root = (PROJECT_ROOT / args.root).resolve()
    reports_dir = target_root / "reports"
    repro_dir = reports_dir / "reproducibility"
    repro_dir.mkdir(parents=True, exist_ok=True)

    commands: list[list[str]] = []

    setup_cmd = [
        sys.executable,
        "scripts/setup_test_root.py",
        "--target-root",
        str(args.root),
    ]
    if args.clean_root:
        setup_cmd.append("--clean")
    commands.append(setup_cmd)

    if not args.skip_pytest:
        commands.append([sys.executable, "-m", "pytest", "-q"])

    commands.append([sys.executable, "cli.py", "run-all", "--root", str(args.root)])

    if not args.skip_robustness:
        commands.append(
            [
                sys.executable,
                "scripts/run_robustness.py",
                "--root",
                str(args.root),
                "--out-dir",
                str(target_root / "reports" / "robustness"),
            ]
        )

    if not args.skip_walk_forward:
        commands.append(
            [
                sys.executable,
                "scripts/run_walk_forward.py",
                "--root",
                str(args.root),
                "--out-dir",
                str(target_root / "reports" / "walk_forward"),
            ]
        )

    for i, command in enumerate(commands, start=1):
        log_path = repro_dir / f"{i:02d}_{Path(command[1]).stem}.log"
        _run_and_log(command, log_path)

    artifacts: dict[str, str] = {}
    if reports_dir.exists():
        for path in sorted(reports_dir.rglob("*")):
            if path.is_file():
                artifacts[str(path.relative_to(PROJECT_ROOT))] = _sha256(path)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "isolated_root": str(target_root),
        "git_commit": _git_commit_sha(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "commands": commands,
        "artifact_sha256": artifacts,
    }
    manifest_path = repro_dir / "validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary_lines = [
        "# Validation Suite Summary",
        "",
        f"- Isolated root: `{target_root}`",
        f"- Git commit: `{manifest['git_commit']}`",
        f"- Python: `{manifest['python_version']}`",
        f"- Platform: `{manifest['platform']}`",
        f"- Commands executed: `{len(commands)}`",
        f"- Hashed artifacts: `{len(artifacts)}`",
        "",
        "## Key Outputs",
        f"- `{target_root / 'reports' / 'benchmarks_summary.csv'}`",
        f"- `{target_root / 'reports' / 'primary_v1_summary.csv'}`",
        f"- `{target_root / 'reports' / 'robustness' / 'robustness_grid_results.csv'}`",
        f"- `{target_root / 'reports' / 'walk_forward' / 'walk_forward_summary.csv'}`",
        f"- `{manifest_path}`",
    ]
    summary_path = repro_dir / "validation_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Saved: {manifest_path}")
    print(f"Saved: {summary_path}")
    print("Validation suite completed successfully.")


if __name__ == "__main__":
    main()
