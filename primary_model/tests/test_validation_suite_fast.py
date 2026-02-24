from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_validation_suite_skip_run_all_and_heavy_steps(tmp_path: Path) -> None:
    run_root = tmp_path / "validation_skip_project"
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    cmd = [
        sys.executable,
        "cli.py",
        "run-validation-suite",
        "--root",
        str(run_root),
        "--skip-pytest",
        "--skip-run-all",
        "--skip-robustness",
        "--skip-walk-forward",
    ]
    subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    repro_dir = run_root / "reports" / "reproducibility"
    manifest_path = repro_dir / "validation_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    commands = manifest.get("commands", [])
    assert len(commands) == 1
    assert "run-all" not in commands[0]


def test_run_validation_fast_command_with_minimal_config(tmp_path: Path) -> None:
    run_root = tmp_path / "validation_fast_project"
    cfg = tmp_path / "validation_fast_minimal.yaml"
    cfg.write_text(
        "\n".join(
            [
                "experiment:",
                '  name: "validation_suite_fast_minimal"',
                "",
                "run:",
                "  skip_pytest: true",
                "  skip_run_all: true",
                "  skip_robustness: true",
                "  skip_walk_forward: true",
                "  run_m1_readiness: false",
            ]
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    cmd = [
        sys.executable,
        "cli.py",
        "run-validation-fast",
        "--root",
        str(run_root),
        "--config",
        str(cfg),
    ]
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Validation suite completed successfully." in completed.stdout
    assert (run_root / "reports" / "reproducibility" / "validation_manifest.json").exists()

