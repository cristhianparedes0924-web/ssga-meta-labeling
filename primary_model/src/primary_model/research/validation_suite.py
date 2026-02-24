"""
Validation suite orchestration module.

Moved from scripts/run_validation_suite.py.
Refactored to orchestrate internally natively where possible.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ValidationSuiteRunConfig:
    project_root: Path
    target_root_relative: Path
    clean_root: bool
    skip_pytest: bool
    skip_robustness: bool
    skip_walk_forward: bool


def _git_commit_sha(project_root: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(project_root.parent), "rev-parse", "HEAD"], text=True
            )
            .strip()
        )
    except Exception:
        return "unknown"


def _run_and_log_subprocess(command: list[str], log_path: Path, project_root: Path) -> bool:
    """Run an external tool boundary command and log it."""
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    
    print(f"Subprocess: {' '.join(command)}")

    completed = subprocess.run(
        command,
        cwd=project_root,
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

    if completed.returncode != 0:
        print(f"FAILED (Exit {completed.returncode}): {log_path}")
        return False
    return True


def _run_and_log_native(
    step_name: str,
    func: Callable[..., Any],
    log_path: Path,
    *args,
    **kwargs,
) -> bool:
    """Run an internal python callable and capture its exception state manually to log."""
    print(f"Native: {step_name}")
    
    success = False
    stdout_capture = "N/A (Native execution)"
    stderr_capture = ""
    exit_code = 0
    
    try:
        func(*args, **kwargs)
        success = True
    except Exception:
        stderr_capture = traceback.format_exc()
        exit_code = 1

    log_lines = [
        f"Native Execution: {step_name}",
        "",
        "STDOUT:",
        stdout_capture,
        "",
        "STDERR (Exception):",
        stderr_capture,
        "",
        f"exit_code={exit_code}",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    if not success:
        print(f"FAILED (Exception): {log_path}")
    return success


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_run_config(
    project_root: Path,
    target_root_relative: Path,
    clean_root: bool,
    skip_pytest: bool,
    skip_robustness: bool,
    skip_walk_forward: bool,
) -> ValidationSuiteRunConfig:
    return ValidationSuiteRunConfig(
        project_root=project_root,
        target_root_relative=target_root_relative,
        clean_root=clean_root,
        skip_pytest=skip_pytest,
        skip_robustness=skip_robustness,
        skip_walk_forward=skip_walk_forward,
    )


def run_experiment(
    project_root: Path,
    target_root_relative: Path,
    clean_root: bool,
    skip_pytest: bool,
    skip_robustness: bool,
    skip_walk_forward: bool,
) -> int:
    """
    Execute the full validation suite natively where possible.
    Returns 0 on full success, 1 on any sequence failure.
    """
    run_config = _resolve_run_config(
        project_root=project_root,
        target_root_relative=target_root_relative,
        clean_root=clean_root,
        skip_pytest=skip_pytest,
        skip_robustness=skip_robustness,
        skip_walk_forward=skip_walk_forward,
    )

    target_root = (run_config.project_root / run_config.target_root_relative).resolve()
    reports_dir = target_root / "reports"
    repro_dir = reports_dir / "reproducibility"
    repro_dir.mkdir(parents=True, exist_ok=True)

    # We track the overall status. If any step fails, we record it but might stop early or continue based on severity.
    # For now, we will strictly halt on failure to match previous subprocess crash behavior natively.
    
    manifest_commands: list[str] = []
    
    # 1. Setup Root (We still use subprocess here as it is a clean utility boundary invoking an explicit argparse CLI)
    setup_cmd = [
        sys.executable,
        "scripts/setup_artifacts_root.py",
        "--target-root",
        str(run_config.target_root_relative),
    ]
    if run_config.clean_root:
        setup_cmd.append("--clean")
    
    manifest_commands.append(" ".join(setup_cmd))
    if not _run_and_log_subprocess(
        setup_cmd, repro_dir / "01_setup_artifacts_root.log", run_config.project_root
    ):
        return 1

    # 2. Pytest (Subprocess - External Boundary)
    if not run_config.skip_pytest:
        pytest_cmd = [sys.executable, "-m", "pytest", "-q"]
        manifest_commands.append(" ".join(pytest_cmd))
        if not _run_and_log_subprocess(
            pytest_cmd, repro_dir / "02_pytest.log", run_config.project_root
        ):
            return 1

    # 3. CLI Run-All Main Pipeline (Subprocess - External Boundary for now as `cli.py` is an independent router)
    cli_cmd = [
        sys.executable,
        "cli.py",
        "run-all",
        "--root",
        str(run_config.target_root_relative),
    ]
    manifest_commands.append(" ".join(cli_cmd))
    if not _run_and_log_subprocess(
        cli_cmd, repro_dir / "03_cli_run_all.log", run_config.project_root
    ):
        return 1

    # 4. Robustness Grid (Native Execution)
    if not run_config.skip_robustness:
        manifest_commands.append("import research.robustness.run_experiment(...)")
        from primary_model.research.robustness import run_experiment as run_robustness
        from primary_model.utils.config_loader import load_merged_config
        config = load_merged_config(
            "base.yaml", Path("configs/experiments/robustness.yaml")
        )

        ok = _run_and_log_native(
            "research.robustness.run_experiment",
            run_robustness,
            repro_dir / "04_robustness.log",
            config=config,
            cli_args=None,
        )
        if not ok:
            return 1

    # 5. Walk-Forward (Native Execution)
    if not run_config.skip_walk_forward:
        manifest_commands.append("import research.walk_forward.run_experiment(...)")
        from primary_model.research.walk_forward import run_experiment as run_walk_forward
        from primary_model.utils.config_loader import load_merged_config
        config = load_merged_config(
            "base.yaml", Path("configs/experiments/walk_forward.yaml")
        )

        ok = _run_and_log_native(
            "research.walk_forward.run_experiment",
            run_walk_forward,
            repro_dir / "05_walk_forward.log",
            config=config,
            cli_args=None,
        )
        if not ok:
            return 1

    # 6. Artifact Compilation
    artifacts: dict[str, str] = {}
    if reports_dir.exists():
        for path in sorted(reports_dir.rglob("*")):
            if path.is_file():
                artifacts[str(path.relative_to(run_config.project_root))] = _sha256(path)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(run_config.project_root),
        "isolated_root": str(target_root),
        "git_commit": _git_commit_sha(run_config.project_root),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "commands": manifest_commands,
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
        f"- Execution Steps: `{len(manifest_commands)}`",
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

    print(f"\nSaved: {manifest_path}")
    print(f"Saved: {summary_path}")
    print("Validation suite completed successfully.")
    
    return 0
