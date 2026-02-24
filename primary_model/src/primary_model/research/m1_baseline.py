"""Stage 1 canonical baseline freeze and reproducibility snapshot."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pandas as pd

from primary_model.backtest.reporting import run_primary_variant1
from primary_model.benchmarks.evaluate import run_experiment as run_benchmarks
from primary_model.data.cleaner import prepare_data
from primary_model.qc.reports import run_data_qc

_CANONICAL_ARTIFACTS = (
    "reports/data_qc.html",
    "reports/primary_v1_backtest.csv",
    "reports/primary_v1_summary.csv",
    "reports/primary_v1_signal.csv",
    "reports/primary_v1_weights.csv",
    "reports/benchmarks_summary.csv",
    "reports/benchmarks_summary.html",
)

_METRIC_COLUMNS = (
    "ann_return",
    "ann_vol",
    "sharpe",
    "max_drawdown",
    "calmar",
    "avg_turnover",
)


@dataclass(frozen=True)
class M1BaselineRunConfig:
    root: Path
    out_dir: Path
    spec_id: str
    spec_version: str
    aggregation_mode: str
    validate_determinism: bool
    determinism_runs: int


def _git_commit_sha(project_root: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
            )
            .strip()
        )
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_run_config(
    config: Mapping[str, Any],
    cli_args: Any | None = None,
) -> M1BaselineRunConfig:
    paths_cfg = config.get("paths", {})
    model_cfg = config.get("model", {})
    run_cfg = config.get("run", {})

    root_override = getattr(cli_args, "root", None) if cli_args else None
    out_dir_override = getattr(cli_args, "out_dir", None) if cli_args else None
    runs_override = getattr(cli_args, "determinism_runs", None) if cli_args else None
    disable_check = bool(getattr(cli_args, "no_determinism_check", False)) if cli_args else False

    root = Path(root_override or paths_cfg.get("root", "artifacts")).resolve()

    out_dir_value = out_dir_override if out_dir_override is not None else run_cfg.get("out_dir")
    if out_dir_value is None:
        out_dir = (root / "reports" / "reproducibility").resolve()
    else:
        out_dir_candidate = Path(out_dir_value)
        out_dir = (
            out_dir_candidate.resolve()
            if out_dir_candidate.is_absolute()
            else (root / out_dir_candidate).resolve()
        )

    validate_determinism = bool(run_cfg.get("validate_determinism", True)) and not disable_check
    run_count = int(runs_override or run_cfg.get("determinism_runs", 2))
    if run_count < 1:
        raise ValueError("determinism_runs must be >= 1.")
    if validate_determinism and run_count < 2:
        run_count = 2
    aggregation_mode = str(run_cfg.get("aggregation_mode", "equal_weight")).strip().lower()
    if aggregation_mode not in {"dynamic", "equal_weight"}:
        raise ValueError("run.aggregation_mode must be one of {'dynamic', 'equal_weight'}.")

    return M1BaselineRunConfig(
        root=root,
        out_dir=out_dir,
        spec_id=str(model_cfg.get("canonical_id", "PrimaryV1.1-baseline")),
        spec_version=str(model_cfg.get("version", "1.1")),
        aggregation_mode=aggregation_mode,
        validate_determinism=validate_determinism,
        determinism_runs=run_count if validate_determinism else 1,
    )


def _run_pipeline_once(root: Path, aggregation_mode: str) -> None:
    # Force non-interactive plotting backend for reproducible headless execution.
    os.environ.setdefault("MPLBACKEND", "Agg")

    prepare_data(root)
    run_data_qc(root)
    run_primary_variant1(root, aggregation_mode=aggregation_mode)
    status = run_benchmarks(
        {"run": {"aggregation_mode": aggregation_mode}},
        cli_args=SimpleNamespace(root=root, aggregation_mode=aggregation_mode),
    )
    if status.get("status") != "ok":
        raise RuntimeError("Benchmark pipeline failed while building M1 baseline snapshot.")


def _collect_artifact_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative_path in _CANONICAL_ARTIFACTS:
        target = root / relative_path
        if not target.exists():
            raise FileNotFoundError(f"Missing canonical artifact: {target}")
        hashes[relative_path] = _sha256(target)
    return hashes


def _load_primary_metrics(root: Path) -> dict[str, Any]:
    summary_path = root / "reports" / "primary_v1_summary.csv"
    summary = pd.read_csv(summary_path, index_col=0)

    strategy = "PrimaryV1" if "PrimaryV1" in summary.index else str(summary.index[0])
    row = summary.loc[strategy]
    perf = {col: float(row[col]) for col in _METRIC_COLUMNS if col in row.index}

    signal_path = root / "reports" / "primary_v1_signal.csv"
    counts = {"BUY": 0, "HOLD": 0, "SELL": 0}
    if signal_path.exists():
        signal_series = pd.read_csv(signal_path)["signal"]
        value_counts = signal_series.value_counts(dropna=False)
        for state in counts:
            counts[state] = int(value_counts.get(state, 0))

    return {
        "strategy": strategy,
        "performance": perf,
        "signal_counts": counts,
    }


def _write_snapshot_markdown(snapshot: Mapping[str, Any], path: Path) -> None:
    det = snapshot["determinism"]
    metrics = snapshot["key_metrics"]["performance"]
    signal_counts = snapshot["key_metrics"]["signal_counts"]

    lines = [
        "# M1 Baseline Snapshot",
        "",
        f"- Spec ID: `{snapshot['spec']['id']}`",
        f"- Spec Version: `{snapshot['spec']['version']}`",
        f"- Root: `{snapshot['root']}`",
        f"- Git commit: `{snapshot['git_commit']}`",
        f"- Determinism check enabled: `{det['enabled']}`",
        f"- Determinism runs: `{det['runs']}`",
        f"- Deterministic artifacts: `{det['deterministic']}`",
        f"- Aggregation mode: `{snapshot['spec']['aggregation_mode']}`",
        "",
        "## PrimaryV1 Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in _METRIC_COLUMNS:
        if key in metrics:
            lines.append(f"| `{key}` | `{metrics[key]:.10f}` |")

    lines.extend(
        [
            "",
            "## PrimaryV1 Signal Counts",
            "",
            "| Signal | Count |",
            "|---|---:|",
            f"| `BUY` | `{signal_counts['BUY']}` |",
            f"| `HOLD` | `{signal_counts['HOLD']}` |",
            f"| `SELL` | `{signal_counts['SELL']}` |",
            "",
            "## Canonical Artifact Hashes",
            "",
            "| Artifact | SHA256 |",
            "|---|---|",
        ]
    )

    for rel_path, digest in snapshot["artifact_hashes"].items():
        lines.append(f"| `{rel_path}` | `{digest}` |")

    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    config: dict[str, Any],
    cli_args: Any = None,
) -> dict[str, Any]:
    """Run frozen M1 baseline workflow and write reproducibility snapshot."""
    run_config = _resolve_run_config(config=config, cli_args=cli_args)
    run_config.out_dir.mkdir(parents=True, exist_ok=True)

    hash_runs: list[dict[str, str]] = []
    for _ in range(run_config.determinism_runs):
        _run_pipeline_once(run_config.root, aggregation_mode=run_config.aggregation_mode)
        hash_runs.append(_collect_artifact_hashes(run_config.root))

    deterministic = True
    if run_config.validate_determinism and len(hash_runs) > 1:
        first = hash_runs[0]
        deterministic = all(run_hash == first for run_hash in hash_runs[1:])

    project_root = Path(__file__).resolve().parents[3]
    snapshot = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok" if deterministic else "failed",
        "spec": {
            "id": run_config.spec_id,
            "version": run_config.spec_version,
            "aggregation_mode": run_config.aggregation_mode,
        },
        "root": str(run_config.root),
        "git_commit": _git_commit_sha(project_root),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "determinism": {
            "enabled": run_config.validate_determinism,
            "runs": run_config.determinism_runs,
            "deterministic": deterministic,
        },
        "key_metrics": _load_primary_metrics(run_config.root),
        "artifact_hashes": hash_runs[0],
        "artifact_hashes_by_run": hash_runs,
    }

    json_path = run_config.out_dir / "m1_baseline_snapshot.json"
    md_path = run_config.out_dir / "m1_baseline_snapshot.md"
    json_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    _write_snapshot_markdown(snapshot, md_path)

    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")
    if run_config.validate_determinism:
        print(f"Determinism check: {deterministic} across {run_config.determinism_runs} run(s).")

    return {
        "status": snapshot["status"],
        "deterministic": deterministic,
        "out_dir": str(run_config.out_dir),
        "snapshot_json": str(json_path),
        "snapshot_md": str(md_path),
        "artifact_count": len(snapshot["artifact_hashes"]),
    }


__all__ = [
    "M1BaselineRunConfig",
    "run_experiment",
]
