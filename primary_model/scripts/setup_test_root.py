#!/usr/bin/env python3
"""Bootstrap an isolated test root with bundled raw input files."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare isolated test root inputs.")
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("test"),
        help="Target isolated root to populate (default: test).",
    )
    parser.add_argument(
        "--source-raw",
        type=Path,
        default=Path("project_results/data/raw"),
        help="Source raw-data directory with Excel files.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing target data/clean and reports before setup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_root = (PROJECT_ROOT / args.target_root).resolve()
    source_raw = (PROJECT_ROOT / args.source_raw).resolve()

    if not source_raw.exists():
        raise FileNotFoundError(f"Source raw directory not found: {source_raw}")

    target_raw = target_root / "data" / "raw"
    target_clean = target_root / "data" / "clean"
    target_reports = target_root / "reports"

    if args.clean:
        if target_clean.exists():
            shutil.rmtree(target_clean)
        if target_reports.exists():
            shutil.rmtree(target_reports)

    target_raw.mkdir(parents=True, exist_ok=True)
    target_clean.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for path in sorted(source_raw.iterdir()):
        if path.is_file():
            dst = target_raw / path.name
            shutil.copy2(path, dst)
            copied.append(dst)

    print(f"Target root: {target_root}")
    print(f"Copied {len(copied)} raw files to: {target_raw}")
    for path in copied:
        print(f"- {path}")


if __name__ == "__main__":
    main()
