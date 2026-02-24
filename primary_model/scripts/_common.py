"""Shared logic for CLI scripts."""
# ruff: noqa: E402

import argparse
from pathlib import Path

try:
    from ._bootstrap import ensure_project_on_path
except ImportError:  # pragma: no cover - direct script execution path
    from _bootstrap import ensure_project_on_path

ensure_project_on_path()

from primary_model.utils.paths import get_artifacts_root
from primary_model.utils.config_loader import load_merged_config

def get_config_parser_args(
    default_config_path: str,
) -> tuple[argparse.ArgumentParser, Path, dict, dict]:
    """
    Standardize the --config loading step for execution scripts.
    Returns: (base_parser, default_root, paths_cfg, run_cfg)
    """
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--config",
        type=Path,
        default=Path(default_config_path),
        help="Path to experiment config YAML.",
    )
    known_args, _ = base_parser.parse_known_args()

    config = load_merged_config("base.yaml", known_args.config)
    paths_cfg = config.get("paths", {})
    run_cfg = config.get("run", {})
    default_root = Path(paths_cfg.get("root", get_artifacts_root()))

    return base_parser, default_root, paths_cfg, run_cfg
