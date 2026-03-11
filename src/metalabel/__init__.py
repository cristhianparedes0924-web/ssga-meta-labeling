"""Project package metadata, paths, and lightweight configuration loading."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


__version__ = "0.1.0"

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_ASSETS_DIR = REPORTS_DIR / "assets"
REPORTS_RESULTS_DIR = REPORTS_DIR / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_PRIMARY_CONFIG_PATH = CONFIGS_DIR / "primary.yaml"

DEFAULT_PRIMARY_CONFIG: dict[str, Any] = {
    "primary": {
        "trend_window": 6,
        "relative_window": 3,
        "zscore_min_periods": 12,
        "duration": 8.5,
        "buy_threshold": 0.31,
        "sell_threshold": -0.31,
        "tcost_bps": 0.0,
        "benchmark_key": "EqualWeight25",
    },
    "validation": {
        "min_train_periods": 120,
        "robustness": {
            "tcost_grid_bps": "0,5,10,25,50",
            "buy_grid": "0.31,0.25,0.50",
            "sell_grid": "-0.31,-0.25,-0.50",
            "duration_grid": "6.0,8.5,10.0",
        },
    },
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_primary_config(path: Path | None = None) -> dict[str, Any]:
    """Load the small primary-model configuration file with sensible defaults."""
    config = deepcopy(DEFAULT_PRIMARY_CONFIG)
    config_path = DEFAULT_PRIMARY_CONFIG_PATH if path is None else Path(path)
    if not config_path.exists():
        return config

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return _deep_update(config, loaded)


__all__ = [
    "CONFIGS_DIR",
    "DATA_DIR",
    "DEFAULT_PRIMARY_CONFIG",
    "DEFAULT_PRIMARY_CONFIG_PATH",
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "REPORTS_ASSETS_DIR",
    "REPORTS_DIR",
    "REPORTS_RESULTS_DIR",
    "SRC_ROOT",
    "__version__",
    "load_primary_config",
]
