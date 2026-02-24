"""Lightweight YAML configuration loader with deep merge support."""
from pathlib import Path
from typing import Any
import yaml

from primary_model.utils.paths import CONFIGS_ROOT

def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries. Override takes precedence."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged

def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file safely."""
    if not path.exists():
        raise FileNotFoundError(f"YAML configuration not found at {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
            return config if config is not None else {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML file at {path}: {e}")

def load_merged_config(base_name: str = "base.yaml", experiment_path: Path | None = None) -> dict[str, Any]:
    """Load base config and merge with experiment config if provided."""
    base_file = CONFIGS_ROOT / base_name
    
    # Load base config, or empty if it doesn't exist
    if base_file.exists():
        config = load_yaml(base_file)
    else:
        config = {}
        
    # Load and merge experiment config
    if experiment_path and experiment_path.exists():
        exp_config = load_yaml(experiment_path)
        config = deep_merge_dicts(config, exp_config)
        
    return config
