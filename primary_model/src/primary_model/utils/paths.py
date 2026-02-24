"""Centralized path resolution for the project."""
from pathlib import Path

# Resolve project root dynamically by finding pyproject.toml
def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parent.parent.parent.parent

PROJECT_ROOT = _find_project_root()

ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
CONFIGS_ROOT = PROJECT_ROOT / "configs"
DATA_ROOT = PROJECT_ROOT / "data"

def get_project_root() -> Path:
    return PROJECT_ROOT

def get_artifacts_root() -> Path:
    return ARTIFACTS_ROOT

def resolve_relative(path_str: str) -> Path:
    """Resolve a path relative to the project root."""
    return PROJECT_ROOT / path_str
