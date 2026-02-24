"""
File writing utilities to separate IO concerns from orchestration maths.
"""

from pathlib import Path
from typing import Any

import pandas as pd


def write_dataframe(df: pd.DataFrame, path: Path, index: bool = True) -> None:
    """Save a DataFrame strictly to disk, ensuring parents exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    print(f"Saved: {path}")


def write_markdown_protocol(lines: list[str], path: Path) -> None:
    """Save a markdown protocol or summary report strictly to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {path}")
