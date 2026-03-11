"""Thin wrapper for data preparation."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from metalabel import PROJECT_ROOT as REPO_ROOT
from metalabel.data import prepare_data


def main() -> None:
    prepare_data(REPO_ROOT)


if __name__ == "__main__":
    main()
