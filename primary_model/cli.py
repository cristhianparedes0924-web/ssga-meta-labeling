"""Stable command-line entrypoint.

This module delegates all behavior to ``primary_model_unified.main`` so the
project can expose a clean CLI without modifying the frozen core script.
"""

from primary_model_unified import main as _core_main


def main() -> None:
    """Run the unified core CLI unchanged."""
    _core_main()


if __name__ == "__main__":
    main()
