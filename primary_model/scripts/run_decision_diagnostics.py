#!/usr/bin/env python3
"""Compatibility wrapper for `cli.py run-decision-diagnostics`."""

from __future__ import annotations

from typing import Sequence

try:
    from ._shim import parse_args_for, run_command
except ImportError:  # pragma: no cover - direct script execution path
    from _shim import parse_args_for, run_command

COMMAND = "run-decision-diagnostics"


def parse_args(argv: Sequence[str] | None = None):
    return parse_args_for(COMMAND, argv)


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(COMMAND, argv)


if __name__ == "__main__":
    raise SystemExit(main())
