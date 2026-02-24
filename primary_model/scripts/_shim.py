"""Shared compatibility shim for script wrappers."""
# ruff: noqa: E402

from __future__ import annotations

import sys
from typing import Sequence

try:
    from ._bootstrap import ensure_project_on_path
except ImportError:  # pragma: no cover - direct script execution path
    from _bootstrap import ensure_project_on_path


ensure_project_on_path()

from cli import build_parser, main as cli_main


def parse_args_for(command: str, argv: Sequence[str] | None = None):
    parser = build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    return parser.parse_args([command, *raw_argv])


def run_command(command: str, argv: Sequence[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    return cli_main([command, *raw_argv])
