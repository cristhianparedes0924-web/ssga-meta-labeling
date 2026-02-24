# Repository Guidelines

## Project Structure & Module Organization
- Core code lives in `src/primary_model/`:
  - `data/` ingestion and cleaning
  - `signals/` signal logic
  - `portfolio/` weight construction
  - `backtest/` execution/reporting
  - `benchmarks/` benchmark experiments
  - `research/` robustness and walk-forward studies
  - `utils/` shared helpers
- CLI entrypoints: `cli.py` (main workflows) and `scripts/run_*.py` (standalone experiment wrappers).
- Configuration: `configs/base.yaml` and `configs/experiments/*.yaml`.
- Tests: `tests/` with `pytest` suites.
- Generated outputs: `artifacts/` (`data/raw`, `data/clean`, `reports/`). Treat this as runtime output, not business logic.

## Build, Test, and Development Commands
- Install package: `python -m pip install -e .`
- Install with dev tools: `python -m pip install -e .[dev]`
- Lint: `ruff check .`
- Tests: `pytest` (or `python -m pytest -q`)
- Full validation (recommended): `make full-validate`
  - Runs tests, full pipeline, robustness, walk-forward, and validation suite.
- Main pipeline only: `python cli.py run-all --root artifacts`

## Coding Style & Naming Conventions
- Target Python `>=3.10`; use 4-space indentation.
- Follow `ruff` defaults and keep code lint-clean before opening a PR.
- Use type hints for public functions and orchestration boundaries.
- Naming:
  - modules/functions/variables: `snake_case`
  - classes: `PascalCase`
  - constants: `UPPER_SNAKE_CASE`
- Keep `scripts/` thin; put reusable logic in `src/primary_model/`.

## Testing Guidelines
- Framework: `pytest` (`tests/`, configured in `pyproject.toml`).
- File/function naming: `test_*.py`, `test_*`.
- Add/adjust tests for each behavior change, especially around:
  - temporal integrity
  - benchmark calculations
  - CLI/script entrypoints
  - artifact generation paths
- Prefer `tmp_path` fixtures for filesystem tests; avoid hard-coded local paths.

## Commit & Pull Request Guidelines
- Commit messages in this repo are short, imperative, and scope-focused (examples: `Add ...`, `Refactor ...`, `feat: ...`).
- Prefer one logical change per commit; keep subject lines concise.
- PRs should include:
  - what changed and why
  - impacted modules/configs
  - exact validation commands run (for example, `ruff check .`, `pytest`, `make full-validate`)
  - sample output paths when reports/artifacts are affected
- Ensure CI passes (`ruff` + `pytest` on Python 3.10/3.11).
