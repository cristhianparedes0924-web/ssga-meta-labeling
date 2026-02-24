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
- Fast development loop (recommended for iteration): `make dev-loop`
- Fast validation profile (recommended default for AI agents): `make validate-fast`
- Full strict validation (for milestone/final checks only): `make full-validate`
  - Runs strict validation suite with clean root.
- Main pipeline only: `python cli.py run-all --root artifacts`
- Primary aggregation A/B:
  - Default baseline is `equal_weight`.
  - Compare modes explicitly: `python cli.py run-all --root artifacts --aggregation-mode dynamic` vs `--aggregation-mode equal_weight`.

## Agent Execution Policy (Critical)
- If the user says "run the whole project" but does not explicitly require strict/final validation, run `make validate-fast` first.
- Only run strict full validation when the user explicitly asks for strict/final/full verification:
  - `python cli.py run-validation-suite --root artifacts --clean-root`
- Do not run redundant heavy commands in sequence (example of what to avoid): running `run-all`, then `run-robustness`, then `run-walk-forward`, and then `run-validation-suite` again.
- Prefer one orchestrator command per request unless the user asks for additional stage artifacts.
- If Stage 2-5 artifacts are explicitly required after a clean strict run, generate them in order:
  1. `python cli.py run-m1-baseline --root artifacts --config configs/experiments/m1_canonical_v1_1.yaml`
  2. `python cli.py run-signal-validation --root artifacts --config configs/experiments/signal_validation.yaml`
  3. `python cli.py run-ablations --root artifacts --config configs/experiments/ablations.yaml`
  4. `python cli.py run-decision-diagnostics --root artifacts --config configs/experiments/decision_diagnostics.yaml`
  5. `python cli.py run-m1-readiness --root artifacts --config configs/experiments/m1_readiness.yaml`

## Runtime Profiles
- Fast path (default): `make validate-fast`
  - Typical runtime: ~30-90 seconds.
- Dev research loop: `make dev-loop`
  - Typical runtime: ~30-120 seconds depending on options.
- Strict path: `make full-validate` (or `python cli.py run-validation-suite --root artifacts --clean-root`)
  - Typical runtime: ~8-15+ minutes.

## Environment Performance Notes
- Prefer Linux Python inside WSL for speed.
- Avoid running Windows Python against `\\wsl.localhost\...` paths when possible; this adds filesystem overhead.
- Heavy runtime is driven mainly by config/workflow choices (strict walk-forward, full robustness grid, high bootstrap, determinism checks), not by dormant legacy files.

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
