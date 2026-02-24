# SSGA Meta Labeling - Primary Model

Standalone primary-model research and backtesting pipeline.

## Quick Start
Please see:
- `INSTRUCTIONS.txt` for onboarding/setup guidance.
- `docs/operator_runbook.md` for the canonical execution and validation flow.

Recommended full validation command:
```bash
make full-validate
```

## Architecture Documentation
For deeper architectural details, package definitions, and design decisions, refer to `docs/architecture.md`.

## Quality Checks
To ensure structural integrity and mathematical correctness, simply run:
```bash
python -m pytest -q
```
