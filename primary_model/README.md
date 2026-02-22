# primary_model

Standalone primary-model research/backtesting pipeline.

## What This Runs
The project supports these workflows:
- `prepare-data`
- `data-qc`
- `run-primary-v1`
- `run-benchmarks`
- `run-all`

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run commands via CLI:
```bash
python cli.py prepare-data
python cli.py data-qc
python cli.py run-primary-v1
python cli.py run-benchmarks
python cli.py run-all
```

Run immediately with bundled sample data/results root:
```bash
python cli.py run-all --root project_results
```

Validation workflows in an isolated test root:
```bash
python scripts/run_validation_suite.py --root test --clean-root
```

Manual isolated steps:
```bash
python scripts/setup_test_root.py --target-root test --clean
pytest -q
python cli.py run-all --root test
python scripts/run_robustness.py --root test --out-dir test/reports/robustness
python scripts/run_walk_forward.py --root test --out-dir test/reports/walk_forward
```

Detailed replication guide:
```text
INSTRUCTIONS.txt
```

Or through the installed entrypoint:
```bash
primary-model run-all
```

## Directory Layout
```text
primary_model/
  cli.py
  data/
    cleaner.py
    loader.py
  signals/
    variant1.py
  portfolio/
    weights.py
  backtest/
    engine.py
    reporting.py
  qc/
    reports.py
  tests/
  docs/
```

## Data Expectations
- Raw files are expected in `data/raw/`:
  - `spx.xlsx`
  - `bcom.xlsx`
  - `treasury_10y.xlsx`
  - `corp_bonds.xlsx`
- Cleaned CSVs are generated into `data/clean/`.
- Reports and plots are generated into `reports/`.

## Development
Install dev tools:
```bash
pip install -e .[dev]
```

Run quality checks:
```bash
ruff check .
pytest
```
