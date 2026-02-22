# primary_model

Professional project scaffold around a **frozen core engine** in `primary_model_unified.py`.

## Project Contract
- `primary_model_unified.py` is the source of truth for pipeline behavior.
- Wrapper modules in `data/`, `signals/`, `portfolio/`, `backtest/`, and `qc/` delegate to that file.
- The goal is modular architecture and clean tooling **without changing core model logic**.

## What This Runs
The unified core supports the following workflows:
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

Or through the installed entrypoint:
```bash
primary-model run-all
```

## Directory Layout
```text
primary_model/
  primary_model_unified.py   # frozen core logic and canonical CLI behavior
  cli.py                     # stable entrypoint delegating to unified main
  data/
    cleaner.py               # cleaning wrappers
    loader.py                # loading/treasury wrappers
  signals/
    variant1.py              # signal construction wrappers
  portfolio/
    weights.py               # allocation wrappers
  backtest/
    engine.py                # backtest/metrics wrappers
    reporting.py             # benchmark/reporting wrappers
  qc/
    reports.py               # data-QC wrappers
  tests/                     # smoke/integration tests
  docs/                      # architecture notes
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

## Notes
- This repo intentionally keeps `primary_model_unified.py` unchanged.
- All package-facing modules are thin adapters to preserve behavior parity.
