# Meta-Labeling Project

Practical quant finance repository for a rule-based primary model that allocates across a small macro multi-asset universe and lays the groundwork for future meta-labeling research.

## Quick Start

If a teammate clones this repository into any local folder, the shortest reliable setup is:

```powershell
cd "path\to\meta-labeling-project"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m metalabel.cli run-all
```

Before running `run-all`, make sure the raw input files are present in `data/raw/` with these exact names:

- `spx.xlsx`
- `bcom.xlsx`
- `treasury_10y.xlsx`
- `corp_bonds.xlsx`

`requirements.txt` now installs the project itself in editable mode through `-e .`, so the one install command above is enough for a clean environment.

## What This Project Does

This repository implements a fully reproducible primary model pipeline for:

- cleaning Bloomberg-style raw Excel files into canonical CSV data
- adjusting 10-year Treasury yield data into a duration-based bond total return proxy
- building a macro composite signal from expanding z-scored indicators
- mapping the signal into long-only portfolio weights
- backtesting the resulting strategy
- benchmarking the strategy against simple allocation baselines
- running robustness, walk-forward validation, and monthly cross-validation workflows

The logic is preserved from the original monolithic `primary_model_0224_fixed.py`, but split into smaller modules so the repository is easier to read, test, and extend.

## Role Of The Primary Model

The primary model is the current implemented strategy engine. It:

- constructs macro indicators from SPX, BCOM, 10-year Treasury, and corporate bond series
- generates a `BUY` / `HOLD` / `SELL` regime label
- converts those labels into portfolio weights
- evaluates performance and classification-style diagnostics

This is the live research baseline. Any future meta-labeling work should sit on top of this primary signal framework rather than replace it.

## Secondary Model Workspace

`src/metalabel/secondary/` is the workspace for ongoing and future meta-labeling development. The current implemented strategy logic still lives in the primary-model modules.

## Repository Layout

```text
meta-labeling-project/
├── README.md
├── requirements.txt
├── .gitignore
├── pyproject.toml
├── data/
│   ├── raw/
│   └── clean/
├── configs/
│   └── primary.yaml
├── src/
│   └── metalabel/
│       ├── __init__.py
│       ├── cli.py
│       ├── data.py
│       ├── reporting.py
│       ├── validation.py
│       ├── primary/
│       │   ├── __init__.py
│       │   ├── signals.py
│       │   ├── portfolio.py
│       │   ├── backtest.py
│       │   ├── metrics.py
│       │   └── pipeline.py
│       └── secondary/
│           └── __init__.py
├── scripts/
│   ├── prepare_data.py
│   ├── run_primary.py
│   ├── run_benchmarks.py
│   ├── run_robustness.py
│   └── run_walk_forward.py
├── tests/
│   ├── test_data.py
│   └── primary/
│       ├── test_signals.py
│       ├── test_portfolio.py
│       └── test_backtest.py
└── reports/
    ├── assets/
    └── results/
```

## Setup

Use a fresh virtual environment and install from `requirements.txt`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

That single install command does two things:

- installs the external dependencies declared by the project
- installs the `metalabel` package in editable mode so `python -m metalabel.cli ...` works

Place the raw Bloomberg-style input files in `data/raw/` using these filenames:

- `spx.xlsx`
- `bcom.xlsx`
- `treasury_10y.xlsx`
- `corp_bonds.xlsx`

## Developer Workflow

### 1. Install

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Validate The Environment

Run these before making larger changes and after finishing code changes:

```powershell
python -m metalabel.cli run-self-tests
python -m pytest
```

### 3. Expected Data Files

The project expects these raw files in `data/raw/`:

- `spx.xlsx`
- `bcom.xlsx`
- `treasury_10y.xlsx`
- `corp_bonds.xlsx`

Important data assumption:

- `treasury_10y.xlsx` is treated as a yield-level series, not a bond total return index.
- The project converts it into a duration-based Treasury total return proxy during processing.

### 4. Main Run Commands

Standard full workflow:

```powershell
python -m metalabel.cli run-all
```

Common command set:

```powershell
python -m metalabel.cli prepare-data
python -m metalabel.cli data-qc
python -m metalabel.cli run-primary-v1
python -m metalabel.cli run-benchmarks
python -m metalabel.cli run-robustness
python -m metalabel.cli run-walk-forward
python -m metalabel.cli run-monthly-cv
python -m metalabel.cli run-validation-suite --root test --clean-root
python -m metalabel.cli run-self-tests
```

### 5. Where Outputs Go

- Raw source files: `data/raw/`
- Cleaned canonical files: `data/clean/`
- Plots: `reports/assets/`
- CSV, HTML, markdown, and validation outputs: `reports/results/`

The repository is configured so collaborators can keep the cleaned data and generated outputs in version control if they want to inspect current results directly.

## Project Conventions

- Do not change strategy logic without documenting the change and rerunning the affected outputs.
- Keep the main research and runtime settings in `configs/primary.yaml`.
- Preserve canonical CLI command names and canonical output file names unless there is an explicit repository-interface change.
- Use `python -m metalabel.cli ...` as the default command interface.
- Use commit messages in imperative mood with the format `<verb> <area> <change>`.
- Good examples:
  - `Add data contract documentation`
  - `Update benchmark reporting outputs`
  - `Change primary threshold logic`
- Keep one main change per commit when practical. If a commit changes model logic, say that explicitly in the commit message.
- Validate normal code changes with:

```powershell
python -m metalabel.cli run-self-tests
python -m pytest
```

- If strategy logic, reporting, or validation workflows change, rerun the relevant workflows and commit the refreshed tracked outputs.

## Configuration

The most important research parameters live in `configs/primary.yaml`, including:

- treasury duration assumption
- signal thresholds
- trend and relative windows
- minimum training history for validation runs
- robustness grids

The config is intentionally small. It is there to expose the key knobs without turning the project into a large configuration framework.

## How To Run

The main entry point is:

```powershell
python -m metalabel.cli <command>
```

If you just want the full standard workflow, run:

```powershell
python -m metalabel.cli run-all
```

That command runs, in order:

1. raw-to-clean data preparation
2. data quality checks
3. the primary strategy pipeline
4. benchmark generation

If the environment is not set up yet, the project will not install dependencies automatically at runtime. The setup section above must be completed first.

### Prepare data

```powershell
python -m metalabel.cli prepare-data
```

Or:

```powershell
python scripts/prepare_data.py
```

### Run the primary pipeline

```powershell
python -m metalabel.cli run-primary-v1
```

Or:

```powershell
python scripts/run_primary.py
```

Outputs are written to `reports/results/`.

### Run benchmarks

```powershell
python -m metalabel.cli run-benchmarks
```

Or:

```powershell
python scripts/run_benchmarks.py
```

Plots are written to `reports/assets/`. Tables and CSV outputs are written to `reports/results/`.

### Run robustness

```powershell
python -m metalabel.cli run-robustness
```

Or:

```powershell
python scripts/run_robustness.py
```

### Run walk-forward validation

```powershell
python -m metalabel.cli run-walk-forward
```

Or:

```powershell
python scripts/run_walk_forward.py
```

### Additional validation workflows

```powershell
python -m metalabel.cli data-qc
python -m metalabel.cli run-monthly-cv
python -m metalabel.cli run-validation-suite --root test --clean-root
python -m metalabel.cli run-self-tests
```

## Results Contract

Treat the following outputs as the canonical shared artifacts of the repository:

- Clean data:
  - `data/clean/spx.csv`
  - `data/clean/bcom.csv`
  - `data/clean/treasury_10y.csv`
  - `data/clean/corp_bonds.csv`
- Core primary-model outputs:
  - `reports/results/primary_v1_backtest.csv`
  - `reports/results/primary_v1_summary.csv`
  - `reports/results/primary_v1_classification.csv`
  - `reports/results/primary_v1_signal.csv`
  - `reports/results/primary_v1_weights.csv`
- Benchmark outputs:
  - `reports/results/benchmarks_summary.csv`
  - `reports/results/benchmarks_summary.html`
  - `reports/assets/equity_curves.png`
  - `reports/assets/drawdowns.png`
  - `reports/assets/rolling_sharpe.png`
- Data QC output:
  - `reports/results/data_qc.html`
- Robustness outputs:
  - `reports/results/robustness/robustness_grid_results.csv`
  - `reports/results/robustness/baseline_cost_sensitivity.csv`
  - `reports/results/robustness/top15_by_sharpe.csv`
  - `reports/results/robustness/robustness_summary.md`
- Walk-forward outputs:
  - `reports/results/walk_forward/walk_forward_backtest.csv`
  - `reports/results/walk_forward/standard_causal_backtest_slice.csv`
  - `reports/results/walk_forward/walk_forward_summary.csv`
  - `reports/results/walk_forward/walk_forward_protocol.md`
- Monthly cross-validation outputs:
  - `reports/results/monthly_cv/monthly_cv_fold_summary.csv`
  - `reports/results/monthly_cv/monthly_cv_oos_backtest.csv`
  - `reports/results/monthly_cv/monthly_cv_summary.csv`
  - `reports/results/monthly_cv/monthly_cv_protocol.md`

These filenames and locations are part of the repository contract for collaborators and agents. Avoid renaming them casually.

## Replication Checklist

For a teammate to reproduce the project successfully on a new machine, they need:

1. Python 3.11+ installed.
2. This repository copied or cloned locally.
3. The four required raw Excel files placed in `data/raw/`.
4. A virtual environment created.
5. `python -m pip install -r requirements.txt` run once inside that environment.
6. Run `python -m metalabel.cli run-self-tests` to confirm setup before longer workflows.

After that, the project can be run with the commands in this README.

## Notes

- Repo-level agent and collaborator instructions live in `AGENTS.md`.
- Raw input requirements and data assumptions are documented in `data/raw/README.md`.
- `reports/results/` contains generated CSV, HTML, markdown, and reproducibility outputs.
- `reports/assets/` contains generated figures.
- `scikit-learn` is included so AUC metrics are available during classification diagnostics.
