# File-by-File Technical Documentation

This document explains what each project file does, what it depends on, and what depends on it.

Companion diagrams:
- `docs/sequence_diagrams.md` for step-by-step execution flows in `static` and `walk_forward` modes.
- `docs/runbook.md` for operational commands, expected artifacts, and troubleshooting.

## 1) End-to-End Architecture

Primary execution flow:

`data/*.xlsx` -> `src/data_loader.py` -> `src/features.py` -> `src/evaluation.py` -> (`scripts/run_evaluation.py` or `notebooks/02_unbiased_evaluation.ipynb`) -> `reports/*.json` + `reports/*.csv`

Exploratory path:

`data/*.xlsx` -> `src/data_loader.py` -> `src/features.py` -> `notebooks/01_signals_analysis.ipynb` -> `data/final_backtest_results.csv`

Quality/validation flow:

`tests/test_data_loader.py` -> validates `src/data_loader.py`  
`tests/test_evaluation_policies.py` -> validates `src/evaluation.py`  
`tests/test_cli_run_evaluation.py` -> validates `scripts/run_evaluation.py`  
`.github/workflows/tests.yml` -> runs all tests in CI on push/PR

---

## 2) Root and Configuration Files

### `.gitignore`
- Purpose: Prevents generated files and environment artifacts from being committed by default.
- Relationship:
- Affects all folders, especially notebooks checkpoints, caches, and local Python artifacts.
- Helps keep Git history focused on source code, docs, and intentional artifacts.

### `README.md`
- Purpose: Main entrypoint documentation (overview, usage, anti-leakage guardrail, CLI examples).
- Relationship:
- Describes how to run `scripts/run_evaluation.py`.
- References notebooks in `notebooks/`.
- Serves as top-level index for new contributors.

### `requirements.txt`
- Purpose: Defines Python runtime dependencies.
- Relationship:
- Used by local setup (`pip install -r requirements.txt`).
- Used by CI in `.github/workflows/tests.yml`.
- Covers libraries used by `src/`, `scripts/`, and notebooks.

### `.github/workflows/tests.yml`
- Purpose: CI pipeline for automated testing.
- Relationship:
- Installs `requirements.txt`.
- Executes `python -m unittest discover -s tests -v`.
- Gatekeeper for code quality on `push` and `pull_request`.

---

## 3) Source Code (`src/`)

### `src/data_loader.py`
- Purpose: Strict, fail-fast ingestion and merge of required Bloomberg monthly files.
- Main responsibilities:
- Defines required source schema (`Date`, `PX_LAST`) and expected merged output columns.
- Validates file existence, read errors, schema, type coercion, duplicates, and merged nulls.
- Exposes `load_data(data_dir=None)` as the canonical data entrypoint.
- Upstream dependencies:
- Reads files from `data/` (`bcom.xlsx`, `spx.xlsx`, `treasury_10y.xlsx`, `corp_bonds.xlsx`).
- Downstream dependents:
- `src/evaluation.py` calls `load_data()` in both static and walk-forward evaluators.
- `notebooks/01_signals_analysis.ipynb` also uses `load_data()`.
- `tests/test_data_loader.py` validates success and failure contracts.

### `src/features.py`
- Purpose: Feature engineering for the five SSGA-style signals.
- Main responsibilities:
- Implements `create_indicators(df)`.
- Generates:
- `Z1_Mom` (12M momentum)
- `Z2_Value` (yield minus inflation proxy)
- `Z3_Carry` (credit minus treasury carry proxy)
- `Z5_Vol` (negative rolling vol)
- `Z4_Trend` (price vs moving average)
- Drops warm-up NaN rows from rolling windows.
- Upstream dependencies:
- Consumes cleaned aligned prices from `src/data_loader.py`.
- Downstream dependents:
- `src/evaluation.py` uses these features for model training and prediction.
- `notebooks/01_signals_analysis.ipynb` uses this for exploratory analysis.

### `src/evaluation.py`
- Purpose: Core unbiased evaluation engine (static split + walk-forward retraining).
- Main responsibilities:
- Dataset construction for long-only meta-labeling:
- Primary signal (`Trade_Signal`) and target (`Meta_Label`).
- Model training and probability generation:
- Random forest with optional probability calibration (`none`, `sigmoid`, `isotonic`).
- Decision policies:
- `threshold` policy (probability >= selected threshold)
- `utility` policy (utility score >= utility margin)
- Backtest metrics and diagnostics:
- returns, CAGR, Sharpe, drawdown, turnover, costs, trade logs.
- Orchestration entrypoints:
- `run_evaluation(...)` for static mode.
- `run_walk_forward_evaluation(...)` for rolling mode.
- Artifact persistence:
- `save_report(...)` and `save_test_trade_log(...)`.
- Upstream dependencies:
- Calls `load_data()` from `src/data_loader.py`.
- Calls `create_indicators()` from `src/features.py`.
- Downstream dependents:
- `scripts/run_evaluation.py`
- `notebooks/02_unbiased_evaluation.ipynb`
- `tests/test_evaluation_policies.py`

---

## 4) Script Entry Point (`scripts/`)

### `scripts/run_evaluation.py`
- Purpose: CLI wrapper for reproducible evaluations and report generation.
- Main responsibilities:
- Parses arguments for mode, split config, threshold search, transaction costs, calibration, and utility policy controls.
- Calls:
- `run_evaluation(...)` for `--mode static`
- `run_walk_forward_evaluation(...)` for `--mode walk_forward`
- Saves output via `save_report(...)` and `save_test_trade_log(...)`.
- Prints concise run summary (policy-aware).
- Upstream dependencies:
- Uses `src/evaluation.py` APIs.
- Downstream outputs:
- Writes JSON and CSV artifacts in `reports/`.
- Used in manual workflows and can be used in scheduled jobs.
- Tested by:
- `tests/test_cli_run_evaluation.py`.

---

## 5) Notebooks (`notebooks/`)

### `notebooks/01_signals_analysis.ipynb`
- Purpose: Exploratory analysis and hypothesis development.
- Main responsibilities:
- Demonstrates data loading and indicator generation.
- Explores strategy behavior and backtest intuition.
- Exports a summarized result file:
- `data/final_backtest_results.csv`.
- Relationship notes:
- Useful for research and feature intuition.
- Not the strict, final unbiased evaluation path for performance claims.

### `notebooks/02_unbiased_evaluation.ipynb`
- Purpose: Notebook-form reproducible evaluation aligned with the script engine.
- Main responsibilities:
- Calls `run_evaluation(...)` and `run_walk_forward_evaluation(...)`.
- Supports current options:
- `DECISION_POLICY` (`threshold`/`utility`)
- `PROBABILITY_CALIBRATION`
- `UTILITY_MARGIN`
- `UTILITY_RISK_AVERSION`
- Plots cumulative performance and policy-specific diagnostics.
- Saves notebook-specific artifacts to `reports/` with `_from_notebook` suffixes.
- Relationship notes:
- Preferred notebook for understanding production evaluation behavior.

---

## 6) Tests (`tests/`)

### `tests/test_data_loader.py`
- Purpose: Regression coverage for ingestion and schema safety.
- Verifies:
- Successful load contract.
- Missing file handling.
- Missing required columns.
- Duplicate date rejection.
- Misaligned merge rejection.
- Targets:
- `src/data_loader.py`.

### `tests/test_evaluation_policies.py`
- Purpose: Regression coverage for evaluator policy behavior.
- Verifies:
- `evaluate_period(...)` contracts for threshold vs utility.
- Required-argument enforcement by decision policy.
- `run_evaluation(...)` static policy-specific outputs.
- `run_walk_forward_evaluation(...)` walk-forward policy-specific outputs.
- Targets:
- `src/evaluation.py`.

### `tests/test_cli_run_evaluation.py`
- Purpose: CLI argument and forwarding contract tests.
- Verifies:
- Invalid CLI choices are rejected.
- Static mode arguments are forwarded correctly.
- Walk-forward utility mode arguments are forwarded correctly.
- Policy-specific summary output text is emitted.
- Targets:
- `scripts/run_evaluation.py`.

---

## 7) Input Data Files (`data/`)

### `data/bcom.xlsx`
- Monthly commodity proxy input.
- Loaded by `src/data_loader.py` as `BCOM_Price`.

### `data/spx.xlsx`
- Monthly S&P 500 input.
- Loaded by `src/data_loader.py` as `SPX_Price`.

### `data/treasury_10y.xlsx`
- Monthly 10Y treasury yield proxy input.
- Loaded by `src/data_loader.py` as `Treasury10Y_Price`.

### `data/corp_bonds.xlsx`
- Monthly IG corporate bond proxy input.
- Loaded by `src/data_loader.py` as `IG_Corp_Price`.

### `data/final_backtest_results.csv`
- Exploratory output artifact from `notebooks/01_signals_analysis.ipynb`.
- Not required by the core evaluator pipeline.

---

## 8) Report Artifacts (`reports/`)

These files are generated outputs. They are not source code dependencies, but they are key analysis deliverables.

### JSON reports
- `reports/evaluation_report_static.json`
- `reports/evaluation_report_walk_forward.json`
- Generated by `scripts/run_evaluation.py`.

- `reports/evaluation_report_static_from_notebook.json`
- `reports/evaluation_report_walk_forward_from_notebook.json`
- Generated by `notebooks/02_unbiased_evaluation.ipynb`.

- `reports/evaluation_report.json`
- `reports/evaluation_report_from_notebook.json`
- Legacy/general report naming from earlier runs.

### CSV trade logs
- `reports/test_trade_log_static.csv`
- `reports/test_trade_log_walk_forward.csv`
- Generated by `scripts/run_evaluation.py`.

- `reports/test_trade_log_static_from_notebook.csv`
- `reports/test_trade_log_walk_forward_from_notebook.csv`
- Generated by `notebooks/02_unbiased_evaluation.ipynb`.

- `reports/test_trade_log.csv`
- `reports/test_trade_log_from_notebook.csv`
- Legacy/general naming from earlier runs.

---

## 9) Relationship Cheat Sheet (What to Edit for What)

- Change data quality rules or accepted schema:
- Edit `src/data_loader.py`.
- Re-run `tests/test_data_loader.py`.

- Change features/signals:
- Edit `src/features.py`.
- Validate behavior in `notebooks/01_signals_analysis.ipynb`.
- Re-run evaluation tests.

- Change model/policy/backtest logic:
- Edit `src/evaluation.py`.
- Re-run `tests/test_evaluation_policies.py`.
- Validate with `scripts/run_evaluation.py`.

- Change CLI UX/flags:
- Edit `scripts/run_evaluation.py`.
- Re-run `tests/test_cli_run_evaluation.py`.

- Change reproducible notebook workflow:
- Edit `notebooks/02_unbiased_evaluation.ipynb`.
- Confirm outputs in `reports/`.

- Change CI behavior:
- Edit `.github/workflows/tests.yml`.
