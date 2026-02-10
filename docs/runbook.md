# Project Runbook

This runbook is the operational guide for running, validating, and troubleshooting the project.

## 1) Environment Setup

From project root:

```bash
cd /home/cristhian/Projects/ssga-meta-labeling
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Required input files in `data/`:
- `bcom.xlsx`
- `spx.xlsx`
- `treasury_10y.xlsx`
- `corp_bonds.xlsx`

## 2) Quick Start Commands

### Baseline static evaluation (threshold policy)

```bash
python scripts/run_evaluation.py --mode static --transaction-cost-bps 0
```

### Baseline walk-forward evaluation (threshold policy)

```bash
python scripts/run_evaluation.py --mode walk_forward --transaction-cost-bps 0
```

### Static utility evaluation with calibration

```bash
python scripts/run_evaluation.py \
  --mode static \
  --decision-policy utility \
  --probability-calibration sigmoid \
  --utility-margin 0.0 \
  --utility-risk-aversion 0.25 \
  --transaction-cost-bps 5
```

### Walk-forward utility evaluation with calibration

```bash
python scripts/run_evaluation.py \
  --mode walk_forward \
  --decision-policy utility \
  --probability-calibration sigmoid \
  --utility-margin 0.0 \
  --utility-risk-aversion 0.25 \
  --transaction-cost-bps 5
```

## 3) Custom Output Paths

Use explicit artifact destinations:

```bash
python scripts/run_evaluation.py \
  --mode static \
  --report-path reports/my_static_report.json \
  --test-trades-path reports/my_static_trades.csv
```

## 4) Notebook Runs

### Exploratory notebook

```bash
jupyter lab notebooks/01_signals_analysis.ipynb
```

Purpose:
- Exploratory analysis and strategy intuition.
- Produces `data/final_backtest_results.csv`.

### Unbiased evaluation notebook

```bash
jupyter lab notebooks/02_unbiased_evaluation.ipynb
```

Purpose:
- Reproducible evaluation aligned with `src/evaluation.py`.
- Supports `threshold` and `utility` policy config toggles.
- Writes notebook-scoped outputs to `reports/*_from_notebook.*`.

## 5) Expected Artifacts

Script default outputs (static mode):
- `reports/evaluation_report_static.json`
- `reports/test_trade_log_static.csv`

Script default outputs (walk-forward mode):
- `reports/evaluation_report_walk_forward.json`
- `reports/test_trade_log_walk_forward.csv`

Notebook outputs (02 notebook):
- `reports/evaluation_report_static_from_notebook.json`
- `reports/test_trade_log_static_from_notebook.csv`
- `reports/evaluation_report_walk_forward_from_notebook.json`
- `reports/test_trade_log_walk_forward_from_notebook.csv`

## 6) How to Read Key Outputs

In report JSON (`evaluation_report_*.json`):
- `config`: exact run parameters (mode, policy, calibration, costs).
- `splits`: date ranges and row counts for train/validation/test.
- `test_metrics.primary` and `test_metrics.meta`: final performance comparison.
- `threshold_summary` (walk-forward threshold mode only).
- `utility_score_summary` (walk-forward utility mode only).

In trade log CSV (`test_trade_log_*.csv`):
- `Model_Probability`: model confidence on tradable events.
- `Selected_Threshold`: populated in threshold policy.
- `Utility_Score`: populated in utility policy.
- `Meta_Take_Trade`: final policy decision (0/1).
- `Meta_Return` and `Cumulative_Meta`: realized strategy path.

## 7) Testing and Quality Gates

Run all tests locally:

```bash
python -m unittest discover -s tests -v
```

CI behavior:
- `.github/workflows/tests.yml` runs the same test command on each `push` and `pull_request`.

## 8) Common Troubleshooting

### Error: missing required file or missing required columns

Cause:
- Strict loader checks in `src/data_loader.py` detected missing files or schema mismatch.

Fix:
- Verify all required files exist in `data/`.
- Confirm each file has `Date` and `PX_LAST` after Bloomberg header rows.

### Error: merged data contains missing values

Cause:
- Date alignment mismatch between one or more input series.

Fix:
- Check date coverage and frequency in all four source files.
- Ensure no shifted month-end calendars across files.

### Error: validation/test split has zero `Trade_Signal` events

Cause:
- Current split fractions plus data regime produced no positive momentum events in a split.

Fix:
- Adjust `--train-frac` / `--val-frac`.
- Reduce `forward_window` if needed.

### Notebook cannot locate project root

Cause:
- Notebook launched from outside repo root.

Fix:
- Start Jupyter from `/home/cristhian/Projects/ssga-meta-labeling`.

### Walk-forward runs are slow

Cause:
- Repeated model fitting each test period.

Fix:
- Reduce `--validation-window`.
- Increase `--threshold-step` for fewer threshold candidates.
- Use smaller datasets for quick iteration.

## 9) Recommended Working Pattern

1. Run static baseline (`threshold`, no calibration, no costs).
2. Add realistic costs (`--transaction-cost-bps 5`).
3. Compare threshold vs utility decisions.
4. Validate robustness with walk-forward mode.
5. Save reports and track parameter changes.
6. Run test suite before commit.
