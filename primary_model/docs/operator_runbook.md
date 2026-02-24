# Operator Runbook

This runbook is the canonical way to execute and validate the `primary_model` pipeline.

## Preconditions

1. Run commands from the project root: `primary_model/`.
2. For best runtime, use Linux Python inside WSL (`/home/...`) instead of Windows Python over `\\wsl.localhost\...` UNC paths.
3. Ensure raw input files exist in `artifacts/data/raw/`:
   - `spx.xlsx`
   - `bcom.xlsx`
   - `treasury_10y.xlsx`
   - `corp_bonds.xlsx`
4. Install dependencies once:
   - `pip install -r requirements.txt`
   - `pip install -r requirements-dev.txt`

## Fast Dev Loop

Run:

```bash
make dev-loop
```

This executes:
1. Main pipeline (`cli.py run-all --root artifacts`) unless explicitly skipped.
2. Stage 2 signal validation using `configs/experiments/signal_validation_fast.yaml`.
3. Stage 3 ablations.
4. Stage 4 decision diagnostics.
5. Stage 5 readiness.

Additional fast commands:

```bash
make validate-fast
make robustness-fast
make walk-forward-fast
make signal-validation-fast
make m1-baseline-fast
```

`walk_forward_fast.yaml` uses `engine_mode: cached_causal`, which is significantly faster than repeated expanding-history recomputation.

## One-Command Full Validation

Run:

```bash
make full-validate
```

This executes:
1. Unit/integration tests (`pytest`).
2. Main pipeline (`cli.py run-all --root artifacts`).
3. Standalone robustness experiment.
4. Standalone walk-forward experiment.
5. Full validation suite with reproducibility manifest.

No `PYTHONPATH` export is required.

## Expected Outputs

Main outputs:
- `artifacts/reports/primary_v1_summary.csv`
- `artifacts/reports/benchmarks_summary.csv`
- `artifacts/reports/benchmarks_summary.html`
- `artifacts/reports/data_qc.html`

Research outputs:
- `artifacts/reports/robustness/robustness_grid_results.csv`
- `artifacts/reports/walk_forward/walk_forward_summary.csv`
- `artifacts/reports/signal_validation/signal_validation_fullsample.csv`
- `artifacts/reports/signal_validation/signal_validation_subperiods.csv`
- `artifacts/reports/signal_validation/signal_validation_monotonicity_summary.csv`
- `artifacts/reports/signal_validation/signal_validation_assessment.json`
- `artifacts/reports/ablations/ablations_variant_summary.csv`
- `artifacts/reports/ablations/ablations_leave_one_out.csv`
- `artifacts/reports/ablations/ablations_single_indicator.csv`
- `artifacts/reports/ablations/ablations_assessment.json`
- `artifacts/reports/decision_diagnostics/decision_transition_summary.csv`
- `artifacts/reports/decision_diagnostics/decision_whipsaw_summary.csv`
- `artifacts/reports/decision_diagnostics/decision_dwell_summary.csv`
- `artifacts/reports/decision_diagnostics/decision_score_zone_bins.csv`
- `artifacts/reports/decision_diagnostics/decision_assessment.json`
- `artifacts/reports/readiness/m1_readiness_checklist.json`
- `artifacts/reports/readiness/m1_readiness_summary.md`

Reproducibility outputs:
- `artifacts/reports/reproducibility/validation_manifest.json`
- `artifacts/reports/reproducibility/validation_summary.md`
- `artifacts/reports/reproducibility/01_setup_artifacts_root.log`
- `artifacts/reports/reproducibility/02_pytest.log`
- `artifacts/reports/reproducibility/03_cli_run_all.log` (when run-all is not skipped)
- `artifacts/reports/reproducibility/04_robustness.log`
- `artifacts/reports/reproducibility/05_walk_forward.log`
- `artifacts/reports/reproducibility/06_m1_readiness.log` (when readiness hook is enabled)
- `artifacts/reports/reproducibility/m1_baseline_snapshot.json`
- `artifacts/reports/reproducibility/m1_baseline_snapshot.md`

Stage 1 baseline freeze command:
```bash
python cli.py run-m1-baseline --config configs/experiments/m1_canonical_v1_1.yaml --root artifacts
```

Stage 2 signal validation command:
```bash
python cli.py run-signal-validation --config configs/experiments/signal_validation.yaml --root artifacts
```

Stage 3 ablation suite command:
```bash
python cli.py run-ablations --config configs/experiments/ablations.yaml --root artifacts
```

Stage 4 decision diagnostics command:
```bash
python cli.py run-decision-diagnostics --config configs/experiments/decision_diagnostics.yaml --root artifacts
```

Stage 5 readiness gate command:
```bash
python cli.py run-m1-readiness --config configs/experiments/m1_readiness.yaml --root artifacts
```

Fast validation suite command:
```bash
python cli.py run-validation-fast --config configs/experiments/validation_suite_fast.yaml --root artifacts
```

## Troubleshooting

- `ModuleNotFoundError: primary_model`:
  Run via project entrypoints (`python cli.py ...`, `python scripts/...`) from the repo root.
- `Missing required input file`:
  Verify the four `.xlsx` files are present in `artifacts/data/raw/`.
- Validation suite is slow:
  Use `run-validation-fast` during iteration and reserve `run-validation-suite` for final checks.
