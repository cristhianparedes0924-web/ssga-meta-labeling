# Operator Runbook

This runbook is the canonical way to execute and validate the `primary_model` pipeline.

## Preconditions

1. Run commands from the project root: `primary_model/`.
2. Ensure raw input files exist in `artifacts/data/raw/`:
   - `spx.xlsx`
   - `bcom.xlsx`
   - `treasury_10y.xlsx`
   - `corp_bonds.xlsx`
3. Install dependencies once:
   - `pip install -r requirements.txt`
   - `pip install -r requirements-dev.txt`

## One-Command Full Validation

Run:

```bash
make full-validate
```

This executes:
1. Unit/integration tests (`pytest`).
2. Main pipeline (`cli.py run-all --root artifacts`).
3. Standalone robustness experiment.
4. Standalone strict walk-forward experiment.
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

Reproducibility outputs:
- `artifacts/reports/reproducibility/validation_manifest.json`
- `artifacts/reports/reproducibility/validation_summary.md`
- `artifacts/reports/reproducibility/01_setup_artifacts_root.log`
- `artifacts/reports/reproducibility/02_pytest.log`
- `artifacts/reports/reproducibility/03_cli_run_all.log`
- `artifacts/reports/reproducibility/04_robustness.log`
- `artifacts/reports/reproducibility/05_walk_forward.log`

## Troubleshooting

- `ModuleNotFoundError: primary_model`:
  Run via project entrypoints (`python cli.py ...`, `python scripts/...`) from the repo root.
- `Missing required input file`:
  Verify the four `.xlsx` files are present in `artifacts/data/raw/`.
- Validation suite is slow:
  `walk_forward` is the longest step by design due to repeated expanding-window recomputation.
