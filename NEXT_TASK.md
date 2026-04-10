# Next Task

## Status

Awaiting planning

## Completed task

Fix PrimaryV1 reporting scope so official performance metrics come from walk-forward OOS results, while full-sample primary and benchmark artifacts remain available with explicit full-sample causal labeling.

## Current repo status

The repo now distinguishes official PrimaryV1 OOS reporting from full-sample causal diagnostics. `reports/results/primary_v1_oos_summary.csv` is sourced from walk-forward validation, while `primary_v1_summary.csv` and benchmark summaries now self-identify as full-sample causal outputs. The protected primary baseline remains unchanged, monthly CV still provides separate OOS validation, and secondary-model training, inference, threshold tuning, sizing, and evaluation outputs still do not appear to be implemented.

## Proposed next bounded tasks

- Add a targeted integration test for `run-monthly-cv` that exercises both `expanding` and `rolling` modes through the CLI and verifies rolling output paths
- Regenerate tracked report artifacts so the committed HTML and CSV outputs reflect the new PrimaryV1 OOS-versus-full-sample labeling contract
- Implement a minimal secondary-model training workflow that consumes the current dataset and causal split utilities without adding inference or dashboards
- Add a lightweight integration test that builds the secondary dataset and generates causal splits from the repo's clean-data path without touching tracked reports

## Recommended next task

Regenerate tracked report artifacts so the committed HTML and CSV outputs reflect the new PrimaryV1 OOS-versus-full-sample labeling contract.
