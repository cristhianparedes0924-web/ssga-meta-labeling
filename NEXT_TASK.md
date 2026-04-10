# Next Task

## Status

Awaiting planning

## Completed task

Document the PrimaryV1/M1 reporting contract in detail, including official OOS metrics, full-sample diagnostics, OOS information ratio calculation, and stale historical IR interpretation.

## Current repo status

The repo now distinguishes official PrimaryV1 OOS reporting from full-sample causal diagnostics and documents that contract in `docs/primary_reporting_contract.md`. `reports/results/primary_v1_oos_summary.csv` is sourced from walk-forward validation and includes an information ratio benchmarked against `EqualWeight25OOS` on the same OOS dates. `primary_v1_summary.csv` and benchmark summaries still self-identify as full-sample causal outputs. The protected primary baseline remains unchanged, monthly CV still provides separate OOS validation, and secondary-model training, inference, threshold tuning, sizing, and evaluation outputs still do not appear to be implemented.

## Proposed next bounded tasks

- Add a targeted integration test for `run-monthly-cv` that exercises both `expanding` and `rolling` modes through the CLI and verifies rolling output paths
- Review whether benchmark plots should get a separate OOS-only version rather than showing only full-sample causal curves
- Implement a minimal secondary-model training workflow that consumes the current dataset and causal split utilities without adding inference or dashboards
- Add a lightweight integration test that builds the secondary dataset and generates causal splits from the repo's clean-data path without touching tracked reports

## Recommended next task

Add a targeted integration test for `run-monthly-cv` that exercises both `expanding` and `rolling` modes through the CLI and verifies rolling output paths.
