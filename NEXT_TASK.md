# Next Task

## Status

Awaiting planning

## Completed task

Refactor monthly cross-validation so it compares expanding versus rolling training windows, keeps OOS evaluation monthly, and supports rolling train lengths of 3, 6, 12, 24, and 36 months.

## Current repo status

The repo now has a corrected monthly CV workflow that preserves monthly OOS folds, uses `window_type=expanding` or `window_type=rolling`, writes rolling comparisons into dedicated subfolders, and includes direct validation tests for expanding-versus-rolling train-window behavior. The protected primary baseline remains unchanged, and secondary-model training, inference, threshold tuning, sizing, and evaluation outputs still do not appear to be implemented.

## Proposed next bounded tasks

- Add a targeted integration test for `run-monthly-cv` that exercises both `expanding` and `rolling` modes through the CLI and verifies rolling output paths
- Extend monthly CV reporting with a compact comparison artifact across expanding and rolling train-window variants without changing the primary strategy logic
- Implement a minimal secondary-model training workflow that consumes the current dataset and causal split utilities without adding inference or dashboards
- Add a lightweight integration test that builds the secondary dataset and generates causal splits from the repo's clean-data path without touching tracked reports

## Recommended next task

Add a targeted integration test for `run-monthly-cv` that exercises both `expanding` and `rolling` modes through the CLI and verifies rolling output paths.
