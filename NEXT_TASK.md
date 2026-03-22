# Next Task

## Status

Awaiting planning

## Completed task

Extend the monthly cross-validation workflow to support configurable out-of-sample fold windows of 1, 3, 6, 9, 12, 15, and 18 calendar months while preserving the default 1-month behavior.

## Current repo status

The repo now has a monthly CV workflow that preserves the canonical 1-month mode, accepts an explicit `test_window_months` horizon parameter, writes non-default horizons into dedicated subfolders, and includes direct validation tests for fold causality and ordering. The protected primary baseline remains unchanged, and secondary-model training, inference, threshold tuning, sizing, and evaluation outputs still do not appear to be implemented.

## Proposed next bounded tasks

- Add a targeted integration test for `run-monthly-cv` that exercises the CLI with a non-default horizon and verifies horizon-specific output paths
- Extend monthly CV reporting with a compact comparison artifact across supported OOS horizons without changing the primary strategy logic
- Implement a minimal secondary-model training workflow that consumes the current dataset and causal split utilities without adding inference or dashboards
- Add a lightweight integration test that builds the secondary dataset and generates causal splits from the repo's clean-data path without touching tracked reports

## Recommended next task

Add a targeted integration test for `run-monthly-cv` that exercises the CLI with a non-default horizon and verifies horizon-specific output paths.
