# Next Task

## Status

Awaiting planning

## Completed task

Merge the GitHub branch `Sam_M2_Trees` into `main`, preserving the current PrimaryV1/M1 OOS reporting contract while adding the branch's M2 tree-model, evaluation, notebook, and report artifacts.

## Current repo status

The repo now includes the M2 tree-model branch on top of the current `main` baseline. The protected PrimaryV1 reporting contract remains in place, including `docs/primary_reporting_contract.md`, `reports/results/primary_v1_oos_summary.csv`, and matched OOS benchmark reporting. The merge adds experimental M2 modeling, feature, evaluation, notebook, script, and report artifacts while retaining the existing package tests and secondary date-grouped split utilities.

## Proposed next bounded tasks

- Review the merged M2 artifacts and decide which scripts/reports should become canonical package CLI commands
- Add a focused integration test for the merged M2 script path most likely to be reused by collaborators
- Add a targeted integration test for `run-monthly-cv` that exercises both `expanding` and `rolling` modes through the CLI and verifies rolling output paths
- Review whether benchmark plots should get a separate OOS-only version rather than showing only full-sample causal curves

## Recommended next task

Review the merged M2 artifacts and decide which scripts/reports should become canonical package CLI commands.
