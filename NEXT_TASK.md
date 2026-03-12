# Next Task

## Status

Awaiting planning

## Completed task

Implement the first-pass secondary dataset builder for meta-labeling.

## Current repo status

The repo now appears to have a leakage-aware secondary dataset builder under `src/metalabel/secondary/` with focused tests, while the primary model remains the protected baseline. Secondary-model training, inference, CLI integration, and evaluation outputs still do not appear to be implemented.

## Proposed next bounded tasks

- Add a small, explicit CLI command for building and saving the secondary dataset to a non-canonical output path
- Define and document the first-pass secondary dataset schema and file contract for future training work
- Implement causal train/validation split utilities for the secondary dataset without adding any classifier training yet
- Add a lightweight integration test that builds the secondary dataset from the repo's clean-data path without touching tracked reports

## Recommended next task

Implement causal train/validation split utilities for the secondary dataset without adding model training yet.
