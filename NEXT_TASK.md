# Next Task

## Status

Awaiting planning

## Completed task

Define the first-pass secondary-model training contract before adding any secondary-model training code.

## Current repo status

The repo now appears to have a first-pass secondary dataset builder, causal temporal split utilities, and a written secondary training contract under `docs/`. The primary model remains the protected baseline, and secondary-model training, inference, threshold tuning, sizing, and evaluation outputs still do not appear to be implemented.

## Proposed next bounded tasks

- Implement a minimal secondary-model training workflow that consumes the current dataset and causal split utilities without adding inference or dashboards
- Add a lightweight integration test that builds the secondary dataset and generates causal splits from the repo's clean-data path without touching tracked reports
- Implement a filtered-strategy evaluation helper that compares accepted primary events against the unchanged primary baseline without adding dashboards
- Add a small, explicit CLI command for building the secondary dataset and inspecting split boundaries without writing to canonical report outputs

## Recommended next task

Implement a minimal secondary-model training workflow that consumes the current dataset and causal split utilities without adding inference or dashboards.
