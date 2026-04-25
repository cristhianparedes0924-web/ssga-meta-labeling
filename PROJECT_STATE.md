# Project State

## Implemented and working

Based on the current repo state, the primary-model repository already appears to include:

- Data preparation from raw Excel inputs into canonical clean CSV files
- A primary pipeline that builds the signal, maps weights, and runs the backtest
- Validation and test workflows through `run-self-tests`, `pytest`, and validation commands
- Benchmark generation and benchmark reporting outputs
- Robustness runs across threshold, duration, and transaction-cost grids
- Walk-forward validation outputs
- Monthly cross-validation outputs comparing expanding versus rolling training windows with monthly OOS evaluation
- Official PrimaryV1 OOS summary export sourced from walk-forward validation, including an OOS-aligned EqualWeight25 information ratio benchmark
- Clearly labeled full-sample causal primary and benchmark summaries
- Detailed PrimaryV1/M1 reporting-scope documentation in `docs/primary_reporting_contract.md`
- Reporting outputs in `reports/results/` and plots in `reports/assets/`
- A first-pass secondary dataset builder for event-level meta-labeling data
- Causal temporal split utilities for secondary train/validation preparation
- M2 secondary-model experiments, including supplemental feature construction, logistic/tree model evaluation helpers, walk-forward prediction utilities, standalone M2 scripts, notebooks, and tracked M2 report artifacts

The primary model is the current baseline. The main implemented strategy code lives under `src/metalabel/primary/`, with CLI entry points in `src/metalabel/cli.py` and runtime settings in `configs/primary.yaml`.

## Current baseline

- Baseline strategy: `PrimaryV1`
- Baseline config source: `configs/primary.yaml`
- Baseline outputs already appear to exist in `reports/results/primary_v1_*`
- `reports/results/primary_v1_summary.csv` now represents full-sample causal backtest metrics only
- `reports/results/primary_v1_oos_summary.csv` is now the canonical PrimaryV1 OOS performance summary sourced from walk-forward validation, with information ratio benchmarked against EqualWeight25 on matched OOS dates
- `docs/primary_reporting_contract.md` documents the difference between official OOS metrics, full-sample diagnostics, and stale historical IR values
- Benchmarks, robustness, walk-forward, and monthly CV outputs also appear to be present
- Monthly CV now supports expanding-history validation plus rolling train windows of 3, 6, 12, 24, and 36 months while keeping OOS evaluation monthly
- Secondary dataset construction now appears to be implemented under `src/metalabel/secondary/`
- Secondary temporal split utilities now appear to be implemented under `src/metalabel/secondary/`
- Secondary M2 modeling utilities now appear to be implemented under `src/metalabel/secondary/model.py`, with related feature helpers and scripts for M2, M2 v2/v3, Ridge, ROC, and carry-forward evaluation

## Not finished yet

Based on the current repo state, secondary-model development now appears to include dataset construction, causal split preparation, and experimental M2 model/evaluation workflows.

The repo does not yet appear to include a promoted production secondary model, a canonical secondary-specific package CLI command, or secondary evaluation dashboards.
