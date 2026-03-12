# Project State

## Implemented and working

Based on the current repo state, the primary-model repository already appears to include:

- Data preparation from raw Excel inputs into canonical clean CSV files
- A primary pipeline that builds the signal, maps weights, and runs the backtest
- Validation and test workflows through `run-self-tests`, `pytest`, and validation commands
- Benchmark generation and benchmark reporting outputs
- Robustness runs across threshold, duration, and transaction-cost grids
- Walk-forward validation outputs
- Monthly cross-validation outputs
- Reporting outputs in `reports/results/` and plots in `reports/assets/`

The primary model is the current baseline. The main implemented strategy code lives under `src/metalabel/primary/`, with CLI entry points in `src/metalabel/cli.py` and runtime settings in `configs/primary.yaml`.

## Current baseline

- Baseline strategy: `PrimaryV1`
- Baseline config source: `configs/primary.yaml`
- Baseline outputs already appear to exist in `reports/results/primary_v1_*`
- Benchmarks, robustness, walk-forward, and monthly CV outputs also appear to be present

## Not finished yet

Based on the current repo state, secondary-model development does not appear to be implemented beyond a placeholder package at `src/metalabel/secondary/__init__.py`.

The repo does not appear to include a secondary-model pipeline, secondary-specific CLI command, secondary tests, or secondary reporting outputs yet.
