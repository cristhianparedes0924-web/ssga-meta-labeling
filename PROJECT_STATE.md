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
- A first-pass secondary dataset builder for event-level meta-labeling data
- Causal temporal split utilities for secondary train/validation preparation

The primary model is the current baseline. The main implemented strategy code lives under `src/metalabel/primary/`, with CLI entry points in `src/metalabel/cli.py` and runtime settings in `configs/primary.yaml`.

## Current baseline

- Baseline strategy: `PrimaryV1`
- Baseline config source: `configs/primary.yaml`
- Baseline outputs already appear to exist in `reports/results/primary_v1_*`
- Benchmarks, robustness, walk-forward, and monthly CV outputs also appear to be present
- Secondary dataset construction now appears to be implemented under `src/metalabel/secondary/`
- Secondary temporal split utilities now appear to be implemented under `src/metalabel/secondary/`

## Not finished yet

Based on the current repo state, secondary-model development now appears to include first-pass dataset construction and causal split preparation, but it does not yet appear to include model training, inference, promotion logic, or secondary reporting outputs.

The repo does not appear to include a trained secondary model, a secondary-specific CLI command, or secondary evaluation dashboards yet.
