# Architecture

## Design Principle
The project is fully modular and self-contained. Each package owns its logic directly.

## Module Map
- `data.cleaner` -> raw Excel cleaning and canonical data contract helpers
- `data.loader` -> clean CSV loading, universe construction, treasury adjustment
- `signals.variant1` -> indicator, z-score, score, and signal generation
- `portfolio.weights` -> signal-to-weights and benchmark allocation logic
- `backtest.engine` -> return attribution, transaction costs, performance metrics
- `backtest.reporting` -> benchmark reporting and chart generation
- `qc.reports` -> data quality checks and QC HTML generation
- `cli` -> command routing for all workflows

## Runtime Flow
1. `prepare-data`
2. `data-qc`
3. `run-primary-v1`
4. `run-benchmarks`

`run-all` executes the full sequence.

## Why This Structure
- No hidden runtime monkeypatching.
- Module boundaries are explicit and testable.
- CLI orchestration is transparent and easy to extend.
