# Architecture

## Design Principle
`primary_model_unified.py` is frozen and treated as the canonical implementation.

All other modules are adapters that make the project easier to navigate, test,
and package without altering model behavior.

## Module Map
- `data.cleaner` -> raw Excel cleaning and canonical data contract helpers
- `data.loader` -> clean CSV loading, universe construction, treasury adjustment
- `signals.variant1` -> indicator, z-score, score, and signal generation
- `portfolio.weights` -> signal-to-weights and benchmark allocation logic
- `backtest.engine` -> return attribution, transaction costs, performance metrics
- `backtest.reporting` -> benchmark reporting and chart generation
- `qc.reports` -> data quality checks and QC HTML generation
- `cli` -> stable entrypoint delegating to unified core `main()`

## Runtime Flow
1. `prepare-data`
2. `data-qc`
3. `run-primary-v1`
4. `run-benchmarks`

`run-all` executes the full sequence.

## Why This Structure
- Maintains behavior parity with the unified file.
- Provides discoverable module boundaries for future extraction.
- Enables incremental refactor from monolith to package without breaking users.
