# Architecture

## Design Principle
The project is fully modular and self-contained. Each package owns its logic directly.

## Module Map
- `scripts/`: Thin CLI wrappers parsing arguments, injecting default configurations, and strictly calling `research/` or `benchmarks/` logic cleanly. No actual business or application logic is kept here.
- `data/`: Ingestion, caching, and fundamental normalization. Exposes standardized dataframes directly.
- `signals/`: Mathematical cross-sectional indicator formulations mapping prices directly into generic non-allocating labels (BUY/SELL/HOLD).
- `portfolio/`: Construction logic converting directional abstract signals directly into concrete target weights.
- `benchmarks/`: Canonical home for static allocation evaluation, moving average rules, baseline wrappers, and standard evaluation scripts computing benchmark artifacts. Native orchestration is executed by `evaluate.run_experiment`.
- `backtest/`: Extremely strict performance arithmetic applying the defined portfolio weights against universe returns factoring in transaction costs. Strict reporting formatting utilities also exist here.
- `research/`: Complex sub-experiments orchestrating strategy evaluations (walk-forward, robustness). Interfaces consistently accept an `args` namespace to pipe runs cleanly to outputs and artifact tracking logs.
- `analytics/`: Scaffolding reserved for diagnostics and custom reporting integrations currently absent.ics
- `benchmarks` -> [SCAFFOLD] future home for tactical/trend baseline strategies
- `research` -> owns execution orchestration logic for experiments
- `scripts` -> thin CLI wrappers orchestrating entry into `research/` routines
- `cli` -> command routing for all workflows

## Runtime Flow
1. `prepare-data`
2. `data-qc`
3. `run-primary-v1`
4. `run-benchmarks`

`run-all` executes the full sequence. Note that `validation_suite` natively orchestrates internal experiments like `walk_forward` and `robustness` by directly invoking their `research.*.run_experiment` endpoints to avoid expensive subprocess calls, falling back to subprocess boundaries only for explicitly separate pipelines like `pytest`. Identically, the `benchmarks` module is explicitly consumed directly as the source of truth for benchmark weights computations and standard static baselines comparisons.

## Why This Structure
- No hidden runtime monkeypatching.
- Module boundaries are explicit and testable.
- CLI orchestration is transparent and easy to extend.

## Output Directories
The canonical root for all generated output and evaluation pipelines is now `artifacts/`.
- `artifacts/data/`: Recreated, cached, and isolated CSV states.
- `artifacts/reports/`: Canonical location for all metric results, generated plots, and reproducibility manifests.

*Note: The project root previously included a `reports/` directory directly alongside source code. This is considered legacy and generated reports should now canonically output to `artifacts/reports/` instead.*
