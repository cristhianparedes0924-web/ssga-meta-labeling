# Decisions

## 2026-03-12

### Decision

Freeze the current primary model as the stable baseline.

### Reason

Based on the current repo state, the primary model already has implemented pipeline code, configuration, tests, and tracked result artifacts. It is the only fully implemented strategy path in the repository today.

### Impact

Future work should treat current primary behavior as the reference implementation. Any intentional baseline change should be explicit, documented, and followed by regeneration of affected outputs.

## 2026-03-12

### Decision

Continue new development in `secondary/` while minimizing risk to `primary/`.

### Reason

The repository already separates `src/metalabel/primary/` from `src/metalabel/secondary/`, and `secondary/` currently appears to be only a placeholder. Using that boundary reduces risk to the working primary pipeline.

### Impact

New experimentation should be added under `secondary/` first, with conservative interfaces and no silent changes to primary commands, configs, or reports. This keeps the baseline stable while opening a safe path for secondary-model work.

## 2026-03-12

### Decision

Use actionable primary events as the first secondary observations, define `meta_label` from the next-period realized primary-strategy net return, and implement secondary dataset construction before any secondary-model training.

### Reason

This keeps the first secondary step aligned with actual primary behavior instead of a proxy target, limits leakage risk by separating decision-time features from `t+1` outcomes, and keeps the rollout incremental.

### Impact

The first secondary dataset is built from `BUY` and `SELL` primary events by default, `meta_target_return` reflects the realized next-period primary-strategy net return implied at the decision date, and model training remains out of scope until this dataset contract is stable.

## 2026-03-12

### Decision

Use chronological date-grouped temporal splits for secondary validation, with expanding-train / forward-validation as the default reusable protocol and a simple time holdout helper for smaller experiments.

### Reason

Secondary rows represent decision-time events, so validation must respect causal order. Grouping by decision date keeps same-date rows together, avoids train/validation leakage at shared timestamps, and gives later training code a simple protocol to reuse.

### Impact

Secondary split utilities now operate on ordered unique decision dates, never shuffle rows, require validation windows to occur strictly after training windows, and expose both a single holdout split and reusable expanding forward splits for future model training.

## 2026-03-12

### Decision

The first secondary training pass will use actionable-event rows to predict `meta_label` with a simple logistic-regression classifier, focusing on false-positive filtering of the unchanged primary strategy rather than sizing.

### Reason

The current repo already defines actionable-event observations, a binary meta-label, and causal split utilities. A simple logistic-regression baseline is conservative, matches the existing dependency stack, and is sufficient for the first pass before adding more complex model families or sizing rules.

### Impact

Future first-pass training work should treat the secondary model as an accept/reject filter on primary actions, train on decision-time features only, keep position sizing out of scope, and evaluate both classification quality and filtered-strategy performance against the unchanged primary baseline.

## 2026-03-22

### Decision

Keep `run-monthly-cv` backward-compatible at a 1-month default horizon, and write multi-month monthly-CV outputs into horizon-specific subfolders under `reports/results/monthly_cv/`.

### Reason

The repository already tracks canonical monthly-CV outputs in `reports/results/monthly_cv/`. Preserving the 1-month default avoids breaking the existing command and artifact paths, while subfolders for wider OOS windows prevent different horizon runs from overwriting one another.

### Impact

Future monthly-CV work should treat `1` month as the canonical default mode, pass wider OOS windows explicitly through `test_window_months`, and expect non-default horizons to write under folders such as `reports/results/monthly_cv/03m_oos/`.

## 2026-03-22

### Decision

Define monthly CV as a comparison between expanding and rolling training windows, keep OOS evaluation monthly, and store rolling-window outputs in dedicated subfolders under `reports/results/monthly_cv/`.

### Reason

The project walkthrough design compares how the same monthly OOS validation behaves under different training-history constructions. Varying OOS horizon length was the wrong interpretation; the correct comparison axis is training-window type, with rolling windows of 3, 6, 12, 24, and 36 months.

### Impact

Future monthly-CV work should keep one OOS month per fold, treat expanding history as the canonical root output in `reports/results/monthly_cv/`, and write rolling comparisons under folders such as `reports/results/monthly_cv/rolling_12m/`.

## 2026-04-10

### Decision

Treat walk-forward validation as the canonical OOS performance source for official `PrimaryV1` reporting, and label `primary_v1_summary.csv` plus benchmark summaries as full-sample causal backtest artifacts.

### Reason

The repository already had a true OOS walk-forward pipeline, but the main `PrimaryV1` and benchmark summary artifacts were full-sample backtests and could be misread as official OOS performance metrics.

### Impact

Future reporting should source official PrimaryV1 performance metrics from `reports/results/primary_v1_oos_summary.csv` or `reports/results/walk_forward/walk_forward_summary.csv`, while treating `primary_v1_summary.csv` and benchmark summary artifacts as full-sample causal diagnostics only.
