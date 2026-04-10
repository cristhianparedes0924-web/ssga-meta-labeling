# PrimaryV1 / M1 Reporting Contract

This document defines how PrimaryV1, also referred to by the team as M1, should
be reported.

## Summary

PrimaryV1 has two valid metric scopes:

- Official out-of-sample metrics
- Full-sample causal diagnostics

Official M1 performance should use the out-of-sample walk-forward artifacts.
Full-sample artifacts remain useful for diagnostics, but they must not be
presented as official OOS performance.

## Model Name

The repository does not contain a separate object named `M1`. The implemented
primary baseline is `PrimaryV1`, and it is the repo's effective M1.

PrimaryV1 is implemented by:

- `src/metalabel/primary/signals.py`
- `src/metalabel/primary/portfolio.py`
- `src/metalabel/primary/backtest.py`
- `src/metalabel/primary/pipeline.py`
- `src/metalabel/validation.py`

The protected baseline signal logic should not be changed just to alter
reported metrics.

## Official OOS Metrics

Official M1 metrics are sourced from walk-forward validation.

Canonical files:

- `reports/results/primary_v1_oos_summary.csv`
- `reports/results/walk_forward/walk_forward_summary.csv`
- `reports/results/walk_forward/walk_forward_oos_benchmark_summary.csv`
- `reports/results/walk_forward/walk_forward_backtest.csv`
- `reports/results/walk_forward/equal_weight_oos_backtest.csv`

The official one-row M1 summary is:

```text
reports/results/primary_v1_oos_summary.csv
```

That file is sourced from the `WalkForwardStrict` row of:

```text
reports/results/walk_forward/walk_forward_summary.csv
```

Its `evaluation_scope` must be:

```text
oos_walk_forward
```

Its `source_validation` must be:

```text
walk_forward
```

Its `benchmark_key` must be:

```text
EqualWeight25OOS
```

## OOS Information Ratio

Information ratio requires a benchmark return series on the same dates as the
strategy return series.

The official M1 OOS information ratio uses:

```text
strategy = WalkForwardStrict
benchmark = EqualWeight25OOS
```

Both series are aligned to the same walk-forward OOS dates.

Formula:

```text
active_return_t = strategy_net_return_t - benchmark_net_return_t
information_ratio = mean(active_return_t) / std(active_return_t) * sqrt(12)
```

The current official OOS value is:

```text
PrimaryV1 OOS IR vs EqualWeight25OOS = 0.6632
```

The benchmark artifact is:

```text
reports/results/walk_forward/equal_weight_oos_backtest.csv
```

The compact OOS comparison artifact is:

```text
reports/results/walk_forward/walk_forward_oos_benchmark_summary.csv
```

## Full-Sample Diagnostics

The following artifacts are full-sample causal diagnostics:

- `reports/results/primary_v1_summary.csv`
- `reports/results/primary_v1_classification.csv`
- `reports/results/benchmarks_summary.csv`
- `reports/results/benchmarks_summary.html`
- `reports/assets/equity_curves.png`
- `reports/assets/drawdowns.png`
- `reports/assets/rolling_sharpe.png`

These artifacts should include or display the full-sample scope explicitly.
CSV summary rows use:

```text
evaluation_scope = full_sample_causal
```

The benchmark HTML report states that it is a full-sample causal benchmark
comparison and points readers to the official OOS artifacts.

The current full-sample benchmark IR for PrimaryV1 is:

```text
PrimaryV1 full-sample IR vs EqualWeight25 = 0.3035
```

This number is not the official M1 OOS IR.

## Historical Stale IR

Before the reporting-scope cleanup, a stale benchmark artifact showed:

```text
PrimaryV1 IR = 0.6383
```

That number came from an older full-sample benchmark report. It should not be
used as a current official metric.

Current interpretation:

```text
0.6632 = official OOS IR vs EqualWeight25OOS
0.3035 = current full-sample benchmark diagnostic IR vs EqualWeight25
0.6383 = stale historical full-sample artifact value
```

## Reporting Rules

When preparing a team update:

- Use `primary_v1_oos_summary.csv` for official M1 performance.
- Use `walk_forward_oos_benchmark_summary.csv` when explaining OOS benchmark
  comparison or OOS information ratio.
- Mention that full-sample primary and benchmark summaries remain diagnostics.
- Do not mix full-sample benchmark metrics with OOS metrics in the same official
  scorecard unless the scope is explicitly labeled.
- Do not use the stale `0.6383` IR except to explain historical artifact drift.

## Regeneration Commands

Preferred full workflow when runtime allows:

```powershell
python -m metalabel.cli run-all
python -m metalabel.cli run-walk-forward
python -m metalabel.cli run-self-tests
python -m pytest
```

If only reporting code changed and existing walk-forward backtest rows are still
valid, the OOS summary and EqualWeight25OOS benchmark can be regenerated from the
current clean data and existing walk-forward backtest files. That shortcut should
not be used after strategy logic, data, costs, thresholds, or walk-forward split
logic changes.

## Validation Expectations

Reporting-scope regressions are covered by:

```text
tests/test_reporting_scope.py
```

The tests assert that:

- full-sample primary summaries are labeled `full_sample_causal`
- full-sample benchmark outputs point readers to OOS artifacts
- official PrimaryV1 OOS reporting is sourced from walk-forward output
- official OOS IR has `benchmark_key = EqualWeight25OOS`
- the OOS benchmark summary contains both PrimaryV1 and EqualWeight25OOS rows
