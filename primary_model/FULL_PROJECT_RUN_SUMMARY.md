# Full Project Execution Summary

Date: 2026-02-24  
Project: `primary_model`  
Run scope: full pipeline + robustness + walk-forward + validation suite + Stage 1-5 research flow

## 1. What Was Executed

1. `python -m pytest -q`
2. `python cli.py run-all --root artifacts`
3. `python cli.py run-robustness --root artifacts --out-dir artifacts/reports/robustness --config configs/experiments/robustness.yaml`
4. `python cli.py run-walk-forward --root artifacts --out-dir artifacts/reports/walk_forward --config configs/experiments/walk_forward.yaml`
5. `python cli.py run-validation-suite --root artifacts --clean-root`
6. `python cli.py run-m1-baseline --root artifacts --config configs/experiments/m1_canonical_v1_1.yaml`
7. `python cli.py run-signal-validation --root artifacts --config configs/experiments/signal_validation.yaml`
8. `python cli.py run-ablations --root artifacts --config configs/experiments/ablations.yaml`
9. `python cli.py run-decision-diagnostics --root artifacts --config configs/experiments/decision_diagnostics.yaml`
10. `python cli.py run-m1-readiness --root artifacts --config configs/experiments/m1_readiness.yaml`

## 2. Overall Status

- Full test suite: passed (`38` tests).
- Main pipeline (`run-all`): passed.
- Robustness study: passed.
- Walk-forward study: passed.
- Validation suite (clean-root): passed.
- Stage 1 baseline: passed with deterministic snapshot.
- Stage 2 signal validation: passed.
- Stage 3 ablations: passed.
- Stage 4 decision diagnostics: passed.
- Stage 5 readiness gate: passed (`readiness_passed = true`).

## 3. Core Strategy Results (PrimaryV1)

Source: `artifacts/reports/primary_v1_summary.csv`

| Metric | Value |
|---|---:|
| Annual Return | 0.0633 |
| Annual Volatility | 0.0770 |
| Sharpe | 0.8379 |
| Max Drawdown | -0.2208 |
| Calmar | 0.2866 |
| Avg Turnover | 0.1892 |

Interpretation: the strategy profile remains consistent with prior Stage work; performance is moderate return with relatively controlled drawdown and higher turnover than static benchmarks.

## 4. Benchmark Comparison Snapshot

Source: `artifacts/reports/benchmarks_summary.csv`

- Best Sharpe in benchmark table: `PrimaryV1` (`0.8379`).
- `PrimaryV1` vs EqualWeight25 annual return spread: positive (from table output and saved benchmark report).

Interpretation: on risk-adjusted return, `PrimaryV1` still leads the configured baseline benchmark set.

## 5. Robustness and Walk-Forward

### Robustness
Source: `artifacts/reports/robustness/robustness_grid_results.csv`

- Grid points evaluated: `135`.
- Best scenario: `S008` (`duration=6.0`, `buy=0.5`, `sell=-0.25`, `tcost=0`) with Sharpe `0.9631`, max DD `-0.1655`.
- Worst scenario Sharpe: `0.5922`, max DD `-0.2877`.

Interpretation: the strategy does not collapse under the tested grid; outcomes vary materially by threshold/cost settings but remain positive across the sweep.

### Walk-Forward
Source: `artifacts/reports/walk_forward/walk_forward_summary.csv`

| Metric | Value |
|---|---:|
| Annual Return | 0.0558 |
| Annual Volatility | 0.0789 |
| Sharpe | 0.7288 |
| Max Drawdown | -0.2208 |
| Calmar | 0.2526 |
| Avg Turnover | 0.2008 |

Interpretation: strict walk-forward is weaker than in-sample/backtest metrics, but remains positive and stable.

## 6. Stage 1-5 Research Outcomes

### Stage 1 (Baseline Freeze)
Source: `artifacts/reports/reproducibility/m1_baseline_snapshot.json`

- Status: `ok`
- Determinism check: `true`
- Determinism runs: `2`

Interpretation: canonical baseline is reproducible under repeated execution.

### Stage 2 (Signal Validity)
Source: `artifacts/reports/signal_validation/signal_validation_assessment.json`

- Passed: `true`
- Composite Spearman: `0.0626`
- Composite top-bottom gap: `0.0090`
- Positive subperiods: `3 / 4`

Interpretation: composite has positive directional and monotonic evidence, with acceptable subperiod sign consistency by current criteria.

### Stage 3 (Ablations)
Source: `artifacts/reports/ablations/ablations_assessment.json`

- Passed: `true`
- Dynamic minus equal-weight Sharpe: `-0.0959`
- Recommended aggregation from evidence: `equal_weight`
- Single-indicator dominance ratio vs baseline: `1.0942`

Interpretation: ablations are informative and complete, and evidence currently favors equal-weight aggregation over dynamic aggregation.

### Stage 4 (Decision Diagnostics)
Source: `artifacts/reports/decision_diagnostics/decision_assessment.json`

- Passed: `true`
- Transition value-add quantified: `true`
- Whipsaw profile comparable: `true`
- Threshold behavior evidence-backed: `true`

Dynamic variant diagnostics used by readiness:
- Min average switch value (BUY->SELL / SELL->BUY): `0.000454`
- Total switch value (BUY<->SELL): `0.0881915`
- False flip rate: `0.5224`

Interpretation: transition behavior is measurable and net-positive under current definitions, but whipsaw is still non-trivial.

### Stage 5 (Readiness Gate)
Source: `artifacts/reports/readiness/m1_readiness_checklist.json`

- Criteria version: `m1_readiness_v1`
- Readiness passed: `true`
- Failed checks: `none`

Interpretation: by current objective gate definitions, M1 is ready.

## 7. Validation Suite Manifest

Source: `artifacts/reports/reproducibility/validation_manifest.json`

- Manifest created (UTC): `2026-02-24T21:56:22.461280+00:00`
- Execution steps recorded: `5`
- Hashed artifacts: `23`
- Python version: `3.14.2`
- Git commit captured: `945e74de524976b09be03e5fc7e92f02af9aff3b`

## 8. Issue Encountered and Resolution

- A run of `run-validation-suite --clean-root --run-m1-readiness` failed because clean-root removed Stage 2-4 artifacts required by Stage 5.
- Resolution:
1. Ran `run-validation-suite --clean-root` successfully.
2. Re-ran Stage 1-5 commands to regenerate all Stage artifacts.
3. Confirmed Stage 5 readiness output as passing.

## 9. Final Conclusion

The full project execution completed successfully with the recent improvements. All core and Stage 1-5 research workflows now run, produce artifacts, and pass their configured acceptance gates, including a passing Stage 5 readiness decision.
