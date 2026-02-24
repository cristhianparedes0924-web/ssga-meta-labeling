# Robustness Summary

- Scenarios evaluated: `135`
- Duration grid: `[6.0, 8.5, 10.0]`
- Buy thresholds: `[0.0001, 0.25, 0.5]`
- Sell thresholds: `[-0.0001, -0.25, -0.5]`
- Transaction-cost grid (bps): `[0.0, 5.0, 10.0, 25.0, 50.0]`

## Best Sharpe Scenario
- Scenario: `S008`
- Duration: `6.0`
- Thresholds: buy `0.5`, sell `-0.25`
- Cost (bps): `0.0`
- Sharpe: `0.9631`
- Ann. return: `6.2805%`
- Max drawdown: `-16.5530%`

## Outputs
- `/home/cristhian/Projects/ssga-meta-labeling/primary_model/artifacts/reports/robustness/robustness_grid_results.csv`
- `/home/cristhian/Projects/ssga-meta-labeling/primary_model/artifacts/reports/robustness/baseline_cost_sensitivity.csv`
- `/home/cristhian/Projects/ssga-meta-labeling/primary_model/artifacts/reports/robustness/top15_by_sharpe.csv`