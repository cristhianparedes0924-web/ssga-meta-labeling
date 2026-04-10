# Walk-Forward Protocol

- Train window: expanding from first observation to decision date `t`.
- Decision at `t`: compute signal using only data through `t`.
- Realization at `t+1`: apply weight decided at `t` to next-period return.
- OOS information ratio uses EqualWeight25 on the same walk-forward OOS dates.
- Minimum train periods before first OOS decision: `120`.
- Treasury duration assumption: `8.5`.
- Thresholds: buy `0.31`, sell `-0.31`.
- Transaction cost: `0.0` bps.
- OOS decisions evaluated: `239`.
- First OOS decision date: `2006-01-31`.
- Last OOS decision date: `2025-11-28`.

## Outputs
- `/home/cristhian/Projects/ssga-meta-labeling/reports/results/walk_forward/walk_forward_backtest.csv`
- `/home/cristhian/Projects/ssga-meta-labeling/reports/results/walk_forward/equal_weight_oos_backtest.csv`
- `/home/cristhian/Projects/ssga-meta-labeling/reports/results/walk_forward/standard_causal_backtest_slice.csv`
- `/home/cristhian/Projects/ssga-meta-labeling/reports/results/walk_forward/walk_forward_summary.csv`
- `/home/cristhian/Projects/ssga-meta-labeling/reports/results/walk_forward/walk_forward_oos_benchmark_summary.csv`
- `/home/cristhian/Projects/ssga-meta-labeling/reports/results/primary_v1_oos_summary.csv`