# Walk-Forward Protocol

- Train window: expanding from first observation to decision date `t`.
- Decision at `t`: compute signal using only data through `t`.
- Realization at `t+1`: apply weight decided at `t` to next-period return.
- Minimum train periods before first OOS decision: `120`.
- Treasury duration assumption: `8.5`.
- Thresholds: buy `0.0001`, sell `-0.0001`.
- Transaction cost: `0.0` bps.
- OOS decisions evaluated: `239`.
- First OOS decision date: `2006-01-31`.
- Last OOS decision date: `2025-11-28`.

## Outputs
- `/home/cristhian/Projects/ssga-meta-labeling/primary_model/artifacts/reports/walk_forward/walk_forward_backtest.csv`
- `/home/cristhian/Projects/ssga-meta-labeling/primary_model/artifacts/reports/walk_forward/standard_causal_backtest_slice.csv`
- `/home/cristhian/Projects/ssga-meta-labeling/primary_model/artifacts/reports/walk_forward/walk_forward_summary.csv`