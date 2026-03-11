# Walk-Forward Protocol

- Train window: expanding from first observation to decision date `t`.
- Decision at `t`: compute signal using only data through `t`.
- Realization at `t+1`: apply weight decided at `t` to next-period return.
- Minimum train periods before first OOS decision: `120`.
- Treasury duration assumption: `8.5`.
- Thresholds: buy `0.31`, sell `-0.31`.
- Transaction cost: `0.0` bps.
- OOS decisions evaluated: `239`.
- First OOS decision date: `2006-01-31`.
- Last OOS decision date: `2025-11-28`.

## Outputs
- `C:\Users\crist\OneDrive\Documents\primary_model_test\meta-labeling-project\reports\results\walk_forward\walk_forward_backtest.csv`
- `C:\Users\crist\OneDrive\Documents\primary_model_test\meta-labeling-project\reports\results\walk_forward\standard_causal_backtest_slice.csv`
- `C:\Users\crist\OneDrive\Documents\primary_model_test\meta-labeling-project\reports\results\walk_forward\walk_forward_summary.csv`