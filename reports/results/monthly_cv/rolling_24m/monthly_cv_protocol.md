# Monthly Cross-Validation Protocol

- Window type: `rolling`.
- Rolling train window months: `24`.
- Fold design: one out-of-sample calendar month per fold with strictly causal training history.
- Expanding mode uses all available history from the start of the sample through the month before the test month.
- Rolling mode uses only the trailing `rolling_train_months` calendar months before the test month.
- Within each fold, only returns realized in the test month are retained in the OOS backtest.
- Minimum train periods before an eligible month: `120`.
- Treasury duration assumption: `8.5`.
- Thresholds: buy `0.31`, sell `-0.31`.
- Transaction cost: `0.0` bps.
- Folds evaluated: `239`.
- First test month: `2006-02`.
- Last test month: `2025-12`.
- OOS observations concatenated: `239`.

## Outputs
- `/home/cristhian789/projects/meta-labeling-project/reports/results/monthly_cv/rolling_24m/monthly_cv_fold_summary.csv`
- `/home/cristhian789/projects/meta-labeling-project/reports/results/monthly_cv/rolling_24m/monthly_cv_oos_backtest.csv`
- `/home/cristhian789/projects/meta-labeling-project/reports/results/monthly_cv/rolling_24m/monthly_cv_summary.csv`