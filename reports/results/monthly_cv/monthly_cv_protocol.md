# Monthly Cross-Validation Protocol

- Fold design: expanding history with `1` out-of-sample calendar month(s) per fold.
- Training history at fold start uses only observations strictly before the first test month.
- Within each fold, only returns realized inside that fold's test window are retained in the OOS backtest.
- Minimum train periods before an eligible month: `120`.
- Test window months per fold: `1`.
- Treasury duration assumption: `8.5`.
- Thresholds: buy `0.31`, sell `-0.31`.
- Transaction cost: `0.0` bps.
- Folds evaluated: `239`.
- First OOS window: `2006-02`.
- Last OOS window: `2025-12`.
- OOS observations concatenated: `239`.

## Outputs
- `/home/cristhian789/projects/meta-labeling-project/reports/results/monthly_cv/monthly_cv_fold_summary.csv`
- `/home/cristhian789/projects/meta-labeling-project/reports/results/monthly_cv/monthly_cv_oos_backtest.csv`
- `/home/cristhian789/projects/meta-labeling-project/reports/results/monthly_cv/monthly_cv_summary.csv`