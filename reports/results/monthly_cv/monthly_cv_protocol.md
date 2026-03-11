# Monthly Cross-Validation Protocol

- Fold design: expanding history with one out-of-sample calendar month per fold.
- Training history at fold start uses only observations strictly before the test month.
- Within each fold, only returns realized in that test month are retained in the OOS backtest.
- Minimum train periods before an eligible month: `120`.
- Treasury duration assumption: `8.5`.
- Thresholds: buy `0.31`, sell `-0.31`.
- Transaction cost: `0.0` bps.
- Folds evaluated: `239`.
- First test month: `2006-02`.
- Last test month: `2025-12`.
- OOS observations concatenated: `239`.

## Outputs
- `C:\Users\crist\OneDrive\Documents\primary_model_test\meta-labeling-project\reports\results\monthly_cv\monthly_cv_fold_summary.csv`
- `C:\Users\crist\OneDrive\Documents\primary_model_test\meta-labeling-project\reports\results\monthly_cv\monthly_cv_oos_backtest.csv`
- `C:\Users\crist\OneDrive\Documents\primary_model_test\meta-labeling-project\reports\results\monthly_cv\monthly_cv_summary.csv`