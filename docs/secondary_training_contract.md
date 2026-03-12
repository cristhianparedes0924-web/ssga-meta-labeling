# Secondary Training Contract

## Purpose

Define the first-pass secondary-model training contract before any secondary-model training code is added.

The first secondary model is intended to filter false-positive primary actions, not to replace the protected primary baseline and not to change position sizing.

## Training Unit

- One training observation is one actionable primary event from the current secondary dataset
- Actionable means `primary_signal` is `BUY` or `SELL`
- `HOLD` rows are excluded from first-pass training

This matches the current dataset-builder default in `src/metalabel/secondary/dataset.py`.

## Target

- Classification target: `meta_label`
- Return field retained for analysis: `meta_target_return`

Definitions:

- `meta_target_return` is the realized next-period net return from the primary-strategy weights implied at decision time `t`
- `meta_label = 1` if `meta_target_return > 0`
- `meta_label = 0` otherwise

The first-pass secondary model predicts whether an actionable primary event should be taken, using the existing target contract already implemented in the repo.

## Feature Inclusion Rule

Training features may only use columns available at decision time `t`.

Allowed feature groups:

1. Primary-model information already present in the dataset:
   - `primary_signal`
   - `composite_score`
   - raw indicator columns from the primary signal pipeline
   - indicator z-score columns from the primary signal pipeline
2. Trailing primary-performance-health features already present in the dataset:
   - trailing hit rate
   - trailing average net return
   - trailing volatility of net returns
   - trailing average turnover
   - `signal_streak`
3. Market-state / regime features already available in the dataset:
   - decision-time indicator and z-score state already produced by the primary pipeline
4. Primary implied weights, if used conservatively:
   - `weight_*` columns may be included in the first pass because they are known at decision time and help identify the primary action regime

## Feature Exclusion Rule

Do not train on leakage-prone, post-outcome, or identity-only columns.

Explicitly exclude:

- `date`
- `realized_date`
- `meta_label`
- `meta_target_return`
- `meta_target_gross_return`
- `event_turnover`

Rationale:

- `realized_date`, `meta_target_return`, and `meta_label` are post-decision outcomes
- `event_turnover` is realized from the executed transition and should be treated as evaluation information, not a training predictor
- `date` is an identifier; it may be used for splitting and reporting, but not as a raw training feature

## Split Protocol

Use the existing causal secondary split utilities in `src/metalabel/secondary/splits.py`.

Default protocol for first-pass training:

- Use `expanding_forward_splits(...)` as the main validation method
- Keep chronological ordering
- Do not shuffle
- Require validation rows to occur strictly after training rows
- Keep same-date rows in the same split

Allowed helper for a smaller final check:

- `holdout_split_by_time(...)`

## First Model Family

Use a regularized logistic regression classifier as the first baseline model.

Why this first:

- simple and conservative
- appropriate for the existing binary `meta_label`
- easy to interpret and debug
- available through the current dependency stack

The first training pass should use the model only as an event filter. Position sizing remains unchanged and out of scope.

## Evaluation Metrics

Evaluate both classification quality and strategy-level impact.

Classification metrics:

- precision on `meta_label = 1`
- recall on `meta_label = 1`
- F1 score
- ROC AUC, if probability outputs are used
- confusion matrix

Strategy-level comparison:

- event count kept vs rejected
- kept-event hit rate
- kept-event average `meta_target_return`
- filtered-strategy return series constructed by accepting primary actions only when the secondary model predicts accept
- annualized return, annualized vol, Sharpe, and max drawdown for the filtered strategy

## Comparison Versus Protected Primary Baseline

The primary baseline remains unchanged.

Comparison protocol:

- Build the current secondary dataset with the existing dataset builder
- Run causal validation with the existing split utilities
- Train the secondary classifier only on training rows
- Generate predictions only for later validation rows
- Compare:
  - unchanged primary baseline
  - filtered primary strategy where the secondary model only decides whether to allow or reject the original primary action

For the first pass:

- do not change primary signal generation
- do not change primary portfolio weights
- do not add position sizing
- do not add threshold tuning as a separate phase

## Non-Goals

Out of scope for this phase:

- model ensembling
- threshold optimization
- probability calibration
- position sizing
- trade sizing by confidence
- secondary inference pipeline
- dashboard/report generation
- replacing the primary model

## Implementation Notes For The Next Task

- Start from `build_secondary_dataset(...)`
- Select training rows where `primary_signal` is actionable
- Drop the excluded columns before fitting
- Encode `primary_signal` in a simple reproducible way
- Use `expanding_forward_splits(...)` for validation first
- Report both classifier metrics and filtered-strategy metrics against the unchanged primary baseline
