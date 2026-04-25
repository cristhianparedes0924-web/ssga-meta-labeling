# Execution Sequence Diagrams

This document provides sequence-style diagrams for the two main evaluation paths:
- `static` mode
- `walk_forward` mode

It also highlights where `threshold` and `utility` decision policies diverge.

## 1) Static Evaluation Sequence

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant CLI as scripts/run_evaluation.py
    participant Eval as src/evaluation.py
    participant Loader as src/data_loader.py
    participant Feat as src/features.py
    participant Model as RandomForest (+ optional calibration)
    participant Reports as reports/*.json + reports/*.csv

    User->>CLI: python scripts/run_evaluation.py --mode static ...
    CLI->>Eval: run_evaluation(config)
    Eval->>Loader: load_data()
    Loader-->>Eval: Aligned monthly prices DataFrame
    Eval->>Feat: create_indicators(df)
    Feat-->>Eval: Feature DataFrame (Z1..Z5)
    Eval->>Eval: create_long_only_meta_dataset()
    Eval->>Eval: split_time_series(train/val/test)
    Eval->>Model: fit_success_model(train events)
    Model-->>Eval: TrainedModel
    Eval->>Eval: period_probabilities(val/test)

    alt decision_policy == threshold
        Eval->>Eval: threshold_sweep(validation)
        Eval->>Eval: selected_threshold
        Eval->>Eval: evaluate_period(test, threshold)
    else decision_policy == utility
        Eval->>Eval: estimate_utility_profile(validation)
        Eval->>Eval: evaluate_period(test, utility_profile, margin, risk_aversion)
    end

    Eval->>Eval: _build_trade_log()
    Eval-->>CLI: report + test_trade_log
    CLI->>Reports: save_report(...)
    CLI->>Reports: save_test_trade_log(...)
    CLI-->>User: Summary metrics printed
```

## 2) Walk-Forward Evaluation Sequence

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant CLI as scripts/run_evaluation.py
    participant Eval as src/evaluation.py
    participant Loader as src/data_loader.py
    participant Feat as src/features.py
    participant Model as RandomForest (+ optional calibration)
    participant Reports as reports/*.json + reports/*.csv

    User->>CLI: python scripts/run_evaluation.py --mode walk_forward ...
    CLI->>Eval: run_walk_forward_evaluation(config)
    Eval->>Loader: load_data()
    Loader-->>Eval: Aligned monthly prices DataFrame
    Eval->>Feat: create_indicators(df)
    Feat-->>Eval: Feature DataFrame (Z1..Z5)
    Eval->>Eval: create_long_only_meta_dataset()
    Eval->>Eval: split_time_series(train/val/test)
    Eval->>Eval: _walk_forward_predictions(start=test_start_index)

    loop each test month t
        Eval->>Eval: Build rolling train_hist + val_hist
        alt insufficient history / no events
            Eval->>Eval: Status = insufficient_history or no_*_events
        else usable window
            Eval->>Model: fit_success_model(train_hist events)
            Model-->>Eval: TrainedModel
            Eval->>Eval: probability for month t

            alt decision_policy == threshold
                Eval->>Eval: threshold_sweep(val_hist)
                Eval->>Eval: take trade if p >= selected_threshold
            else decision_policy == utility
                Eval->>Eval: estimate_utility_profile(val_hist)
                Eval->>Eval: utility_score_from_probability(p)
                Eval->>Eval: take trade if utility_score >= margin
            end
        end
    end

    Eval->>Eval: _evaluate_from_take_signals(test horizon)
    Eval->>Eval: _build_trade_log() + status column
    Eval-->>CLI: report + test_trade_log
    CLI->>Reports: save_report(...)
    CLI->>Reports: save_test_trade_log(...)
    CLI-->>User: Summary metrics printed
```

## 3) Notebook Path (`notebooks/02_unbiased_evaluation.ipynb`)

```mermaid
sequenceDiagram
    autonumber
    actor Analyst
    participant NB as notebooks/02_unbiased_evaluation.ipynb
    participant Eval as src/evaluation.py
    participant Reports as reports/*_from_notebook.*

    Analyst->>NB: Set config (policy/calibration/costs/windows)
    NB->>Eval: run_evaluation(...)
    Eval-->>NB: static_report + static_trade_log
    NB->>Eval: run_walk_forward_evaluation(...)
    Eval-->>NB: wf_report + wf_trade_log
    NB->>NB: Plot metrics and cumulative curves
    NB->>Reports: save_report(...) + save_test_trade_log(...)
```

## 4) Practical Interpretation

- `static` mode calibrates once, then evaluates once on held-out test.
- `walk_forward` mode recalibrates repeatedly before each test month.
- `threshold` policy is probability gating.
- `utility` policy is expected-value gating with explicit cost and uncertainty penalty controls.

