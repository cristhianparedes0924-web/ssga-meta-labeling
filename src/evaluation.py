from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

from src.data_loader import load_data
from src.features import create_indicators

FEATURE_COLUMNS = ["Z1_Mom", "Z2_Value", "Z3_Carry", "Z5_Vol", "Z4_Trend"]
MONTHS_PER_YEAR = 12.0
DECISION_POLICIES = {"threshold", "utility"}
PROBABILITY_CALIBRATIONS = {"none", "sigmoid", "isotonic"}


@dataclass
class TrainedModel:
    model: RandomForestClassifier | None
    constant_probability: float | None
    calibrated_model: CalibratedClassifierCV | None = None
    probability_calibration: str = "none"


def create_long_only_meta_dataset(df: pd.DataFrame, forward_window: int = 1) -> pd.DataFrame:
    """
    Build a long-only meta-labeling dataset.

    Trade_Signal:
        Primary model signal, long when momentum is positive.
    Meta_Label:
        1 if the long trade is profitable on the forward horizon, else 0.
        It is only defined where Trade_Signal is True.
    """
    if forward_window < 1:
        raise ValueError("forward_window must be >= 1")

    required_cols = {"Date", "SPX_Price", "Z1_Mom", *FEATURE_COLUMNS}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"Missing required columns: {missing}")

    labeled = df.copy()
    labeled["Future_Return"] = labeled["SPX_Price"].pct_change(forward_window).shift(-forward_window)
    labeled["Trade_Signal"] = labeled["Z1_Mom"] > 0
    labeled["Meta_Label"] = np.where(
        labeled["Trade_Signal"],
        (labeled["Future_Return"] > 0).astype(int),
        np.nan,
    )
    labeled = labeled.dropna(subset=["Future_Return"]).reset_index(drop=True)
    return labeled


def split_time_series(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1")
    if len(df) < 3:
        raise ValueError("Need at least 3 rows for train/validation/test split")

    train_end = int(len(df) * train_frac)
    val_end = int(len(df) * (train_frac + val_frac))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError("Split created an empty partition; adjust train_frac/val_frac")

    return train, val, test


def fit_success_model(
    train_events: pd.DataFrame,
    random_state: int = 42,
    n_estimators: int = 300,
    min_samples_leaf: int = 5,
    probability_calibration: str = "none",
) -> TrainedModel:
    if train_events.empty:
        raise ValueError("No training events available after filtering Trade_Signal")
    if probability_calibration not in PROBABILITY_CALIBRATIONS:
        choices = ", ".join(sorted(PROBABILITY_CALIBRATIONS))
        raise ValueError(f"probability_calibration must be one of: {choices}")

    X_train = train_events[FEATURE_COLUMNS]
    y_train = train_events["Meta_Label"].astype(int)

    if y_train.nunique() < 2:
        return TrainedModel(model=None, constant_probability=float(y_train.iloc[0]), probability_calibration="none")

    if probability_calibration == "none":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        return TrainedModel(model=model, constant_probability=None, probability_calibration="none")

    max_splits = len(train_events) - 1
    if max_splits < 2:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        return TrainedModel(model=model, constant_probability=None, probability_calibration="none")

    n_splits = min(3, max_splits)
    calibrated_model = CalibratedClassifierCV(
        estimator=RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        ),
        method=probability_calibration,
        cv=TimeSeriesSplit(n_splits=n_splits),
    )
    calibrated_model.fit(X_train, y_train)
    return TrainedModel(
        model=None,
        constant_probability=None,
        calibrated_model=calibrated_model,
        probability_calibration=probability_calibration,
    )


def predict_success_probability(trained_model: TrainedModel, features: pd.DataFrame) -> np.ndarray:
    if features.empty:
        return np.array([], dtype=float)

    if trained_model.constant_probability is not None:
        return np.full(len(features), trained_model.constant_probability, dtype=float)

    if trained_model.calibrated_model is not None:
        probabilities = trained_model.calibrated_model.predict_proba(features)[:, 1]
    elif trained_model.model is not None:
        probabilities = trained_model.model.predict_proba(features)[:, 1]
    else:
        raise ValueError("TrainedModel has neither constant_probability nor a fitted model")
    return probabilities.astype(float)


def period_probabilities(period_df: pd.DataFrame, trained_model: TrainedModel) -> pd.Series:
    probabilities = pd.Series(np.nan, index=period_df.index, dtype=float)
    event_mask = period_df["Trade_Signal"]
    event_features = period_df.loc[event_mask, FEATURE_COLUMNS]
    probabilities.loc[event_mask] = predict_success_probability(trained_model, event_features)
    return probabilities


def max_drawdown(returns: pd.Series) -> float:
    cumulative = (1.0 + returns).cumprod()
    running_peak = cumulative.cummax()
    drawdown = (cumulative - running_peak) / running_peak
    return float(drawdown.min())


def strategy_metrics(
    net_returns: pd.Series,
    trade_signal: pd.Series,
    take_trade: pd.Series,
    transaction_costs: pd.Series,
    position_changes: pd.Series,
    gross_returns: pd.Series | None = None,
) -> dict[str, float | int | None]:
    periods = len(net_returns)
    cumulative = (1.0 + net_returns).cumprod()
    final_value = float(cumulative.iloc[-1]) if periods > 0 else None
    years = periods / MONTHS_PER_YEAR if periods > 0 else None
    cagr = (final_value ** (1.0 / years) - 1.0) if (final_value is not None and years and years > 0) else None

    monthly_mean = float(net_returns.mean()) if periods > 0 else None
    monthly_std_raw = net_returns.std(ddof=1) if periods > 1 else 0.0
    monthly_std = float(monthly_std_raw) if not pd.isna(monthly_std_raw) else 0.0
    annualized_vol = monthly_std * np.sqrt(MONTHS_PER_YEAR)
    sharpe = (monthly_mean / monthly_std) * np.sqrt(MONTHS_PER_YEAR) if (monthly_mean is not None and monthly_std > 0) else None

    trades_available = int(trade_signal.sum())
    trades_taken = int(take_trade.sum())

    if trades_taken > 0:
        winning_trades = int(((net_returns > 0) & take_trade).sum())
        hit_rate = winning_trades / trades_taken
    else:
        hit_rate = None

    turnover_events = int(position_changes.sum())
    annualized_turnover = (turnover_events / years) if (years and years > 0) else None
    avg_exposure = float(take_trade.mean()) if periods > 0 else None
    total_transaction_cost = float(transaction_costs.sum())

    metrics: dict[str, float | int | None] = {
        "periods": periods,
        "trades_available": trades_available,
        "trades_taken": trades_taken,
        "hit_rate": _to_float_or_none(hit_rate),
        "avg_exposure": _to_float_or_none(avg_exposure),
        "turnover_events": turnover_events,
        "annualized_turnover": _to_float_or_none(annualized_turnover),
        "total_transaction_cost": _to_float_or_none(total_transaction_cost),
        "total_return": _to_float_or_none(final_value - 1.0) if final_value is not None else None,
        "final_value": _to_float_or_none(final_value),
        "cagr": _to_float_or_none(cagr),
        "annualized_volatility": _to_float_or_none(annualized_vol),
        "sharpe": _to_float_or_none(sharpe),
        "max_drawdown": _to_float_or_none(max_drawdown(net_returns)) if periods > 0 else None,
    }

    if gross_returns is not None and periods > 0:
        gross_cumulative = (1.0 + gross_returns).cumprod()
        gross_final_value = float(gross_cumulative.iloc[-1])
        metrics["gross_final_value"] = _to_float_or_none(gross_final_value)
        metrics["gross_total_return"] = _to_float_or_none(gross_final_value - 1.0)

    return metrics


def evaluate_period(
    period_df: pd.DataFrame,
    probabilities: pd.Series,
    threshold: float | None = None,
    transaction_cost_bps: float = 0.0,
    decision_policy: str = "threshold",
    utility_profile: dict[str, float | int] | None = None,
    utility_margin: float = 0.0,
    utility_risk_aversion: float = 0.0,
) -> dict[str, object]:
    if decision_policy not in DECISION_POLICIES:
        choices = ", ".join(sorted(DECISION_POLICIES))
        raise ValueError(f"decision_policy must be one of: {choices}")

    trade_signal = period_df["Trade_Signal"].astype(bool)
    utility_score_series: pd.Series | None = None

    if decision_policy == "threshold":
        if threshold is None:
            raise ValueError("threshold must be provided when decision_policy='threshold'")
        meta_take = trade_signal & (probabilities >= threshold)
        threshold_series = pd.Series(threshold, index=period_df.index, dtype=float)
    else:
        if utility_profile is None:
            raise ValueError("utility_profile must be provided when decision_policy='utility'")
        utility_score_series = utility_score_from_probabilities(
            probabilities=probabilities,
            utility_profile=utility_profile,
            transaction_cost_bps=transaction_cost_bps,
            utility_risk_aversion=utility_risk_aversion,
        )
        meta_take = trade_signal & (utility_score_series >= utility_margin)
        threshold_series = pd.Series(np.nan, index=period_df.index, dtype=float)

    return _evaluate_from_take_signals(
        period_df=period_df,
        primary_take=trade_signal,
        meta_take=meta_take,
        transaction_cost_bps=transaction_cost_bps,
        probabilities=probabilities,
        threshold_series=threshold_series,
        utility_score_series=utility_score_series,
    )


def threshold_sweep(
    val_df: pd.DataFrame,
    val_probabilities: pd.Series,
    thresholds: Iterable[float],
    objective: str,
    transaction_cost_bps: float = 0.0,
) -> tuple[float, pd.DataFrame]:
    rows = []

    for threshold in thresholds:
        results = evaluate_period(
            period_df=val_df,
            probabilities=val_probabilities,
            threshold=threshold,
            transaction_cost_bps=transaction_cost_bps,
        )
        meta_metrics = results["meta"]
        rows.append(
            {
                "threshold": float(threshold),
                "objective_value": _objective_value(meta_metrics, objective=objective),
                "final_value": meta_metrics["final_value"],
                "max_drawdown": meta_metrics["max_drawdown"],
                "sharpe": meta_metrics["sharpe"],
                "trades_taken": meta_metrics["trades_taken"],
                "total_transaction_cost": meta_metrics["total_transaction_cost"],
            }
        )

    sweep_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

    sortable = sweep_df.copy()
    sortable["objective_sort"] = sortable["objective_value"].fillna(-np.inf)
    sortable["final_sort"] = sortable["final_value"].fillna(-np.inf)
    sortable["drawdown_sort"] = sortable["max_drawdown"].fillna(-np.inf)

    best_row = sortable.sort_values(
        ["objective_sort", "final_sort", "drawdown_sort", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]

    return float(best_row["threshold"]), sweep_df


def estimate_utility_profile(period_df: pd.DataFrame) -> dict[str, float | int]:
    event_returns = period_df.loc[period_df["Trade_Signal"], "Future_Return"].dropna()
    if event_returns.empty:
        raise ValueError("Cannot estimate utility profile: period has zero Trade_Signal events")

    gains = event_returns[event_returns > 0]
    losses = event_returns[event_returns <= 0]
    avg_gain = float(gains.mean()) if not gains.empty else 0.0
    avg_loss = float((-losses).mean()) if not losses.empty else 0.0

    # Keep a small positive floor so utility scoring remains numerically stable.
    if avg_gain <= 0:
        avg_gain = 0.001
    if avg_loss <= 0:
        avg_loss = avg_gain

    return {
        "event_count": int(len(event_returns)),
        "positive_count": int(len(gains)),
        "negative_count": int(len(losses)),
        "avg_gain": float(avg_gain),
        "avg_loss": float(avg_loss),
    }


def utility_score_from_probabilities(
    probabilities: pd.Series,
    utility_profile: dict[str, float | int],
    transaction_cost_bps: float,
    utility_risk_aversion: float = 0.0,
) -> pd.Series:
    avg_gain = float(utility_profile["avg_gain"])
    avg_loss = float(utility_profile["avg_loss"])
    p = probabilities.astype(float)

    expected_return = p * avg_gain - (1.0 - p) * avg_loss
    uncertainty_penalty = utility_risk_aversion * (p * (1.0 - p)) * (avg_gain + avg_loss)
    cost_penalty = transaction_cost_bps / 10000.0
    score = expected_return - uncertainty_penalty - cost_penalty
    score[p.isna()] = np.nan
    return score.astype(float)


def utility_score_from_probability(
    probability: float,
    utility_profile: dict[str, float | int],
    transaction_cost_bps: float,
    utility_risk_aversion: float = 0.0,
) -> float:
    avg_gain = float(utility_profile["avg_gain"])
    avg_loss = float(utility_profile["avg_loss"])
    p = float(probability)
    expected_return = p * avg_gain - (1.0 - p) * avg_loss
    uncertainty_penalty = utility_risk_aversion * (p * (1.0 - p)) * (avg_gain + avg_loss)
    cost_penalty = transaction_cost_bps / 10000.0
    return float(expected_return - uncertainty_penalty - cost_penalty)


def run_evaluation(
    forward_window: int = 1,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    threshold_min: float = 0.30,
    threshold_max: float = 0.70,
    threshold_step: float = 0.01,
    objective: str = "sharpe",
    random_state: int = 42,
    transaction_cost_bps: float = 0.0,
    decision_policy: str = "threshold",
    probability_calibration: str = "none",
    utility_margin: float = 0.0,
    utility_risk_aversion: float = 0.0,
) -> tuple[dict[str, object], pd.DataFrame]:
    if threshold_step <= 0:
        raise ValueError("threshold_step must be > 0")
    if threshold_min > threshold_max:
        raise ValueError("threshold_min must be <= threshold_max")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be >= 0")
    if decision_policy not in DECISION_POLICIES:
        choices = ", ".join(sorted(DECISION_POLICIES))
        raise ValueError(f"decision_policy must be one of: {choices}")
    if probability_calibration not in PROBABILITY_CALIBRATIONS:
        choices = ", ".join(sorted(PROBABILITY_CALIBRATIONS))
        raise ValueError(f"probability_calibration must be one of: {choices}")

    raw_data = load_data()
    signals = create_indicators(raw_data)
    labeled = create_long_only_meta_dataset(signals, forward_window=forward_window)

    train_df, val_df, test_df = split_time_series(labeled, train_frac=train_frac, val_frac=val_frac)

    train_events = train_df[train_df["Trade_Signal"]].copy()
    val_events = val_df[val_df["Trade_Signal"]].copy()
    test_events = test_df[test_df["Trade_Signal"]].copy()

    if val_events.empty:
        raise ValueError("Validation split has zero Trade_Signal events; adjust split fractions.")
    if test_events.empty:
        raise ValueError("Test split has zero Trade_Signal events; adjust split fractions.")

    trained_model = fit_success_model(
        train_events,
        random_state=random_state,
        probability_calibration=probability_calibration,
    )

    val_probabilities = period_probabilities(val_df, trained_model)
    test_probabilities = period_probabilities(test_df, trained_model)

    selected_threshold: float | None = None
    sweep_df = pd.DataFrame()
    utility_profile: dict[str, float | int] | None = None

    if decision_policy == "threshold":
        thresholds = np.arange(threshold_min, threshold_max + (threshold_step / 2.0), threshold_step)
        selected_threshold, sweep_df = threshold_sweep(
            val_df=val_df,
            val_probabilities=val_probabilities,
            thresholds=thresholds,
            objective=objective,
            transaction_cost_bps=transaction_cost_bps,
        )
        val_results = evaluate_period(
            period_df=val_df,
            probabilities=val_probabilities,
            threshold=selected_threshold,
            transaction_cost_bps=transaction_cost_bps,
            decision_policy=decision_policy,
            utility_margin=utility_margin,
            utility_risk_aversion=utility_risk_aversion,
        )
        test_results = evaluate_period(
            period_df=test_df,
            probabilities=test_probabilities,
            threshold=selected_threshold,
            transaction_cost_bps=transaction_cost_bps,
            decision_policy=decision_policy,
            utility_margin=utility_margin,
            utility_risk_aversion=utility_risk_aversion,
        )
    else:
        utility_profile = estimate_utility_profile(val_df)
        val_results = evaluate_period(
            period_df=val_df,
            probabilities=val_probabilities,
            transaction_cost_bps=transaction_cost_bps,
            decision_policy=decision_policy,
            utility_profile=utility_profile,
            utility_margin=utility_margin,
            utility_risk_aversion=utility_risk_aversion,
        )
        test_results = evaluate_period(
            period_df=test_df,
            probabilities=test_probabilities,
            transaction_cost_bps=transaction_cost_bps,
            decision_policy=decision_policy,
            utility_profile=utility_profile,
            utility_margin=utility_margin,
            utility_risk_aversion=utility_risk_aversion,
        )

    threshold_series = test_results["threshold_series"]
    test_trade_log = _build_trade_log(
        period_df=test_df,
        probabilities=test_probabilities,
        threshold_series=threshold_series,
        primary_take=test_results["primary_take"],
        meta_take=test_results["meta_take"],
        primary_series=test_results["primary_series"],
        meta_series=test_results["meta_series"],
        utility_score_series=test_results["utility_score_series"],
    )

    report = {
        "config": {
            "mode": "static",
            "forward_window": forward_window,
            "train_frac": train_frac,
            "val_frac": val_frac,
            "threshold_min": threshold_min,
            "threshold_max": threshold_max,
            "threshold_step": threshold_step,
            "objective": objective,
            "random_state": random_state,
            "transaction_cost_bps": transaction_cost_bps,
            "decision_policy": decision_policy,
            "probability_calibration": trained_model.probability_calibration,
            "utility_margin": utility_margin,
            "utility_risk_aversion": utility_risk_aversion,
            "feature_columns": FEATURE_COLUMNS,
        },
        "data_summary": {
            "raw_rows": int(len(raw_data)),
            "signal_rows": int(len(signals)),
            "labeled_rows": int(len(labeled)),
        },
        "splits": {
            "train": _split_summary(train_df),
            "validation": _split_summary(val_df),
            "test": _split_summary(test_df),
        },
        "selected_threshold": selected_threshold,
        "validation_threshold_table": _records_with_native_types(sweep_df) if not sweep_df.empty else [],
        "utility_profile": utility_profile,
        "validation_metrics": {
            "primary": val_results["primary"],
            "meta": val_results["meta"],
            "diagnostics": val_results["diagnostics"],
        },
        "test_metrics": {
            "primary": test_results["primary"],
            "meta": test_results["meta"],
            "diagnostics": test_results["diagnostics"],
        },
    }

    return report, test_trade_log


def run_walk_forward_evaluation(
    forward_window: int = 1,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    threshold_min: float = 0.30,
    threshold_max: float = 0.70,
    threshold_step: float = 0.01,
    objective: str = "sharpe",
    random_state: int = 42,
    validation_window: int = 60,
    min_train_window: int = 120,
    transaction_cost_bps: float = 0.0,
    decision_policy: str = "threshold",
    probability_calibration: str = "none",
    utility_margin: float = 0.0,
    utility_risk_aversion: float = 0.0,
) -> tuple[dict[str, object], pd.DataFrame]:
    if threshold_step <= 0:
        raise ValueError("threshold_step must be > 0")
    if threshold_min > threshold_max:
        raise ValueError("threshold_min must be <= threshold_max")
    if validation_window < 1:
        raise ValueError("validation_window must be >= 1")
    if min_train_window < 1:
        raise ValueError("min_train_window must be >= 1")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be >= 0")
    if decision_policy not in DECISION_POLICIES:
        choices = ", ".join(sorted(DECISION_POLICIES))
        raise ValueError(f"decision_policy must be one of: {choices}")
    if probability_calibration not in PROBABILITY_CALIBRATIONS:
        choices = ", ".join(sorted(PROBABILITY_CALIBRATIONS))
        raise ValueError(f"probability_calibration must be one of: {choices}")

    raw_data = load_data()
    signals = create_indicators(raw_data)
    labeled = create_long_only_meta_dataset(signals, forward_window=forward_window)

    train_df, val_df, test_df = split_time_series(labeled, train_frac=train_frac, val_frac=val_frac)
    test_start_index = len(train_df) + len(val_df)
    thresholds = np.arange(threshold_min, threshold_max + (threshold_step / 2.0), threshold_step)

    wf_predictions = _walk_forward_predictions(
        labeled_df=labeled,
        start_index=test_start_index,
        thresholds=thresholds,
        objective=objective,
        validation_window=validation_window,
        min_train_window=min_train_window,
        random_state=random_state,
        transaction_cost_bps=transaction_cost_bps,
        decision_policy=decision_policy,
        probability_calibration=probability_calibration,
        utility_margin=utility_margin,
        utility_risk_aversion=utility_risk_aversion,
    )

    wf_predictions = wf_predictions.reindex(test_df.index)
    probabilities = wf_predictions["Model_Probability"]
    threshold_series = wf_predictions["Selected_Threshold"]
    utility_score_series = wf_predictions["Utility_Score"]
    meta_take = wf_predictions["Meta_Take_Trade"].fillna(False).astype(bool)
    primary_take = test_df["Trade_Signal"].astype(bool)

    test_results = _evaluate_from_take_signals(
        period_df=test_df,
        primary_take=primary_take,
        meta_take=meta_take,
        transaction_cost_bps=transaction_cost_bps,
        probabilities=probabilities,
        threshold_series=threshold_series,
        utility_score_series=utility_score_series,
    )

    test_trade_log = _build_trade_log(
        period_df=test_df,
        probabilities=probabilities,
        threshold_series=threshold_series,
        primary_take=primary_take,
        meta_take=meta_take,
        primary_series=test_results["primary_series"],
        meta_series=test_results["meta_series"],
        utility_score_series=utility_score_series,
    )
    test_trade_log["WalkForward_Status"] = wf_predictions["Status"].values

    threshold_summary = _threshold_summary(threshold_series) if decision_policy == "threshold" else None
    utility_score_summary = _utility_score_summary(utility_score_series) if decision_policy == "utility" else None
    status_counts = {
        str(key): int(value)
        for key, value in wf_predictions["Status"].value_counts(dropna=False).to_dict().items()
    }

    report = {
        "config": {
            "mode": "walk_forward",
            "forward_window": forward_window,
            "train_frac": train_frac,
            "val_frac": val_frac,
            "threshold_min": threshold_min,
            "threshold_max": threshold_max,
            "threshold_step": threshold_step,
            "objective": objective,
            "random_state": random_state,
            "validation_window": validation_window,
            "min_train_window": min_train_window,
            "transaction_cost_bps": transaction_cost_bps,
            "decision_policy": decision_policy,
            "probability_calibration": probability_calibration,
            "utility_margin": utility_margin,
            "utility_risk_aversion": utility_risk_aversion,
            "feature_columns": FEATURE_COLUMNS,
        },
        "data_summary": {
            "raw_rows": int(len(raw_data)),
            "signal_rows": int(len(signals)),
            "labeled_rows": int(len(labeled)),
        },
        "splits": {
            "train": _split_summary(train_df),
            "validation": _split_summary(val_df),
            "test": _split_summary(test_df),
        },
        "walk_forward_summary": {
            "test_rows": int(len(test_df)),
            "status_counts": status_counts,
        },
        "threshold_summary": threshold_summary,
        "utility_score_summary": utility_score_summary,
        "test_metrics": {
            "primary": test_results["primary"],
            "meta": test_results["meta"],
            "diagnostics": test_results["diagnostics"],
        },
    }

    return report, test_trade_log


def save_report(report: dict[str, object], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def save_test_trade_log(test_trade_log: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_trade_log.to_csv(output_path, index=False)
    return output_path


def _evaluate_from_take_signals(
    period_df: pd.DataFrame,
    primary_take: pd.Series,
    meta_take: pd.Series,
    transaction_cost_bps: float,
    probabilities: pd.Series | None = None,
    threshold_series: pd.Series | None = None,
    utility_score_series: pd.Series | None = None,
) -> dict[str, object]:
    primary_take = primary_take.reindex(period_df.index).fillna(False).astype(bool)
    meta_take = meta_take.reindex(period_df.index).fillna(False).astype(bool)
    trade_signal = period_df["Trade_Signal"].astype(bool)
    future_return = period_df["Future_Return"]

    primary_series = _strategy_return_series(future_return, primary_take, transaction_cost_bps=transaction_cost_bps)
    meta_series = _strategy_return_series(future_return, meta_take, transaction_cost_bps=transaction_cost_bps)

    profitable_signal = trade_signal & (future_return > 0)
    losing_signal = trade_signal & (future_return <= 0)

    diagnostics = {
        "profitable_signals": int(profitable_signal.sum()),
        "profitable_signals_taken": int((profitable_signal & meta_take).sum()),
        "profitable_signals_missed": int((profitable_signal & ~meta_take).sum()),
        "losing_signals_avoided": int((losing_signal & ~meta_take).sum()),
    }

    return {
        "primary": strategy_metrics(
            net_returns=primary_series["net"],
            trade_signal=trade_signal,
            take_trade=primary_take,
            transaction_costs=primary_series["costs"],
            position_changes=primary_series["position_changes"],
            gross_returns=primary_series["gross"],
        ),
        "meta": strategy_metrics(
            net_returns=meta_series["net"],
            trade_signal=trade_signal,
            take_trade=meta_take,
            transaction_costs=meta_series["costs"],
            position_changes=meta_series["position_changes"],
            gross_returns=meta_series["gross"],
        ),
        "diagnostics": diagnostics,
        "returns": {
            "primary": primary_series["net"],
            "meta": meta_series["net"],
            "primary_gross": primary_series["gross"],
            "meta_gross": meta_series["gross"],
        },
        "primary_take": primary_take,
        "meta_take": meta_take,
        "primary_series": primary_series,
        "meta_series": meta_series,
        "probabilities": probabilities,
        "threshold_series": threshold_series,
        "utility_score_series": utility_score_series,
    }


def _strategy_return_series(future_return: pd.Series, take_trade: pd.Series, transaction_cost_bps: float) -> dict[str, pd.Series]:
    take_trade = take_trade.astype(bool)
    gross_returns = pd.Series(np.where(take_trade, future_return, 0.0), index=future_return.index, dtype=float)

    take_int = take_trade.astype(int)
    position_changes = take_int.diff().abs().fillna(take_int).astype(int)
    cost_rate = transaction_cost_bps / 10000.0
    transaction_costs = pd.Series(position_changes * cost_rate, index=future_return.index, dtype=float)

    net_returns = gross_returns - transaction_costs
    return {
        "gross": gross_returns,
        "costs": transaction_costs,
        "net": net_returns,
        "position_changes": position_changes,
    }


def _build_trade_log(
    period_df: pd.DataFrame,
    probabilities: pd.Series,
    threshold_series: pd.Series,
    primary_take: pd.Series,
    meta_take: pd.Series,
    primary_series: dict[str, pd.Series],
    meta_series: dict[str, pd.Series],
    utility_score_series: pd.Series | None = None,
) -> pd.DataFrame:
    log = pd.DataFrame(
        {
            "Date": period_df["Date"],
            "Trade_Signal": period_df["Trade_Signal"].astype(int),
            "Meta_Label": period_df["Meta_Label"],
            "Future_Return": period_df["Future_Return"],
            "Model_Probability": probabilities.reindex(period_df.index),
            "Selected_Threshold": threshold_series.reindex(period_df.index),
            "Utility_Score": utility_score_series.reindex(period_df.index) if utility_score_series is not None else np.nan,
            "Primary_Take_Trade": primary_take.astype(int),
            "Meta_Take_Trade": meta_take.astype(int),
            "Primary_Gross_Return": primary_series["gross"],
            "Primary_Transaction_Cost": primary_series["costs"],
            "Primary_Return": primary_series["net"],
            "Meta_Gross_Return": meta_series["gross"],
            "Meta_Transaction_Cost": meta_series["costs"],
            "Meta_Return": meta_series["net"],
            "Cumulative_Primary": (1.0 + primary_series["net"]).cumprod(),
            "Cumulative_Meta": (1.0 + meta_series["net"]).cumprod(),
        }
    )
    return log.reset_index(drop=True)


def _walk_forward_predictions(
    labeled_df: pd.DataFrame,
    start_index: int,
    thresholds: np.ndarray,
    objective: str,
    validation_window: int,
    min_train_window: int,
    random_state: int,
    transaction_cost_bps: float,
    decision_policy: str,
    probability_calibration: str,
    utility_margin: float,
    utility_risk_aversion: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for idx in range(start_index, len(labeled_df)):
        current = labeled_df.iloc[idx]
        trade_signal = bool(current["Trade_Signal"])
        history = labeled_df.iloc[:idx]

        selected_threshold: float | None = None
        model_probability: float | None = None
        utility_score: float | None = None
        status = "ok"

        if len(history) < (min_train_window + validation_window):
            status = "insufficient_history"
        else:
            train_hist = history.iloc[:-validation_window]
            val_hist = history.iloc[-validation_window:]
            train_events = train_hist[train_hist["Trade_Signal"]].copy()
            val_events = val_hist[val_hist["Trade_Signal"]].copy()

            if train_events.empty:
                status = "no_train_events"
            elif val_events.empty:
                status = "no_validation_events"
            else:
                trained_model = fit_success_model(
                    train_events,
                    random_state=random_state,
                    probability_calibration=probability_calibration,
                )
                val_probabilities = period_probabilities(val_hist, trained_model)
                current_row = labeled_df.iloc[[idx]]
                current_probability = period_probabilities(current_row, trained_model).iloc[0]
                if pd.notna(current_probability):
                    model_probability = float(current_probability)

                if decision_policy == "threshold":
                    selected_threshold, _ = threshold_sweep(
                        val_df=val_hist,
                        val_probabilities=val_probabilities,
                        thresholds=thresholds,
                        objective=objective,
                        transaction_cost_bps=transaction_cost_bps,
                    )
                else:
                    utility_profile = estimate_utility_profile(val_hist)
                    if model_probability is not None:
                        utility_score = utility_score_from_probability(
                            probability=model_probability,
                            utility_profile=utility_profile,
                            transaction_cost_bps=transaction_cost_bps,
                            utility_risk_aversion=utility_risk_aversion,
                        )

        if decision_policy == "threshold":
            meta_take_trade = bool(
                trade_signal
                and (selected_threshold is not None)
                and (model_probability is not None)
                and (model_probability >= selected_threshold)
            )
        else:
            meta_take_trade = bool(
                trade_signal
                and (utility_score is not None)
                and (utility_score >= utility_margin)
            )

        records.append(
            {
                "index": idx,
                "Trade_Signal": trade_signal,
                "Model_Probability": model_probability,
                "Selected_Threshold": selected_threshold,
                "Utility_Score": utility_score,
                "Meta_Take_Trade": meta_take_trade,
                "Status": status,
            }
        )

    predictions = pd.DataFrame(records).set_index("index")
    return predictions


def _split_summary(df: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(df)),
        "start_date": _date_or_none(df["Date"].iloc[0]),
        "end_date": _date_or_none(df["Date"].iloc[-1]),
        "trade_signal_rows": int(df["Trade_Signal"].sum()),
    }


def _threshold_summary(threshold_series: pd.Series) -> dict[str, float | int | None]:
    values = threshold_series.dropna()
    if values.empty:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
        }

    std_raw = values.std(ddof=0)
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(std_raw) if not pd.isna(std_raw) else None,
    }


def _utility_score_summary(score_series: pd.Series) -> dict[str, float | int | None]:
    values = score_series.dropna()
    if values.empty:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
        }

    std_raw = values.std(ddof=0)
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(std_raw) if not pd.isna(std_raw) else None,
    }


def _date_or_none(value: object) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _objective_value(metrics: dict[str, float | int | None], objective: str) -> float | None:
    if objective == "sharpe":
        return _to_float_or_none(metrics.get("sharpe"))
    if objective == "cagr":
        return _to_float_or_none(metrics.get("cagr"))
    if objective == "final_value":
        return _to_float_or_none(metrics.get("final_value"))
    raise ValueError("objective must be one of: sharpe, cagr, final_value")


def _to_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    return float(value)


def _records_with_native_types(df: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in df.to_dict(orient="records"):
        clean_row: dict[str, object] = {}
        for key, value in row.items():
            if pd.isna(value):
                clean_row[key] = None
            elif isinstance(value, (np.floating, float)):
                clean_row[key] = float(value)
            elif isinstance(value, (np.integer, int)):
                clean_row[key] = int(value)
            else:
                clean_row[key] = value
        records.append(clean_row)
    return records
