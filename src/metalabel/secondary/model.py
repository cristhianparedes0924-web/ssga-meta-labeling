"""M2 secondary meta-labeling classifier.

Trains a logistic-regression classifier to predict whether a primary-model
(M1) signal will be profitable.  All training uses a causal expanding-window
walk-forward so no future information is used.

Usage
-----
    from metalabel.secondary.model import run_walk_forward

    predictions = run_walk_forward(df)   # df = secondary dataset
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from metalabel.secondary.split import walk_forward_splits


# ---------------------------------------------------------------------------
# Feature specification
# ---------------------------------------------------------------------------

#: Full 22-feature set including VIX and OAS regime features.
M2_FEATURES: list[str] = [
    # Primary model output context
    "primary_signal_buy",       # BUY=1 / SELL=0
    "composite_score",
    # Primary z-scored indicators
    "spx_trend_z",
    "bcom_trend_z",
    "credit_vs_rates_z",
    "risk_breadth_z",
    "bcom_accel_z",
    "yield_mom_z",
    # Trailing performance of M1
    "trailing_hit_rate_12",
    "trailing_avg_net_return_12",
    "trailing_vol_net_return_12",
    "signal_streak",
    # VIX regime features
    "vix_level_z",
    "vix_change_z",
    "vix_trend",
    "vix_high_regime",
    "vix_rising",
    # OAS / liquidity regime features
    "oas_level_z",
    "oas_change_z",
    "oas_trend",
    "oas_wide_regime",
    "oas_widening",
]

#: Core 12-feature set — M1 context and track record only.
#: Drops VIX and OAS features which EDA showed carry no independent signal.
M2_FEATURES_CORE: list[str] = [
    "primary_signal_buy",
    "composite_score",
    "spx_trend_z",
    "bcom_trend_z",
    "credit_vs_rates_z",
    "risk_breadth_z",
    "bcom_accel_z",
    "yield_mom_z",
    "trailing_hit_rate_12",
    "trailing_avg_net_return_12",
    "trailing_vol_net_return_12",
    "signal_streak",
]

_TARGET = "meta_label"


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Return a feature matrix ready for sklearn.

    * Encodes ``primary_signal`` as ``primary_signal_buy`` (BUY=1, SELL=0).
    * Selects only the columns listed in ``features`` (defaults to M2_FEATURES).
    * Fills known NaN positions (early bcom_accel_z) with 0.

    Parameters
    ----------
    df:
        Secondary dataset rows (may include non-feature columns).
    features:
        Feature list to use. Defaults to ``M2_FEATURES`` (all 22 features).
        Pass ``M2_FEATURES_CORE`` to drop VIX/OAS columns.

    Returns
    -------
    DataFrame with exactly the requested feature columns, no NaNs.
    """
    if features is None:
        features = M2_FEATURES

    out = df.copy()
    out["primary_signal_buy"] = (out["primary_signal"] == "BUY").astype(float)

    missing = [c for c in features if c not in out.columns]
    if missing:
        raise KeyError(f"Secondary dataset is missing M2 feature columns: {missing}")

    X = out[features].fillna(0.0)
    return X


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _make_model() -> LogisticRegression:
    """Return a fresh, unfitted logistic-regression classifier."""
    return LogisticRegression(
        C=0.5,
        max_iter=500,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )


def _make_ridge_model() -> LogisticRegressionCV:
    """Return a Ridge logistic regression with C tuned by inner cross-validation.

    Searches over Cs = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0] using 5-fold
    stratified CV within each walk-forward training window, selecting the C
    that maximises ROC-AUC.  All other settings match the baseline model.
    """
    return LogisticRegressionCV(
        Cs=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        cv=5,
        penalty="l2",
        scoring="roc_auc",
        solver="lbfgs",
        max_iter=500,
        class_weight="balanced",
        random_state=42,
    )


# ---------------------------------------------------------------------------
# Walk-forward prediction
# ---------------------------------------------------------------------------

def run_walk_forward(
    df: pd.DataFrame,
    min_train_size: int = 60,
    step: int = 1,
    threshold: float = 0.5,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Run expanding-window walk-forward prediction with M2.

    At each step:
    1. Fit a logistic regression on all events seen so far.
    2. Predict the probability of M1 winning on the next event(s).
    3. Apply ``threshold`` to produce a binary M2 recommendation.

    Parameters
    ----------
    df:
        Time-ordered secondary dataset (earliest row first).
    min_train_size:
        Events used for the first training window (not predicted).
    step:
        Events predicted per walk-forward step.
    threshold:
        Probability cut-off above which M2 says "trust M1" (m2_approve=1).
    features:
        Feature list to use. Defaults to ``M2_FEATURES``.
        Pass ``M2_FEATURES_CORE`` to drop VIX/OAS columns.

    Returns
    -------
    DataFrame with one row per out-of-sample event containing:
        date, primary_signal, meta_label,
        m2_prob   – P(M1 wins | features),
        m2_approve – 1 if m2_prob >= threshold else 0,
        meta_target_return – realised net return of M1 at that event.
    """
    df = df.sort_values("date").reset_index(drop=True)
    records: list[dict] = []

    for train_df, test_df in walk_forward_splits(df, min_train_size=min_train_size, step=step):
        X_train = prepare_features(train_df, features=features).values
        y_train = train_df[_TARGET].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        model = _make_model()
        model.fit(X_train_s, y_train)

        X_test = prepare_features(test_df, features=features).values
        X_test_s = scaler.transform(X_test)

        probs = model.predict_proba(X_test_s)[:, 1]

        for i, (_, row) in enumerate(test_df.iterrows()):
            records.append(
                {
                    "date": row["date"],
                    "primary_signal": row["primary_signal"],
                    "meta_label": int(row[_TARGET]),
                    "meta_target_return": float(row["meta_target_return"]),
                    "m2_prob": float(probs[i]),
                    "m2_approve": int(probs[i] >= threshold),
                }
            )

    return pd.DataFrame(records)


def run_walk_forward_ridge(
    df: pd.DataFrame,
    min_train_size: int = 60,
    step: int = 1,
    threshold: float = 0.5,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, list[float]]:
    """Walk-forward prediction using Ridge logistic regression with tuned C.

    Same as ``run_walk_forward`` but uses ``LogisticRegressionCV`` to select
    the best regularisation strength at each step via inner 5-fold CV.

    Returns
    -------
    predictions : DataFrame  (same schema as run_walk_forward)
    c_values    : list of C values chosen at each walk-forward step
    """
    df = df.sort_values("date").reset_index(drop=True)
    records: list[dict] = []
    c_values: list[float] = []

    for train_df, test_df in walk_forward_splits(df, min_train_size=min_train_size, step=step):
        X_train = prepare_features(train_df, features=features).values
        y_train = train_df[_TARGET].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        model = _make_ridge_model()
        model.fit(X_train_s, y_train)
        c_values.append(float(model.C_[0]))

        X_test = prepare_features(test_df, features=features).values
        X_test_s = scaler.transform(X_test)
        probs = model.predict_proba(X_test_s)[:, 1]

        for i, (_, row) in enumerate(test_df.iterrows()):
            records.append(
                {
                    "date": row["date"],
                    "primary_signal": row["primary_signal"],
                    "meta_label": int(row[_TARGET]),
                    "meta_target_return": float(row["meta_target_return"]),
                    "m2_prob": float(probs[i]),
                    "m2_approve": int(probs[i] >= threshold),
                }
            )

    return pd.DataFrame(records), c_values


def run_walk_forward_rf(
    df: pd.DataFrame,
    min_train_size: int = 60,
    step: int = 1,
    threshold: float = 0.5,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Walk-forward prediction using a Random Forest classifier with tuned hyperparameters.

    At each step, fits a RandomForestClassifier with hyperparameters selected
    via 5-fold stratified GridSearchCV within the training window only.
    No scaling is applied (trees are scale-invariant).

    Grid searched:
        n_estimators : [100, 200]
        max_depth    : [2, 3, 4]
        min_samples_leaf : [5, 10]

    Parameters
    ----------
    df : time-ordered secondary dataset.
    min_train_size : events used for first training window.
    step : events predicted per walk-forward step.
    threshold : probability cut-off for m2_approve.
    features : feature list. Defaults to M2_FEATURES_CORE.

    Returns
    -------
    predictions : DataFrame (same schema as run_walk_forward)
    best_params  : list of dicts, one per walk-forward step
    """
    if features is None:
        features = M2_FEATURES_CORE

    df = df.sort_values("date").reset_index(drop=True)
    records: list[dict] = []
    best_params: list[dict] = []

    param_grid = {
        "n_estimators":      [100, 200],
        "max_depth":         [2, 3, 4],
        "min_samples_leaf":  [5, 10],
    }

    for train_df, test_df in walk_forward_splits(df, min_train_size=min_train_size, step=step):
        X_train = prepare_features(train_df, features=features).values
        y_train = train_df[_TARGET].values

        base_rf = RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        cv = GridSearchCV(
            base_rf,
            param_grid,
            cv=5,
            scoring="roc_auc",
            refit=True,
            n_jobs=-1,
        )
        cv.fit(X_train, y_train)
        best_params.append(cv.best_params_)

        X_test = prepare_features(test_df, features=features).values
        probs = cv.predict_proba(X_test)[:, 1]

        for i, (_, row) in enumerate(test_df.iterrows()):
            records.append(
                {
                    "date": row["date"],
                    "primary_signal": row["primary_signal"],
                    "meta_label": int(row[_TARGET]),
                    "meta_target_return": float(row["meta_target_return"]),
                    "m2_prob": float(probs[i]),
                    "m2_approve": int(probs[i] >= threshold),
                }
            )

    return pd.DataFrame(records), best_params


def run_walk_forward_svm(
    df: pd.DataFrame,
    min_train_size: int = 60,
    step: int = 1,
    threshold: float = 0.5,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Walk-forward prediction using SVM with RBF kernel.

    Uses SVC with probability=True (Platt scaling) to produce calibrated
    probabilities. C and gamma are tuned via 5-fold stratified GridSearchCV
    within each training window. Features are scaled before fitting.

    Grid searched:
        C     : [0.1, 1.0, 10.0]
        gamma : ['scale', 0.1, 0.01]

    Parameters
    ----------
    df : time-ordered secondary dataset.
    min_train_size : events used for first training window.
    step : events predicted per walk-forward step.
    threshold : probability cut-off for m2_approve.
    features : feature list. Defaults to M2_FEATURES_CORE.

    Returns
    -------
    predictions : DataFrame (same schema as run_walk_forward)
    best_params  : list of dicts, one per walk-forward step
    """
    if features is None:
        features = M2_FEATURES_CORE

    df = df.sort_values("date").reset_index(drop=True)
    records: list[dict] = []
    best_params: list[dict] = []

    param_grid = {
        "C":     [0.1, 1.0, 10.0],
        "gamma": ["scale", 0.1, 0.01],
    }

    for train_df, test_df in walk_forward_splits(df, min_train_size=min_train_size, step=step):
        X_train = prepare_features(train_df, features=features).values
        y_train = train_df[_TARGET].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        base_svm = SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )
        cv = GridSearchCV(
            base_svm,
            param_grid,
            cv=5,
            scoring="roc_auc",
            refit=True,
        )
        cv.fit(X_train_s, y_train)
        best_params.append(cv.best_params_)

        X_test_s = scaler.transform(prepare_features(test_df, features=features).values)
        probs = cv.predict_proba(X_test_s)[:, 1]

        for i, (_, row) in enumerate(test_df.iterrows()):
            records.append(
                {
                    "date": row["date"],
                    "primary_signal": row["primary_signal"],
                    "meta_label": int(row[_TARGET]),
                    "meta_target_return": float(row["meta_target_return"]),
                    "m2_prob": float(probs[i]),
                    "m2_approve": int(probs[i] >= threshold),
                }
            )

    return pd.DataFrame(records), best_params


def sweep_thresholds(
    predictions: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Evaluate M2 economic performance across a range of approval thresholds.

    Parameters
    ----------
    predictions:
        Output of ``run_walk_forward`` (must contain m2_prob and meta_target_return).
    thresholds:
        List of thresholds to test. Defaults to 0.1 through 0.9 in steps of 0.05.

    Returns
    -------
    DataFrame with one row per threshold showing n_approved, win_rate,
    avg_return, sharpe, and roc_auc.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.10, 0.95, 0.05)]

    rows = []
    returns = predictions["meta_target_return"].values

    m1_sharpe = (
        float(np.mean(returns) / np.std(returns) * np.sqrt(12))
        if np.std(returns) > 0 else 0.0
    )

    for t in thresholds:
        approved = predictions[predictions["m2_prob"] >= t]
        n = len(approved)

        if n < 2:
            rows.append({"threshold": t, "n_approved": n,
                         "win_rate": None, "avg_return": None, "sharpe": None})
            continue

        wr  = float(approved["meta_label"].mean())
        ret = float(approved["meta_target_return"].mean())
        std = float(approved["meta_target_return"].std())
        sh  = float(ret / std * np.sqrt(12)) if std > 0 else 0.0

        rows.append({
            "threshold":  t,
            "n_approved": n,
            "pct_traded": round(n / len(predictions), 3),
            "win_rate":   round(wr, 4),
            "avg_return": round(ret, 6),
            "sharpe":     round(sh, 3),
            "vs_m1_sharpe": round(sh - m1_sharpe, 3),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def compute_carry_returns(
    predictions: pd.DataFrame,
    asset_rets: pd.DataFrame,
) -> np.ndarray:
    """Compute the carry-forward return at each OOS step.

    When M2 sizes down (position_size < 1), the unallocated fraction is not
    sitting in cash - it is still invested in the PREVIOUS period's M1
    allocation.  This function computes what that previous allocation earns
    at each step using actual asset returns.

    Parameters
    ----------
    predictions:
        Output of ``run_walk_forward`` merged with weight columns and
        realized_date from the secondary dataset.  Required columns:
        realized_date, weight_spx, weight_bcom, weight_treasury_10y,
        weight_corp_bonds, meta_target_return.
    asset_rets:
        DataFrame indexed by date with columns:
        spx, bcom, corp_bonds, treasury_10y.

    Returns
    -------
    carry : np.ndarray of shape (n,)
        Carry-forward return at each step. At step 0 (no prior allocation)
        falls back to M1's own return.
    """
    required = ["realized_date", "weight_spx", "weight_bcom",
                "weight_treasury_10y", "weight_corp_bonds", "meta_target_return"]
    missing = [c for c in required if c not in predictions.columns]
    if missing:
        raise KeyError(
            f"predictions is missing columns needed for carry-forward: {missing}. "
            "Merge weight columns from the secondary dataset before calling."
        )

    weights  = predictions[["weight_spx", "weight_bcom",
                             "weight_treasury_10y", "weight_corp_bonds"]].values
    m1_rets  = predictions["meta_target_return"].values

    def _asset_ret(realized_date):
        if realized_date in asset_rets.index:
            return asset_rets.loc[realized_date]
        idx = asset_rets.index.get_indexer([realized_date], method="nearest")[0]
        return asset_rets.iloc[idx]

    # Asset return matrix aligned to OOS events
    # column order: spx=0, bcom=1, corp_bonds=2, treasury_10y=3
    A = np.array([_asset_ret(rd).values
                  for rd in predictions["realized_date"]])

    carry = np.zeros(len(predictions))
    for i in range(len(predictions)):
        if i == 0:
            carry[i] = m1_rets[i]          # no prior allocation, fallback
        else:
            pw = weights[i - 1]            # prev weights: [spx, bcom, tsy, corp]
            ca = A[i]                      # current asset rets: [spx, bcom, corp, tsy]
            carry[i] = (pw[0] * ca[0] +   # spx
                        pw[1] * ca[1] +   # bcom
                        pw[2] * ca[3] +   # treasury_10y (col 3 in A)
                        pw[3] * ca[2])    # corp_bonds   (col 2 in A)
    return carry


def apply_position_sizing(
    predictions: pd.DataFrame,
    normalize: bool = True,
    carry_returns: np.ndarray | None = None,
) -> pd.DataFrame:
    """Replace binary gate with probability-scaled position sizes.

    Instead of approve/reject, M2's probability scales how much of each
    trade is taken.

    Sizing formula (normalize=True, default):
        position_size[t] = m2_prob[t] / expanding_mean(m2_prob[0:t])

        Uses an expanding mean: at each step t, divides by the mean of all
        OOS probabilities seen so far (steps 0 through t-1), not the global
        mean computed over the full sample. This avoids any look-ahead bias
        since the normalisation denominator at step t uses no future data.
        At step 0 (no prior OOS history) falls back to dividing by the
        probability itself, giving size = 1.0.

    Return formula:
        If carry_returns provided (correct economic behavior):
            sized_return = size * m1_return + (1 - size) * carry_return
            The unallocated fraction earns the previous period's M1
            allocation return, not 0%.
        If carry_returns is None (legacy):
            sized_return = size * m1_return
            The unallocated fraction implicitly earns 0%.

    Parameters
    ----------
    predictions:
        Output of ``run_walk_forward`` (must contain m2_prob and
        meta_target_return).
    normalize:
        Whether to rescale sizes using the expanding mean of past OOS probs.
    carry_returns:
        Array of carry-forward returns computed by ``compute_carry_returns``.
        When provided the sized return is economically correct.
        When None falls back to the legacy zero-return assumption.

    Returns
    -------
    predictions DataFrame with two new columns:
        position_size  – the multiplier applied to each trade
        sized_return   – economically correct portfolio return
    """
    probs = predictions["m2_prob"].values.copy()

    if normalize:
        sizes = np.empty(len(probs))
        for t in range(len(probs)):
            if t == 0:
                # No prior OOS history - size = 1.0 for the first prediction
                sizes[t] = 1.0
            else:
                expanding_mean = probs[:t].mean()
                if expanding_mean == 0:
                    raise ValueError(f"Expanding mean is zero at step {t}.")
                sizes[t] = probs[t] / expanding_mean
    else:
        sizes = probs

    m1_rets = predictions["meta_target_return"].values

    if carry_returns is not None:
        sized_rets = sizes * m1_rets + (1 - sizes) * carry_returns
    else:
        sized_rets = sizes * m1_rets

    result = predictions.copy()
    result["position_size"] = sizes
    result["sized_return"]  = sized_rets
    return result


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_predictions(predictions: pd.DataFrame, path: Path) -> None:
    """Write walk-forward predictions to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)
