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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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

def apply_position_sizing(
    predictions: pd.DataFrame,
    normalize: bool = True,
) -> pd.DataFrame:
    """Replace binary gate with probability-scaled position sizes.

    Instead of approve/reject, M2's probability scales how much of each
    trade is taken.  Two modes:

    * normalize=True  (default): scale so average position = 1.0, keeping
      the same total market exposure as M1 baseline.  Allows fair Sharpe
      and IR comparison against M1.
    * normalize=False: raw probability as size (0 to 1), so the portfolio
      is always partially de-risked vs M1.

    Parameters
    ----------
    predictions:
        Output of ``run_walk_forward`` (must contain m2_prob and
        meta_target_return).
    normalize:
        Whether to rescale sizes so their mean equals 1.0.

    Returns
    -------
    predictions DataFrame with two new columns:
        position_size  – the multiplier applied to each trade
        sized_return   – position_size * meta_target_return
    """
    probs = predictions["m2_prob"].values.copy()

    if normalize:
        mean_prob = probs.mean()
        if mean_prob == 0:
            raise ValueError("All probabilities are zero — cannot normalise.")
        sizes = probs / mean_prob
    else:
        sizes = probs

    result = predictions.copy()
    result["position_size"] = sizes
    result["sized_return"]  = sizes * result["meta_target_return"].values
    return result


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_predictions(predictions: pd.DataFrame, path: Path) -> None:
    """Write walk-forward predictions to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)
