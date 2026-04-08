"""Primary-model signal construction logic."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from metalabel.data import _REQUIRED_ASSETS


def expanding_zscore(
    series: pd.Series,
    min_periods: int = 12,
    ddof: int = 1,
) -> pd.Series:
    """Compute expanding z-score using only prior observations."""
    if min_periods < 1:
        raise ValueError("min_periods must be >= 1.")

    x = pd.to_numeric(series, errors="coerce")
    history = x.shift(1)
    hist_mean = history.expanding(min_periods=min_periods).mean()
    hist_std = history.expanding(min_periods=min_periods).std(ddof=ddof)
    z = (x - hist_mean) / hist_std
    return z.replace([np.inf, -np.inf], np.nan)


def build_variant1_indicators(
    universe: dict[str, pd.DataFrame],
    trend_window: int = 6,
    relative_window: int = 3,
) -> pd.DataFrame:
    """Build Variant 1 raw indicators from universe data."""
    if trend_window < 1 or relative_window < 1:
        raise ValueError("trend_window and relative_window must be >= 1.")

    missing = [asset for asset in _REQUIRED_ASSETS if asset not in universe]
    if missing:
        raise ValueError(f"Universe is missing required asset(s): {missing}")

    spx_price = pd.to_numeric(universe["spx"]["Price"], errors="coerce")
    bcom_price = pd.to_numeric(universe["bcom"]["Price"], errors="coerce")

    spx_ret = pd.to_numeric(universe["spx"]["Return"], errors="coerce")
    bcom_ret = pd.to_numeric(universe["bcom"]["Return"], errors="coerce")
    corp_ret = pd.to_numeric(universe["corp_bonds"]["Return"], errors="coerce")
    ust_ret = pd.to_numeric(universe["treasury_10y"]["Return"], errors="coerce")

    spx_trend = spx_price / spx_price.rolling(trend_window, min_periods=trend_window).mean() - 1.0
    bcom_trend = (
        bcom_price / bcom_price.rolling(trend_window, min_periods=trend_window).mean() - 1.0
    )
    credit_vs_rates = (corp_ret - ust_ret).rolling(
        relative_window, min_periods=relative_window
    ).mean()
    risk_breadth = (pd.concat([spx_ret, bcom_ret, corp_ret], axis=1).mean(axis=1) - ust_ret).rolling(
        relative_window, min_periods=relative_window
    ).mean()

    bcom_accel = bcom_ret.rolling(3, min_periods=3).mean() - bcom_ret.rolling(12, min_periods=12).mean()

    ust_price = pd.to_numeric(universe["treasury_10y"]["Price"], errors="coerce")
    yield_mom = -(ust_price.diff(3))

    indicators = pd.DataFrame(
        {
            "spx_trend": spx_trend,
            "bcom_trend": bcom_trend,
            "credit_vs_rates": credit_vs_rates,
            "risk_breadth": risk_breadth,
            "bcom_accel": bcom_accel,
            "yield_mom": yield_mom,
        }
    ).sort_index()
    return indicators


def _weight_series(columns: pd.Index, weights: Mapping[str, float] | None) -> pd.Series:
    if len(columns) == 0:
        raise ValueError("No indicator columns were provided.")

    if weights is None:
        return pd.Series(1.0, index=columns, dtype=float)

    provided = pd.Series(weights, dtype=float)
    missing = [col for col in columns if col not in provided.index]
    extra = [col for col in provided.index if col not in columns]
    if missing:
        raise ValueError(f"Missing indicator weight(s): {missing}")
    if extra:
        raise ValueError(f"Unknown indicator weight(s): {extra}")

    if np.isclose(provided.abs().sum(), 0.0):
        raise ValueError("At least one indicator weight must be non-zero.")
    return provided.reindex(columns)


def composite_score(
    zscores: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
) -> pd.Series:
    """Compute a weighted composite score from indicator z-scores."""
    w = _weight_series(zscores.columns, weights)
    weighted_sum = zscores.mul(w, axis=1).sum(axis=1, min_count=1)
    active_weight = zscores.notna().mul(w.abs(), axis=1).sum(axis=1)
    score = weighted_sum / active_weight.replace(0.0, np.nan)
    return score.replace([np.inf, -np.inf], np.nan)


def score_to_signal(
    score: pd.Series,
    buy_threshold: float = 0.31,
    sell_threshold: float = -0.31,
) -> pd.Series:
    """Convert a continuous score to discrete BUY/HOLD/SELL labels."""
    if buy_threshold <= sell_threshold:
        raise ValueError("buy_threshold must be greater than sell_threshold.")

    out = pd.Series("HOLD", index=score.index, dtype=object)
    clean_score = pd.to_numeric(score, errors="coerce")
    out.loc[clean_score > buy_threshold] = "BUY"
    out.loc[clean_score < sell_threshold] = "SELL"
    out.loc[clean_score.isna()] = np.nan
    return out


def _equal_weight_series(columns: pd.Index) -> pd.Series:
    """Return equal weights for the provided columns."""
    return pd.Series(1.0 / len(columns), index=columns, dtype=float)


def _positive_spearman_ic_mask(ic_window: pd.DataFrame, columns: pd.Index) -> pd.Series:
    """Build a binary IC mask from the provided trailing window."""
    corrs = ic_window.corr(method="spearman")["target_ret"].drop("target_ret")
    return (corrs.reindex(columns) > 0.0).astype(float)


def _dynamic_composite_score(
    zscores: pd.DataFrame,
    target_ret: pd.Series,
    spx_returns: pd.Series | None = None,
) -> pd.Series:
    """Compute expanding IC/correlation-adjusted factor score."""
    active_cols = [c for c in zscores.columns if zscores[c].notna().any()]
    if not active_cols:
        return pd.Series(np.nan, index=zscores.index, name="composite_score", dtype=float)

    active_index = pd.Index(active_cols)
    active_zscores = zscores[active_cols]

    ic_data = active_zscores.shift(1).copy()
    ic_data["target_ret"] = pd.to_numeric(target_ret, errors="coerce").reindex(zscores.index)

    vix_proxy = pd.DataFrame(index=zscores.index, columns=["vol_30d", "vol_90d", "shock"], dtype=float)
    if spx_returns is not None:
        spx_ret = pd.to_numeric(spx_returns, errors="coerce").reindex(zscores.index)
        vix_proxy["vol_30d"] = spx_ret.rolling(30, min_periods=30).std()
        vix_proxy["vol_90d"] = spx_ret.rolling(90, min_periods=90).std()
        vix_proxy["shock"] = spx_ret.abs()
        vix_proxy = vix_proxy.apply(expanding_zscore, axis=0)

    weights = pd.DataFrame(0.0, index=zscores.index, columns=zscores.columns, dtype=float)

    for i in range(len(zscores)):
        window = active_zscores.iloc[: i + 1].dropna(how="all")

        ic_mask = pd.Series(1.0, index=active_index, dtype=float)
        force_equal_weights = False
        if i >= 36:
            # Including row i is causal: ic_data aligns z_{t-1} with realized return_t,
            # and the primary signal at t is used for deployment at t+1.
            full_ic_window = ic_data.iloc[max(0, i - 35) : i + 1].dropna()
            if len(full_ic_window) < 12:
                force_equal_weights = True
            else:
                ic_window = full_ic_window
                current_proxy = vix_proxy.iloc[i]
                if current_proxy.isna().any():
                    proxy_window = vix_proxy.iloc[0:0]
                else:
                    proxy_window = vix_proxy.iloc[: i + 1].dropna()

                if len(proxy_window) < 24:
                    ic_window = full_ic_window
                else:
                    try:
                        hmm = GaussianHMM(
                            n_components=2,
                            covariance_type="diag",
                            n_iter=300,
                            random_state=42,
                        )
                        regime_input = proxy_window.to_numpy()
                        hmm.fit(regime_input)
                        predicted_states = hmm.predict(regime_input)
                        high_vol_state = int(np.argmax(hmm.means_[:, 0]))
                        current_state = predicted_states[-1]
                        current_is_high_vol = current_state == high_vol_state
                        regime_state = high_vol_state if current_is_high_vol else 1 - high_vol_state
                        regime_index = proxy_window.index[predicted_states == regime_state]
                        regime_ic_window = full_ic_window.loc[
                            full_ic_window.index.intersection(regime_index)
                        ]
                    except Exception:
                        ic_window = full_ic_window
                    else:
                        if len(regime_ic_window) < 8:
                            ic_window = full_ic_window
                        else:
                            ic_window = regime_ic_window

                ic_mask = _positive_spearman_ic_mask(ic_window, active_index)

        if force_equal_weights or len(window) < 12:
            base_w = _equal_weight_series(active_index)
        else:
            corr_mat = window.corr()
            if corr_mat.isna().any().any():
                base_w = _equal_weight_series(active_index)
            else:
                sum_corr = corr_mat.sum(axis=0).clip(lower=1.0)
                inv_corr = 1.0 / sum_corr
                base_w = inv_corr / inv_corr.sum()

        final_w = base_w * ic_mask
        if final_w.sum() > 0:
            final_w = final_w / final_w.sum()
        else:
            final_w = _equal_weight_series(active_index)

        weights.loc[zscores.index[i], active_index] = final_w.to_numpy()

    return (zscores * weights).sum(axis=1).rename("composite_score")


def build_primary_signal_variant1(
    universe: dict[str, pd.DataFrame],
    trend_window: int = 6,
    relative_window: int = 3,
    zscore_min_periods: int = 12,
    indicator_weights: Mapping[str, float] | None = None,
    buy_threshold: float = 0.31,
    sell_threshold: float = -0.31,
) -> pd.DataFrame:
    """Build Variant 1 indicators, z-scores, composite score, and signal."""
    indicators = build_variant1_indicators(
        universe=universe,
        trend_window=trend_window,
        relative_window=relative_window,
    )

    zscores = indicators.apply(
        expanding_zscore,
        axis=0,
        min_periods=zscore_min_periods,
    )
    zscores.columns = [f"{col}_z" for col in zscores.columns]

    zscores_for_composite = zscores.copy()
    zscores_for_composite["bcom_trend_z"] = np.nan

    if indicator_weights is None:
        spx_price = pd.to_numeric(universe["spx"]["Price"], errors="coerce")
        spx_returns = pd.to_numeric(universe["spx"]["Return"], errors="coerce")
        target_ret = spx_price.pct_change()
        score = _dynamic_composite_score(
            zscores=zscores_for_composite,
            target_ret=target_ret,
            spx_returns=spx_returns,
        )
    else:
        score = composite_score(zscores=zscores_for_composite, weights=indicator_weights).rename("composite_score")

    signal = score_to_signal(
        score=score,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    ).rename("signal")

    return pd.concat([indicators, zscores, score, signal], axis=1)
