#!/usr/bin/env python3
"""Primary side model built from Joubert (2022) and QuantAtoZ principles.

The model is intentionally primary-only:
- predicts side (0/1 for long-only, or -1/0/1 for long-short),
- emits a continuous raw factor score and a probability,
- does not perform advanced probability-to-size optimization.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


ROOT_DIR = Path(__file__).resolve().parent
RAW_DIR = ROOT_DIR / "data" / "raw"
REPORTS_DIR = ROOT_DIR / "reports"

RAW_FILES = {
    "bcom": RAW_DIR / "bcom.xlsx",
    "spx": RAW_DIR / "spx.xlsx",
    "treasury_10y": RAW_DIR / "treasury_10y.xlsx",
    "corp_bonds": RAW_DIR / "corp_bonds.xlsx",
}

ASSET_ALIASES = {
    "bcom": "BCOM",
    "spx": "SPX",
    "treasury_10y": "Treasury10Y",
    "corp_bonds": "IG_Corp",
}


@dataclass(frozen=True)
class FactorSpec:
    name: str
    theme: str
    orientation: int
    dominant: bool = False


FACTOR_SPECS: List[FactorSpec] = [
    FactorSpec("spx_mom_3m", "momentum", 1, True),
    FactorSpec("spx_mom_12m", "momentum", 1, True),
    FactorSpec("spx_vs_bcom_6m", "cross_asset", 1, False),
    FactorSpec("rates_change_6m", "rates", -1, False),
    FactorSpec("credit_carry_3m", "credit", 1, False),
    FactorSpec("vol_regime_3_12", "volatility", -1, True),
    FactorSpec("drawdown_6m", "risk", 1, False),
    FactorSpec("credit_vs_rates_3m", "cross_asset", 1, False),
]


@dataclass
class ModelConfig:
    mode: str = "long_only"
    train_ratio: float = 0.60
    window_months: int = 120
    min_train_obs: int = 72
    min_score_coverage: float = 0.75
    transaction_cost_bps: float = 5.0
    min_history_for_z: int = 24
    winsor_clip: float = 3.0
    winsor_iters: int = 5
    max_factors: int = 10
    min_factors: int = 4
    max_factors_per_theme: int = 2
    min_factor_weight: float = 0.03
    max_theme_weight: float = 0.50
    target_recall: float = 0.65
    min_precision: float = 0.60
    threshold_lookback_months: int = 72
    score_gate_quantiles: Tuple[float, ...] = (0.10, 0.25, 0.40, 0.55)
    hysteresis_gap: float = 0.08
    min_exit_threshold: float = 0.48
    max_active_rate: float = 0.85
    target_max_drawdown: float = -0.30
    max_strategy_spx_corr: float = 0.90
    max_factor_pair_corr: float = 0.75
    min_ic_lag1: float = 0.01
    min_ic_pos_rate: float = 0.55
    min_pure_tstat: float = 0.75
    label_vol_lookback: int = 36
    label_noise_quantile: float = 0.35
    label_noise_scale: float = 0.40
    label_min_hurdle_bps: float = 2.0
    ambiguous_label_weight: float = 0.35
    min_prob_train_rows: int = 40
    prob_class_weight_up: float = 1.00
    prob_l2_c: float = 0.80
    max_missing_rate: float = 0.10
    max_return_gap_p95: float = 0.15
    min_timestamp_alignment: float = 0.90
    min_test_periods: int = 24
    min_test_recall: float = 0.55
    min_test_sharpe: float = 0.0
    max_f1_drop: float = 0.40


def _safe_std(series: pd.Series) -> float:
    value = float(series.std(ddof=1))
    return value if np.isfinite(value) and value > 1e-12 else np.nan


def _annualized_return(returns: pd.Series) -> float:
    rets = returns.dropna().astype(float)
    if rets.empty:
        return np.nan
    equity = (1.0 + rets).cumprod()
    return float(equity.iloc[-1] ** (12.0 / len(rets)) - 1.0)


def _annualized_vol(returns: pd.Series) -> float:
    rets = returns.dropna().astype(float)
    if rets.empty:
        return np.nan
    std = _safe_std(rets)
    if np.isnan(std):
        return np.nan
    return float(std * np.sqrt(12.0))


def _max_drawdown(returns: pd.Series) -> float:
    rets = returns.dropna().astype(float)
    if rets.empty:
        return np.nan
    equity = (1.0 + rets).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def _tstat(series: pd.Series) -> float:
    s = series.dropna().astype(float)
    n = len(s)
    if n < 5:
        return np.nan
    std = _safe_std(s)
    if np.isnan(std):
        return np.nan
    return float(s.mean() / (std / np.sqrt(n)))


def find_header_row(raw_df: pd.DataFrame) -> int:
    for i, row in raw_df.iterrows():
        values = [str(v).strip().upper() for v in row.tolist() if pd.notna(v)]
        if "DATE" in values and ("PX_LAST" in values or "CLOSE" in values):
            return int(i)
    raise ValueError("Could not find header row with Date and PX_LAST/Close.")


def load_single_asset(path: Path, alias: str) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None, dtype=object)
    header_row = find_header_row(raw)

    header = raw.iloc[header_row].tolist()
    data = raw.iloc[header_row + 1 :].copy()
    data.columns = header
    data = data.loc[:, [col for col in data.columns if pd.notna(col)]]
    data.columns = [str(c).strip() for c in data.columns]

    date_col = next((c for c in data.columns if c.lower() == "date"), None)
    price_col = next((c for c in data.columns if c.upper() == "PX_LAST" or "CLOSE" in c.upper()), None)
    ret_col = next((c for c in data.columns if c.upper() == "CHG_PCT_1D" or "RETURN" in c.upper()), None)

    if date_col is None or price_col is None:
        raise ValueError(f"Missing Date/PX_LAST columns in {path}")

    clean = data[[date_col, price_col]].copy()
    clean.columns = ["Date", "Price"]
    clean["Date"] = pd.to_datetime(clean["Date"], errors="coerce")
    clean["Price"] = pd.to_numeric(clean["Price"], errors="coerce")

    if ret_col is not None:
        returns = (
            data[ret_col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        clean["Return"] = pd.to_numeric(returns, errors="coerce") / 100.0
    else:
        clean["Return"] = np.nan

    clean = clean.dropna(subset=["Date"]).drop_duplicates(subset=["Date"], keep="last")
    clean = clean.sort_values("Date").reset_index(drop=True)

    if clean["Return"].isna().all():
        clean["Return"] = clean["Price"].pct_change()

    clean = clean.rename(
        columns={
            "Price": f"{alias}_Price",
            "Return": f"{alias}_Return",
        }
    )
    return clean


def load_merged_data() -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for key, file_path in RAW_FILES.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required file: {file_path}")
        alias = ASSET_ALIASES[key]
        parts.append(load_single_asset(file_path, alias))

    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on="Date", how="outer")

    merged = merged.sort_values("Date").reset_index(drop=True)

    price_cols = [c for c in merged.columns if c.endswith("_Price")]
    merged[price_cols] = merged[price_cols].ffill()

    # Keep returns explicit: never forward-fill returns. Missing returns are
    # backfilled from price relatives only at the same timestamp.
    for alias in ASSET_ALIASES.values():
        price_col = f"{alias}_Price"
        ret_col = f"{alias}_Return"
        implied = merged[price_col].pct_change()
        merged[ret_col] = merged[ret_col].where(merged[ret_col].notna(), implied)

    return merged.dropna(subset=price_cols)


def build_data_qc_report(merged: pd.DataFrame, cfg: ModelConfig) -> Tuple[pd.DataFrame, bool]:
    records: List[Dict[str, object]] = []

    required_cols: List[str] = []
    for alias in ASSET_ALIASES.values():
        required_cols.extend([f"{alias}_Price", f"{alias}_Return"])

    universe_consistent = all(col in merged.columns for col in required_cols)
    date_monotonic = bool(merged["Date"].is_monotonic_increasing and merged["Date"].is_unique)

    price_cols = [f"{alias}_Price" for alias in ASSET_ALIASES.values()]
    timestamp_alignment_rate = float(merged[price_cols].notna().all(axis=1).mean()) if price_cols else 0.0
    timestamp_alignment_pass = timestamp_alignment_rate >= cfg.min_timestamp_alignment

    history_obs_list: List[int] = []
    asset_pass_list: List[bool] = []
    coverage_target = cfg.window_months + cfg.min_train_obs

    for alias in ASSET_ALIASES.values():
        price_col = f"{alias}_Price"
        ret_col = f"{alias}_Return"

        px = merged[price_col]
        rt = merged[ret_col]
        implied = px.pct_change()
        gap = (rt - implied).abs()

        overlap_obs = int(gap.notna().sum())
        p95_gap = float(gap.quantile(0.95)) if overlap_obs > 0 else np.nan

        missing_price_rate = float(px.isna().mean())
        missing_return_rate = float(rt.isna().mean())
        history_obs = int(px.notna().sum())
        history_obs_list.append(history_obs)

        start_date = merged.loc[px.notna(), "Date"].min() if px.notna().any() else pd.NaT
        end_date = merged.loc[px.notna(), "Date"].max() if px.notna().any() else pd.NaT

        missing_pass = (missing_price_rate <= cfg.max_missing_rate) and (missing_return_rate <= cfg.max_missing_rate)
        corporate_action_proxy_pass = bool(np.isfinite(p95_gap) and (p95_gap <= cfg.max_return_gap_p95))
        history_pass = history_obs >= coverage_target
        asset_pass = missing_pass and corporate_action_proxy_pass and history_pass
        asset_pass_list.append(asset_pass)

        records.append(
            {
                "scope": "asset",
                "asset": alias,
                "start_date": start_date,
                "end_date": end_date,
                "history_obs": history_obs,
                "coverage_target_obs": coverage_target,
                "missing_price_rate": missing_price_rate,
                "missing_return_rate": missing_return_rate,
                "return_gap_p95_abs": p95_gap,
                "missing_pass": missing_pass,
                "corporate_action_proxy_pass": corporate_action_proxy_pass,
                "history_pass": history_pass,
                "asset_pass": asset_pass,
            }
        )

    min_hist = min(history_obs_list) if history_obs_list else 0
    max_hist = max(history_obs_list) if history_obs_list else 0
    survivorship_proxy_ratio = float(min_hist / max_hist) if max_hist > 0 else 0.0
    survivorship_proxy_pass = survivorship_proxy_ratio >= 0.90

    overall_pass = (
        universe_consistent
        and date_monotonic
        and timestamp_alignment_pass
        and survivorship_proxy_pass
        and all(asset_pass_list)
    )

    records.append(
        {
            "scope": "global",
            "asset": "__GLOBAL__",
            "universe_consistent": universe_consistent,
            "date_monotonic": date_monotonic,
            "timestamp_alignment_rate": timestamp_alignment_rate,
            "timestamp_alignment_pass": timestamp_alignment_pass,
            "survivorship_proxy_ratio": survivorship_proxy_ratio,
            "survivorship_proxy_pass": survivorship_proxy_pass,
            "overall_data_qc_pass": overall_pass,
        }
    )

    qc_df = pd.DataFrame(records)
    return qc_df, bool(overall_pass)


def build_raw_factors(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    out["spx_mom_3m"] = df["SPX_Price"].pct_change(3)
    out["spx_mom_12m"] = df["SPX_Price"].pct_change(12)

    out["spx_vs_bcom_6m"] = df["SPX_Price"].pct_change(6) - df["BCOM_Price"].pct_change(6)
    out["rates_change_6m"] = df["Treasury10Y_Price"].diff(6)
    out["credit_carry_3m"] = df["IG_Corp_Return"].rolling(3).mean()

    out["vol_regime_3_12"] = (
        df["SPX_Return"].rolling(3).std(ddof=1) / df["SPX_Return"].rolling(12).std(ddof=1)
    )

    out["drawdown_6m"] = df["SPX_Price"] / df["SPX_Price"].rolling(6).max() - 1.0
    out["credit_vs_rates_3m"] = (
        df["IG_Corp_Return"].rolling(3).mean() - df["Treasury10Y_Return"].rolling(3).mean()
    )

    out.index = df["Date"]
    return out


def causal_iterative_standardize(
    series: pd.Series,
    min_history: int,
    clip: float,
    max_iter: int,
) -> pd.Series:
    values = series.astype(float).to_numpy()
    out = np.full(len(values), np.nan)

    for i in range(len(values)):
        hist = values[: i + 1]
        hist = hist[np.isfinite(hist)]
        if hist.size < min_history:
            continue

        work = hist.copy()
        for _ in range(max_iter):
            mu = float(np.mean(work))
            sd = float(np.std(work, ddof=1))
            if not np.isfinite(sd) or sd < 1e-12:
                break
            z = (work - mu) / sd
            z = np.clip(z, -clip, clip)
            if np.allclose(work, z, rtol=1e-6, atol=1e-8):
                work = z
                break
            work = z

        out[i] = work[-1]

    return pd.Series(out, index=series.index, name=series.name)


def build_oriented_z_factors(raw_factors: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    z = pd.DataFrame(index=raw_factors.index)
    orientation_map = {spec.name: spec.orientation for spec in FACTOR_SPECS}

    for col in raw_factors.columns:
        z_col = causal_iterative_standardize(
            raw_factors[col],
            min_history=cfg.min_history_for_z,
            clip=cfg.winsor_clip,
            max_iter=cfg.winsor_iters,
        )
        z[col] = z_col * orientation_map[col]

    return z


def enforce_missing_data_rules(z_factors: pd.DataFrame, cfg: ModelConfig) -> Tuple[pd.DataFrame, pd.Series]:
    dominant_cols = [spec.name for spec in FACTOR_SPECS if spec.dominant]

    if not dominant_cols:
        coverage = z_factors.notna().mean(axis=1)
        valid = coverage >= cfg.min_score_coverage
    else:
        dominant_ok = z_factors[dominant_cols].notna().all(axis=1)
        coverage = z_factors.notna().mean(axis=1)
        valid = dominant_ok & (coverage >= cfg.min_score_coverage)

    filtered = z_factors.where(valid)
    return filtered, coverage


def rolling_rank_ic(factor: pd.Series, target: pd.Series, window: int = 36) -> pd.Series:
    joined = pd.concat([factor, target], axis=1).dropna()
    if len(joined) < window:
        return pd.Series(dtype=float)

    out = []
    idx = []
    for i in range(window, len(joined) + 1):
        f_slice = joined.iloc[i - window : i, 0]
        t_slice = joined.iloc[i - window : i, 1]
        value = f_slice.corr(t_slice, method="spearman")
        out.append(value)
        idx.append(joined.index[i - 1])

    return pd.Series(out, index=idx)


def ols_factor_tstat(y: pd.Series, factor: pd.Series, controls: pd.DataFrame) -> Tuple[float, float]:
    data = pd.concat([y.rename("y"), factor.rename("factor"), controls], axis=1).dropna()
    if len(data) < 30:
        return np.nan, np.nan

    y_vec = data["y"].to_numpy(dtype=float)
    x_cols = ["factor"] + [c for c in controls.columns]
    x_mat = data[x_cols].to_numpy(dtype=float)
    x_mat = np.column_stack([np.ones(len(data)), x_mat])

    xtx = x_mat.T @ x_mat
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (x_mat.T @ y_vec)

    residuals = y_vec - x_mat @ beta
    dof = len(y_vec) - x_mat.shape[1]
    if dof <= 1:
        return np.nan, np.nan

    sigma2 = float((residuals @ residuals) / dof)
    cov = sigma2 * xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 1e-14, None))

    factor_beta = float(beta[1])
    factor_t = float(beta[1] / se[1]) if se[1] > 0 else np.nan
    return factor_beta, factor_t


def fractile_metrics(factor: pd.Series, target: pd.Series, buckets: int = 5) -> Tuple[float, float, float]:
    data = pd.concat([factor.rename("factor"), target.rename("target")], axis=1).dropna()
    if len(data) < buckets * 10:
        return np.nan, np.nan, np.nan

    ranked = data["factor"].rank(method="first")
    q = pd.qcut(ranked, q=buckets, labels=False, duplicates="drop")
    if q.nunique() < 3:
        return np.nan, np.nan, np.nan

    grouped = data.groupby(q)["target"].mean()
    spread = float(grouped.iloc[-1] - grouped.iloc[0])
    monotonicity = float(pd.Series(range(len(grouped))).corr(grouped.reset_index(drop=True), method="spearman"))
    turnover = float(q.diff().ne(0).mean())
    return spread, monotonicity, turnover


def factor_diagnostics(train_df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
    records: List[Dict[str, float | str]] = []

    controls = train_df[["Risk_BCOM", "Risk_Rates", "Risk_Credit", "Risk_Vol"]]
    target = train_df["Forward_Return"]

    for spec in FACTOR_SPECS:
        if spec.name not in factor_cols:
            continue

        factor = train_df[spec.name]
        joined = pd.concat([factor, target], axis=1).dropna()

        if len(joined) < 30:
            records.append(
                {
                    "factor": spec.name,
                    "theme": spec.theme,
                    "ic_lag1": np.nan,
                    "rolling_ic_mean": np.nan,
                    "rolling_ic_pos_rate": np.nan,
                    "ic_tstat": np.nan,
                    "ic_decay_1": np.nan,
                    "ic_decay_3": np.nan,
                    "ic_decay_6": np.nan,
                    "ic_decay_12": np.nan,
                    "fractile_spread": np.nan,
                    "fractile_monotonicity": np.nan,
                    "fractile_turnover": np.nan,
                    "pure_beta": np.nan,
                    "pure_tstat": np.nan,
                }
            )
            continue

        ic_lag1 = float(joined.iloc[:, 0].corr(joined.iloc[:, 1], method="spearman"))

        roll_ic = rolling_rank_ic(factor, target, window=36)
        rolling_ic_mean = float(roll_ic.mean()) if not roll_ic.empty else np.nan
        rolling_ic_pos_rate = float((roll_ic > 0).mean()) if not roll_ic.empty else np.nan
        ic_tstat = _tstat(roll_ic)

        decay = {}
        for lag in [1, 3, 6, 12]:
            shifted_target = train_df["SPX_Return"].shift(-lag)
            aligned = pd.concat([factor, shifted_target], axis=1).dropna()
            decay[lag] = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")) if len(aligned) >= 20 else np.nan

        spread, monotonicity, turnover = fractile_metrics(factor, target, buckets=5)
        pure_beta, pure_t = ols_factor_tstat(target, factor, controls)

        records.append(
            {
                "factor": spec.name,
                "theme": spec.theme,
                "ic_lag1": ic_lag1,
                "rolling_ic_mean": rolling_ic_mean,
                "rolling_ic_pos_rate": rolling_ic_pos_rate,
                "ic_tstat": ic_tstat,
                "ic_decay_1": decay[1],
                "ic_decay_3": decay[3],
                "ic_decay_6": decay[6],
                "ic_decay_12": decay[12],
                "fractile_spread": spread,
                "fractile_monotonicity": monotonicity,
                "fractile_turnover": turnover,
                "pure_beta": pure_beta,
                "pure_tstat": pure_t,
            }
        )

    diag = pd.DataFrame(records)

    for col in ["ic_lag1", "ic_tstat", "fractile_spread", "fractile_monotonicity", "pure_tstat", "rolling_ic_pos_rate"]:
        s = diag[col].replace([np.inf, -np.inf], np.nan)
        sd = _safe_std(s.dropna())
        if np.isnan(sd):
            diag[f"z_{col}"] = 0.0
        else:
            diag[f"z_{col}"] = ((s - s.mean()) / sd).fillna(0.0)

    diag["quality_score"] = (
        0.25 * diag["z_ic_lag1"]
        + 0.20 * diag["z_ic_tstat"]
        + 0.15 * diag["z_fractile_spread"]
        + 0.15 * diag["z_fractile_monotonicity"]
        + 0.15 * diag["z_pure_tstat"]
        + 0.10 * diag["z_rolling_ic_pos_rate"]
    )

    diag["pass_significance"] = (diag["ic_tstat"] >= 1.0) | (diag["pure_tstat"] >= 1.0)
    diag["pass_stability"] = (diag["rolling_ic_mean"] > 0) & (diag["rolling_ic_pos_rate"] >= 0.52)
    diag["pass_decay"] = diag["ic_decay_1"] > 0
    diag["pass_fractile"] = (diag["fractile_spread"] > 0) & (diag["fractile_monotonicity"] > 0)
    diag["pass_all"] = (
        diag["pass_significance"]
        & diag["pass_stability"]
        & diag["pass_decay"]
        & diag["pass_fractile"]
    )

    return diag.sort_values("quality_score", ascending=False).reset_index(drop=True)


def select_factors(diag: pd.DataFrame, train_factors: pd.DataFrame, cfg: ModelConfig) -> List[str]:
    selected: List[str] = []
    per_theme: Dict[str, int] = {}
    corr = train_factors.corr().abs().replace([np.inf, -np.inf], np.nan)

    preferred = diag[
        (diag["pass_all"])
        & (diag["ic_lag1"] >= cfg.min_ic_lag1)
        & (diag["rolling_ic_pos_rate"] >= cfg.min_ic_pos_rate)
        & ((diag["pure_tstat"] >= cfg.min_pure_tstat) | (diag["ic_tstat"] >= 1.0))
    ].copy()
    if preferred.empty:
        preferred = diag[
            (diag["ic_lag1"] >= 0)
            & (diag["rolling_ic_pos_rate"] >= 0.50)
        ].copy()
    if preferred.empty:
        preferred = diag.copy()

    for _, row in preferred.sort_values("quality_score", ascending=False).iterrows():
        factor = str(row["factor"])
        theme = str(row["theme"])
        if pd.isna(row["ic_lag1"]) or row["ic_lag1"] <= 0:
            continue
        if per_theme.get(theme, 0) >= cfg.max_factors_per_theme:
            continue
        if selected and factor in corr.index:
            c = corr.loc[factor, selected].max(skipna=True)
            if np.isfinite(c) and c > cfg.max_factor_pair_corr:
                continue
        selected.append(factor)
        per_theme[theme] = per_theme.get(theme, 0) + 1
        if len(selected) >= cfg.max_factors:
            break

    if len(selected) < cfg.min_factors:
        fallback = diag.sort_values("quality_score", ascending=False)
        for _, row in fallback.iterrows():
            factor = str(row["factor"])
            if factor in selected:
                continue
            if selected and factor in corr.index:
                c = corr.loc[factor, selected].max(skipna=True)
                if np.isfinite(c) and c > cfg.max_factor_pair_corr:
                    continue
            selected.append(factor)
            if len(selected) >= cfg.min_factors:
                break

    return selected


def apply_theme_cap(weights: pd.Series, cfg: ModelConfig) -> pd.Series:
    weights = weights.copy().clip(lower=0.0)
    theme_map = {spec.name: spec.theme for spec in FACTOR_SPECS}

    if weights.sum() <= 0:
        return weights

    weights = weights / weights.sum()

    for _ in range(8):
        theme_totals = weights.groupby(weights.index.map(lambda x: theme_map.get(x, "other"))).sum()
        over = theme_totals[theme_totals > cfg.max_theme_weight]
        if over.empty:
            break

        for theme_name, total in over.items():
            idx = [col for col in weights.index if theme_map.get(col, "other") == theme_name]
            scale = cfg.max_theme_weight / float(total)
            weights.loc[idx] = weights.loc[idx] * scale

        residual = 1.0 - float(weights.sum())
        if residual <= 1e-12:
            break

        under_themes = theme_totals[theme_totals <= cfg.max_theme_weight].index.tolist()
        under_idx = [col for col in weights.index if theme_map.get(col, "other") in under_themes]

        if not under_idx:
            weights = weights / weights.sum()
            break

        base = weights.loc[under_idx]
        if base.sum() <= 0:
            weights.loc[under_idx] = weights.loc[under_idx] + residual / len(under_idx)
        else:
            weights.loc[under_idx] = weights.loc[under_idx] + residual * (base / base.sum())

        weights = weights.clip(lower=0.0)
        if weights.sum() > 0:
            weights = weights / weights.sum()

    return weights


def correlation_aware_weights(train_factors: pd.DataFrame, diag: pd.DataFrame, cfg: ModelConfig) -> pd.Series:
    if train_factors.shape[1] == 1:
        return pd.Series([1.0], index=train_factors.columns)

    di = diag.set_index("factor")
    ic_vec = di.loc[train_factors.columns, "ic_lag1"].fillna(0.0).clip(lower=0.0)

    corr = train_factors.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    corr = corr + np.eye(len(corr)) * 1e-6

    adjusted = np.linalg.pinv(corr.to_numpy()) @ ic_vec.to_numpy()
    adjusted = np.clip(adjusted, 0.0, None)

    weights = pd.Series(adjusted, index=train_factors.columns)

    if weights.sum() <= 0:
        weights = ic_vec.copy()

    if weights.sum() <= 0:
        weights = pd.Series(1.0, index=train_factors.columns)

    weights = weights / weights.sum()

    weights = weights.where(weights >= cfg.min_factor_weight, 0.0)
    if weights.sum() <= 0:
        weights = pd.Series(1.0 / len(weights), index=weights.index)
    else:
        weights = weights / weights.sum()

    return apply_theme_cap(weights, cfg)


def composite_score(z_factors: pd.DataFrame, weights: pd.Series, min_coverage: float) -> Tuple[pd.Series, pd.Series]:
    aligned_weights = weights.reindex(z_factors.columns).fillna(0.0)
    if aligned_weights.sum() <= 0:
        score = pd.Series(np.nan, index=z_factors.index)
        coverage = pd.Series(0.0, index=z_factors.index)
        return score, coverage

    aligned_weights = aligned_weights / aligned_weights.sum()

    weighted = z_factors.fillna(0.0).mul(aligned_weights, axis=1)
    raw_score = weighted.sum(axis=1)

    weighted_coverage = z_factors.notna().mul(aligned_weights, axis=1).sum(axis=1)
    score = raw_score.where(weighted_coverage >= min_coverage)
    score = score / weighted_coverage.where(weighted_coverage > 0)

    return score, weighted_coverage


def build_primary_target_and_weights(forward_return: pd.Series, cfg: ModelConfig) -> Tuple[pd.Series, pd.Series, pd.Series]:
    abs_ret = forward_return.abs()
    noise = abs_ret.rolling(cfg.label_vol_lookback, min_periods=12).quantile(cfg.label_noise_quantile).shift(1)

    cost_floor = (cfg.transaction_cost_bps + cfg.label_min_hurdle_bps) / 10000.0
    hurdle = (noise * cfg.label_noise_scale).clip(lower=cost_floor).fillna(cost_floor)

    target_up = (forward_return > hurdle).astype(int)
    ambiguous = forward_return.abs() <= hurdle

    weights = pd.Series(1.0, index=forward_return.index)
    weights.loc[ambiguous] = cfg.ambiguous_label_weight
    weights = weights.where(forward_return.notna(), np.nan)

    return target_up, hurdle, weights


def _prepare_probability_features(feature_df: pd.DataFrame, cols: List[str], medians: pd.Series) -> pd.DataFrame:
    x = feature_df.copy()
    for col in cols:
        if col not in x.columns:
            x[col] = np.nan
    x = x[cols]
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(medians)
    return x


def fit_probability_model(
    feature_df: pd.DataFrame,
    target: pd.Series,
    sample_weight: pd.Series,
    cfg: ModelConfig,
) -> Tuple[str, object]:
    data = pd.concat(
        [
            feature_df.replace([np.inf, -np.inf], np.nan),
            target.rename("target"),
            sample_weight.rename("sample_weight"),
        ],
        axis=1,
    ).dropna(subset=["target", "sample_weight"])

    if len(data) < cfg.min_prob_train_rows:
        return "constant", float(data["target"].mean()) if len(data) else 0.5

    feat_cols = [c for c in feature_df.columns]
    x_raw = data[feat_cols].copy()
    med = x_raw.median(numeric_only=True).fillna(0.0)
    x_raw = _prepare_probability_features(x_raw, feat_cols, med)

    mean = x_raw.mean(numeric_only=True)
    std = x_raw.std(ddof=1).replace(0, np.nan).fillna(1.0)
    x_scaled = (x_raw - mean) / std

    y = data["target"].astype(int).to_numpy()
    w = data["sample_weight"].astype(float).to_numpy()

    if np.unique(y).size < 2:
        return "constant", float(y.mean())

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight={0: 1.0, 1: cfg.prob_class_weight_up},
        C=cfg.prob_l2_c,
        random_state=42,
    )
    model.fit(x_scaled.to_numpy(dtype=float), y, sample_weight=w)

    model_obj = {
        "model": model,
        "feature_cols": feat_cols,
        "medians": med,
        "mean": mean,
        "std": std,
    }
    return "logit", model_obj


def predict_probability(model_kind: str, model_obj: object, feature_df: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=feature_df.index)
    if feature_df.empty:
        return out

    if model_kind == "constant":
        out.loc[:] = float(model_obj)
        return out

    obj = model_obj
    cols: List[str] = obj["feature_cols"]
    med = obj["medians"]
    mean = obj["mean"]
    std = obj["std"]
    model = obj["model"]

    x = _prepare_probability_features(feature_df, cols, med)
    x_scaled = (x - mean) / std
    probs = model.predict_proba(x_scaled.to_numpy(dtype=float))[:, 1]
    out.loc[x_scaled.index] = probs
    return out


def tune_threshold(
    probabilities: pd.Series,
    target: pd.Series,
    forward_returns: pd.Series,
    scores: pd.Series,
    cfg: ModelConfig,
) -> Tuple[Dict[str, float | str], pd.DataFrame]:
    data = pd.concat(
        [
            probabilities.rename("prob"),
            target.rename("target"),
            forward_returns.rename("forward_return"),
            scores.rename("score"),
        ],
        axis=1,
    ).dropna()

    if cfg.threshold_lookback_months > 0 and len(data) > cfg.threshold_lookback_months:
        data = data.tail(cfg.threshold_lookback_months).copy()

    if len(data) < 30:
        if cfg.mode == "long_short":
            return {"mode": "long_short", "lower": 0.40, "upper": 0.60, "score_gate": 0.0}, pd.DataFrame()
        return {
            "mode": "long_only",
            "entry_threshold": 0.64,
            "exit_threshold": 0.56,
            "score_gate": 0.0,
        }, pd.DataFrame()

    rows: List[Dict[str, float]] = []

    if cfg.mode == "long_short":
        lowers = np.arange(0.20, 0.46, 0.05)
        uppers = np.arange(0.55, 0.81, 0.05)

        for lo in lowers:
            for hi in uppers:
                if hi <= lo:
                    continue
                side = np.where(data["prob"] >= hi, 1, np.where(data["prob"] <= lo, -1, 0))

                truth = np.where(data["target"] == 1, 1, -1)
                active = side != 0
                correct_active = (side == truth) & active

                precision = float(correct_active.sum() / active.sum()) if active.any() else 0.0
                long_recall = float(((side == 1) & (truth == 1)).sum() / (truth == 1).sum()) if (truth == 1).sum() > 0 else 0.0
                short_recall = float(((side == -1) & (truth == -1)).sum() / (truth == -1).sum()) if (truth == -1).sum() > 0 else 0.0
                recall = 0.5 * (long_recall + short_recall)
                f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

                side_s = pd.Series(side, index=data.index, dtype=float)
                trade = build_trade_frame(side_s, data["forward_return"], cfg.transaction_cost_bps)
                sharpe = annualized_sharpe(trade["net_return"]) or 0.0
                turnover = float(trade["turnover"].mean())
                corr_to_spx = float(trade["net_return"].corr(data["forward_return"])) if len(trade) > 3 else np.nan
                max_dd = _max_drawdown(trade["net_return"])

                corr_pen = max(0.0, abs(corr_to_spx) - cfg.max_strategy_spx_corr) if np.isfinite(corr_to_spx) else 0.0
                dd_pen = max(0.0, cfg.target_max_drawdown - max_dd) if np.isfinite(max_dd) else 0.0

                objective = (
                    0.42 * recall
                    + 0.28 * precision
                    + 0.15 * f1
                    + 0.10 * np.tanh(max(sharpe, -3.0) / 3.0)
                    - 0.07 * turnover
                    - 0.09 * corr_pen
                    - 0.12 * dd_pen
                )
                if precision < cfg.min_precision:
                    objective -= (cfg.min_precision - precision) * 0.8
                if recall < cfg.target_recall:
                    objective -= (cfg.target_recall - recall) * 0.5

                rows.append(
                    {
                        "lower": lo,
                        "upper": hi,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "sharpe": sharpe,
                        "turnover": turnover,
                        "corr_to_spx": corr_to_spx,
                        "max_drawdown": max_dd,
                        "objective": objective,
                    }
                )

        tuning = pd.DataFrame(rows).sort_values("objective", ascending=False)
        best = tuning.iloc[0]
        return {
            "mode": "long_short",
            "lower": float(best["lower"]),
            "upper": float(best["upper"]),
            "score_gate": 0.0,
        }, tuning

    thresholds = np.arange(0.52, 0.86, 0.03)
    gate_quantiles = cfg.score_gate_quantiles

    for tau in thresholds:
        for gq in gate_quantiles:
            score_gate = float(data["score"].abs().quantile(gq)) if gq > 0 else 0.0
            side = ((data["prob"] >= tau) & (data["score"].abs() >= score_gate)).astype(int)

            precision = float(precision_score(data["target"], side, zero_division=0))
            recall = float(recall_score(data["target"], side, zero_division=0))
            f1 = float(f1_score(data["target"], side, zero_division=0))

            trade = build_trade_frame(side.astype(float), data["forward_return"], cfg.transaction_cost_bps)
            sharpe = annualized_sharpe(trade["net_return"]) or 0.0
            turnover = float(trade["turnover"].mean())
            corr_to_spx = float(trade["net_return"].corr(data["forward_return"])) if len(trade) > 3 else np.nan
            max_dd = _max_drawdown(trade["net_return"])

            corr_pen = max(0.0, abs(corr_to_spx) - cfg.max_strategy_spx_corr) if np.isfinite(corr_to_spx) else 0.0
            dd_pen = max(0.0, cfg.target_max_drawdown - max_dd) if np.isfinite(max_dd) else 0.0
            active_rate = float(side.mean())

            objective = (
                0.42 * recall
                + 0.28 * precision
                + 0.15 * f1
                + 0.10 * np.tanh(max(sharpe, -3.0) / 3.0)
                - 0.07 * turnover
                - 0.09 * corr_pen
                - 0.12 * dd_pen
            )
            if precision < cfg.min_precision:
                objective -= (cfg.min_precision - precision) * 0.8
            if recall < cfg.target_recall:
                objective -= (cfg.target_recall - recall) * 0.5
            if active_rate > cfg.max_active_rate:
                objective -= (active_rate - cfg.max_active_rate) * 0.5

            rows.append(
                {
                    "threshold": tau,
                    "score_gate_q": gq,
                    "score_gate": score_gate,
                    "active_rate": active_rate,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "sharpe": sharpe,
                    "turnover": turnover,
                    "corr_to_spx": corr_to_spx,
                    "max_drawdown": max_dd,
                    "objective": objective,
                }
            )

    tuning = pd.DataFrame(rows).sort_values("objective", ascending=False)
    best = tuning.iloc[0]
    entry = float(best["threshold"])
    exit_tau = max(cfg.min_exit_threshold, entry - cfg.hysteresis_gap)
    return {
        "mode": "long_only",
        "entry_threshold": entry,
        "exit_threshold": float(exit_tau),
        "score_gate": float(best["score_gate"]),
    }, tuning


def side_from_probability(
    probability: float,
    score: float,
    threshold: Dict[str, float | str],
    mode: str,
    prev_side: float,
) -> float:
    if np.isnan(probability):
        return np.nan

    if mode == "long_short":
        low = float(threshold.get("lower", 0.40))
        high = float(threshold.get("upper", 0.60))
        if probability >= high:
            return 1.0
        if probability <= low:
            return -1.0
        return 0.0

    entry = float(threshold.get("entry_threshold", 0.60))
    exit_tau = float(threshold.get("exit_threshold", 0.52))
    score_gate = float(threshold.get("score_gate", 0.0))
    score_abs = abs(score) if np.isfinite(score) else 0.0

    if prev_side >= 0.5:
        return 1.0 if (probability >= exit_tau and score_abs >= score_gate) else 0.0
    return 1.0 if (probability >= entry and score_abs >= score_gate) else 0.0


def build_trade_frame(side: pd.Series, forward_return: pd.Series, cost_bps: float) -> pd.DataFrame:
    df = pd.DataFrame(index=side.index)
    df["side"] = side.fillna(0.0)
    df["forward_return"] = forward_return

    df["turnover"] = df["side"].diff().abs()
    if not df.empty:
        df.iloc[0, df.columns.get_loc("turnover")] = abs(df.iloc[0]["side"])

    df["gross_return"] = df["side"] * df["forward_return"]
    df["cost_drag"] = df["turnover"] * (cost_bps / 10000.0)
    df["net_return"] = df["gross_return"] - df["cost_drag"]

    df["equity_gross"] = (1.0 + df["gross_return"].fillna(0.0)).cumprod()
    df["equity_net"] = (1.0 + df["net_return"].fillna(0.0)).cumprod()
    df["drawdown_net"] = df["equity_net"] / df["equity_net"].cummax() - 1.0
    return df


def annualized_sharpe(returns: pd.Series) -> float:
    ann_ret = _annualized_return(returns)
    ann_vol = _annualized_vol(returns)
    if np.isnan(ann_ret) or np.isnan(ann_vol) or ann_vol <= 1e-12:
        return np.nan
    return float(ann_ret / ann_vol)


def trading_metrics(returns: pd.Series, turnover: pd.Series | None = None) -> Dict[str, float]:
    ann_ret = _annualized_return(returns)
    ann_vol = _annualized_vol(returns)
    sharpe = annualized_sharpe(returns)
    max_dd = _max_drawdown(returns)

    calmar = np.nan
    if np.isfinite(ann_ret) and np.isfinite(max_dd) and max_dd < 0:
        calmar = float(ann_ret / abs(max_dd))

    out = {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": float((returns.dropna() > 0).mean()) if returns.notna().any() else np.nan,
        "best_period": float(returns.max()) if returns.notna().any() else np.nan,
        "worst_period": float(returns.min()) if returns.notna().any() else np.nan,
        "periods": int(returns.dropna().shape[0]),
    }

    if turnover is not None:
        out["avg_turnover"] = float(turnover.dropna().mean()) if turnover.notna().any() else np.nan

    return out


def classification_metrics(
    target: pd.Series,
    side: pd.Series,
    probability: pd.Series,
    mode: str,
) -> Dict[str, float]:
    y = target.dropna().astype(int)
    idx = y.index.intersection(side.dropna().index)
    if len(idx) == 0:
        return {
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "accuracy": np.nan,
            "roc_auc": np.nan,
            "directional_accuracy_active": np.nan,
        }

    y_true = y.loc[idx]

    if mode == "long_short":
        pred_side = side.loc[idx].astype(int)
        pred_binary = (pred_side == 1).astype(int)

        active = pred_side != 0
        truth_dir = np.where(y_true == 1, 1, -1)
        pred_dir = pred_side.to_numpy()
        if active.any():
            directional_active = float((pred_dir[active.values] == truth_dir[active.values]).mean())
        else:
            directional_active = np.nan

        precision = float(precision_score(y_true, pred_binary, zero_division=0))
        recall = float(recall_score(y_true, pred_binary, zero_division=0))
        f1 = float(f1_score(y_true, pred_binary, zero_division=0))
        accuracy = float(accuracy_score(y_true, pred_binary))

        prob = probability.loc[idx]
        try:
            roc_auc = float(roc_auc_score(y_true, prob))
        except Exception:
            roc_auc = np.nan

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "directional_accuracy_active": directional_active,
        }

    y_pred = side.loc[idx].astype(int)
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    prob = probability.loc[idx]
    try:
        roc_auc = float(roc_auc_score(y_true, prob))
    except Exception:
        roc_auc = np.nan

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "directional_accuracy_active": np.nan,
    }


def run_walk_forward_model(data: pd.DataFrame, factor_cols: List[str], cfg: ModelConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: List[Dict[str, object]] = []
    weight_records: List[Dict[str, object]] = []
    latest_diag = pd.DataFrame()
    prev_side = 0.0

    for i in range(cfg.window_months, len(data)):
        train = data.iloc[i - cfg.window_months : i].copy()
        curr = data.iloc[i : i + 1].copy()

        train = train.dropna(subset=["Forward_Return", "Target_Up", "Target_Weight"])
        if len(train) < cfg.min_train_obs:
            continue

        diag = factor_diagnostics(train, factor_cols)
        latest_diag = diag.copy()

        selected = select_factors(diag, train[factor_cols], cfg)
        if len(selected) < cfg.min_factors:
            continue

        train_selected = train[selected]
        curr_selected = curr[selected]

        weights = correlation_aware_weights(train_selected, diag, cfg)
        score_train, cov_train = composite_score(train_selected, weights, cfg.min_score_coverage)
        score_curr, cov_curr = composite_score(curr_selected, weights, cfg.min_score_coverage)

        prob_train_features = train_selected.copy()
        prob_train_features["composite_score"] = score_train
        prob_curr_features = curr_selected.copy()
        prob_curr_features["composite_score"] = score_curr

        model_kind, model_obj = fit_probability_model(
            feature_df=prob_train_features,
            target=train["Target_Up"],
            sample_weight=train["Target_Weight"],
            cfg=cfg,
        )
        prob_train = predict_probability(model_kind, model_obj, prob_train_features)

        threshold, _ = tune_threshold(
            probabilities=prob_train,
            target=train["Target_Up"],
            forward_returns=train["Forward_Return"],
            scores=score_train,
            cfg=cfg,
        )

        prob_curr = predict_probability(model_kind, model_obj, prob_curr_features)
        this_prob = float(prob_curr.iloc[0]) if not prob_curr.empty else np.nan
        this_score = float(score_curr.iloc[0]) if not score_curr.empty else np.nan
        this_cov = float(cov_curr.iloc[0]) if not cov_curr.empty else np.nan

        side = side_from_probability(
            probability=this_prob,
            score=this_score,
            threshold=threshold,
            mode=cfg.mode,
            prev_side=prev_side,
        )
        if np.isfinite(side):
            prev_side = float(side)

        current_date = curr.index[0]
        records.append(
            {
                "Date": current_date,
                "raw_score": this_score,
                "prob_up": this_prob,
                "threshold": float(threshold.get("entry_threshold", np.nan)),
                "threshold_exit": float(threshold.get("exit_threshold", np.nan)),
                "threshold_score_gate": float(threshold.get("score_gate", np.nan)),
                "threshold_low": float(threshold.get("lower", np.nan)),
                "threshold_high": float(threshold.get("upper", np.nan)),
                "side": side,
                "score_coverage": this_cov,
                "selected_factor_count": len(selected),
                "Forward_Return": float(curr["Forward_Return"].iloc[0]),
                "Target_Up": int(curr["Target_Up"].iloc[0]),
            }
        )

        for factor_name, weight_value in weights.items():
            weight_records.append(
                {
                    "Date": current_date,
                    "factor": factor_name,
                    "weight": float(weight_value),
                }
            )

    signal_df = pd.DataFrame(records)
    if signal_df.empty:
        return signal_df, pd.DataFrame(), latest_diag

    signal_df = signal_df.sort_values("Date").reset_index(drop=True)
    split_idx = int(len(signal_df) * cfg.train_ratio)
    signal_df["Set"] = "train"
    signal_df.loc[split_idx:, "Set"] = "test"

    signal_df = signal_df.set_index("Date")
    trade_df = build_trade_frame(signal_df["side"], signal_df["Forward_Return"], cfg.transaction_cost_bps)
    trade_df = trade_df.drop(columns=["side"])
    combined = signal_df.join(trade_df, how="left")
    combined = combined.reset_index()

    weight_df = pd.DataFrame(weight_records)
    return combined, weight_df, latest_diag


def summarize_results(combined: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []

    for split in ["train", "test", "all"]:
        if split == "all":
            part = combined.copy()
        else:
            part = combined[combined["Set"] == split].copy()

        if part.empty:
            continue

        part = part.set_index("Date")

        cls = classification_metrics(
            target=part["Target_Up"],
            side=part["side"],
            probability=part["prob_up"],
            mode=cfg.mode,
        )

        trade = trading_metrics(part["net_return"], turnover=part["turnover"])

        row = {"split": split}
        row.update(cls)
        row.update(trade)
        rows.append(row)

    return pd.DataFrame(rows)


def build_model_bias_report(model_df: pd.DataFrame, factor_cols: List[str]) -> Tuple[pd.DataFrame, bool]:
    if "Target_Hurdle" in model_df.columns:
        target_expected = (model_df["SPX_Return"].shift(-1) > model_df["Target_Hurdle"]).astype(int)
    else:
        target_expected = (model_df["SPX_Return"].shift(-1) > 0).astype(int)
    target_match = model_df["Target_Up"].iloc[:-1].eq(target_expected.iloc[:-1])
    target_shift_pass = bool(target_match.all()) if not target_match.empty else False

    fwd = model_df["Forward_Return"].to_numpy(dtype=float)[:-1]
    next_ret = model_df["SPX_Return"].to_numpy(dtype=float)[1:]
    mask = np.isfinite(fwd) & np.isfinite(next_ret)
    if not np.any(mask):
        forward_alignment_rmse = np.nan
        forward_alignment_pass = False
    else:
        diff = fwd[mask] - next_ret[mask]
        forward_alignment_rmse = float(np.sqrt(np.mean(diff**2)))
        forward_alignment_pass = forward_alignment_rmse <= 1e-12

    suspicious_factor_count = 0
    for col in factor_cols:
        sample = pd.concat([model_df[col], model_df["Forward_Return"]], axis=1).dropna()
        if len(sample) < 30:
            continue
        same_ratio = float(np.isclose(sample.iloc[:, 0], sample.iloc[:, 1], rtol=0.0, atol=1e-12).mean())
        if same_ratio > 0.95:
            suspicious_factor_count += 1

    lookahead_proxy_pass = target_shift_pass and forward_alignment_pass and (suspicious_factor_count == 0)

    bias_df = pd.DataFrame(
        [
            {
                "scope": "model_bias",
                "target_shift_pass": target_shift_pass,
                "forward_alignment_rmse": forward_alignment_rmse,
                "forward_alignment_pass": forward_alignment_pass,
                "suspicious_factor_count": suspicious_factor_count,
                "lookahead_proxy_pass": lookahead_proxy_pass,
            }
        ]
    )
    return bias_df, bool(lookahead_proxy_pass)


def evaluate_oos_gate(summary_df: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    train = summary_df.loc[summary_df["split"] == "train"]
    test = summary_df.loc[summary_df["split"] == "test"]

    reasons: List[str] = []
    gate_pass = True

    if test.empty:
        gate_pass = False
        reasons.append("missing_test_split")
        test_row = pd.Series(dtype=float)
    else:
        test_row = test.iloc[0]

    if gate_pass:
        test_periods = int(test_row.get("periods", 0))
        test_recall = float(test_row.get("recall", np.nan))
        test_precision = float(test_row.get("precision", np.nan))
        test_sharpe = float(test_row.get("sharpe", np.nan))
        test_ann_return = float(test_row.get("ann_return", np.nan))

        if test_periods < cfg.min_test_periods:
            gate_pass = False
            reasons.append("insufficient_test_periods")
        if (not np.isfinite(test_recall)) or (test_recall < cfg.min_test_recall):
            gate_pass = False
            reasons.append("test_recall_below_min")
        if (not np.isfinite(test_precision)) or (test_precision < cfg.min_precision):
            gate_pass = False
            reasons.append("test_precision_below_min")
        if (not np.isfinite(test_sharpe)) or (test_sharpe < cfg.min_test_sharpe):
            gate_pass = False
            reasons.append("test_sharpe_below_min")
        if (not np.isfinite(test_ann_return)) or (test_ann_return <= 0):
            gate_pass = False
            reasons.append("test_ann_return_non_positive")

        if not train.empty:
            train_row = train.iloc[0]
            train_f1 = float(train_row.get("f1", np.nan))
            test_f1 = float(test_row.get("f1", np.nan))
            if np.isfinite(train_f1) and np.isfinite(test_f1) and train_f1 > 0:
                f1_drop = train_f1 - test_f1
                if f1_drop > cfg.max_f1_drop:
                    gate_pass = False
                    reasons.append("f1_drop_train_to_test_too_large")

            train_sharpe = float(train_row.get("sharpe", np.nan))
            if np.isfinite(train_sharpe) and np.isfinite(test_sharpe) and train_sharpe > 0:
                if test_sharpe < (0.25 * train_sharpe):
                    gate_pass = False
                    reasons.append("test_sharpe_too_low_vs_train")

    gate_row = {
        "oos_gate_pass": gate_pass,
        "reject_reason": ";".join(reasons) if reasons else "passed",
        "min_test_periods": cfg.min_test_periods,
        "min_test_recall": cfg.min_test_recall,
        "min_test_precision": cfg.min_precision,
        "min_test_sharpe": cfg.min_test_sharpe,
        "max_f1_drop": cfg.max_f1_drop,
    }

    if not test.empty:
        test_row = test.iloc[0]
        gate_row["test_periods"] = test_row.get("periods", np.nan)
        gate_row["test_recall"] = test_row.get("recall", np.nan)
        gate_row["test_precision"] = test_row.get("precision", np.nan)
        gate_row["test_f1"] = test_row.get("f1", np.nan)
        gate_row["test_sharpe"] = test_row.get("sharpe", np.nan)
        gate_row["test_ann_return"] = test_row.get("ann_return", np.nan)

    if not train.empty:
        train_row = train.iloc[0]
        gate_row["train_f1"] = train_row.get("f1", np.nan)
        gate_row["train_sharpe"] = train_row.get("sharpe", np.nan)

    return pd.DataFrame([gate_row])


def save_outputs(
    combined: pd.DataFrame,
    weight_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    data_qc_df: pd.DataFrame,
    bias_qc_df: pd.DataFrame,
    oos_gate_df: pd.DataFrame,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    signal_path = REPORTS_DIR / "primary_jq_signal.csv"
    weights_path = REPORTS_DIR / "primary_jq_weights_history.csv"
    diag_path = REPORTS_DIR / "primary_jq_factor_diagnostics_latest.csv"
    summary_path = REPORTS_DIR / "primary_jq_summary.csv"
    usage_path = REPORTS_DIR / "primary_jq_factor_usage.csv"
    data_qc_path = REPORTS_DIR / "primary_jq_data_qc.csv"
    bias_qc_path = REPORTS_DIR / "primary_jq_bias_qc.csv"
    gate_path = REPORTS_DIR / "primary_jq_oos_gate.csv"

    combined.to_csv(signal_path, index=False)
    weight_df.to_csv(weights_path, index=False)
    diagnostics_df.to_csv(diag_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    data_qc_df.to_csv(data_qc_path, index=False)
    bias_qc_df.to_csv(bias_qc_path, index=False)
    oos_gate_df.to_csv(gate_path, index=False)

    if not weight_df.empty:
        usage = (
            weight_df.groupby("factor")
            .agg(
                selected_periods=("weight", "size"),
                avg_weight=("weight", "mean"),
                median_weight=("weight", "median"),
                max_weight=("weight", "max"),
            )
            .reset_index()
            .sort_values("avg_weight", ascending=False)
        )
        usage.to_csv(usage_path, index=False)
    else:
        pd.DataFrame(columns=["factor", "selected_periods", "avg_weight", "median_weight", "max_weight"]).to_csv(usage_path, index=False)

    print(f"Saved: {signal_path}")
    print(f"Saved: {weights_path}")
    print(f"Saved: {diag_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {usage_path}")
    print(f"Saved: {data_qc_path}")
    print(f"Saved: {bias_qc_path}")
    print(f"Saved: {gate_path}")


def print_summary(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        print("No summary available (insufficient data after preprocessing).")
        return

    show_cols = [
        "split",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "roc_auc",
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "calmar",
        "avg_turnover",
    ]

    present = [c for c in show_cols if c in summary_df.columns]
    print("\nPrimary Model Summary")
    print("=" * 90)
    print(summary_df[present].to_string(index=False))


def build_model_dataset(cfg: ModelConfig) -> Tuple[pd.DataFrame, List[str], pd.DataFrame, pd.DataFrame, bool]:
    merged = load_merged_data()
    data_qc_df, data_qc_pass = build_data_qc_report(merged, cfg)

    raw_factors = build_raw_factors(merged)
    z_factors = build_oriented_z_factors(raw_factors, cfg)
    z_factors, coverage = enforce_missing_data_rules(z_factors, cfg)

    model_df = pd.DataFrame(index=merged["Date"])
    model_df["SPX_Return"] = merged["SPX_Return"].to_numpy()
    model_df["BCOM_Return"] = merged["BCOM_Return"].to_numpy()
    model_df["Treasury10Y_Return"] = merged["Treasury10Y_Return"].to_numpy()
    model_df["IG_Corp_Return"] = merged["IG_Corp_Return"].to_numpy()

    model_df["Forward_Return"] = model_df["SPX_Return"].shift(-1)
    target_up, target_hurdle, target_weight = build_primary_target_and_weights(model_df["Forward_Return"], cfg)
    model_df["Target_Up"] = target_up
    model_df["Target_Hurdle"] = target_hurdle
    model_df["Target_Weight"] = target_weight

    model_df["Risk_BCOM"] = model_df["BCOM_Return"]
    model_df["Risk_Rates"] = model_df["Treasury10Y_Return"]
    model_df["Risk_Credit"] = model_df["IG_Corp_Return"]
    model_df["Risk_Vol"] = model_df["SPX_Return"].rolling(6).std(ddof=1)

    for col in z_factors.columns:
        model_df[col] = z_factors[col]

    model_df["factor_coverage_raw"] = coverage

    factor_cols = [spec.name for spec in FACTOR_SPECS]

    valid = model_df["Forward_Return"].notna()
    model_df = model_df.loc[valid].copy()

    bias_qc_df, bias_qc_pass = build_model_bias_report(model_df, factor_cols)
    qc_pass = bool(data_qc_pass and bias_qc_pass)

    return model_df, factor_cols, data_qc_df, bias_qc_df, qc_pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Primary side model from Joubert + QuantAtoZ principles.")
    parser.add_argument(
        "--mode",
        choices=["long_only", "long_short"],
        default="long_only",
        help="Signal mode for side output.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.60,
        help="Sequential train ratio for final evaluation split.",
    )
    parser.add_argument(
        "--window-months",
        type=int,
        default=120,
        help="Rolling estimation window used for diagnostics/weights/probability fitting.",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=5.0,
        help="One-way transaction cost in basis points.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = ModelConfig(
        mode=args.mode,
        train_ratio=args.train_ratio,
        window_months=args.window_months,
        transaction_cost_bps=args.transaction_cost_bps,
    )

    print("Building Joubert+QuantAtoZ primary model...")
    model_df, factor_cols, data_qc_df, bias_qc_df, qc_pass = build_model_dataset(cfg)
    if not qc_pass:
        print("Warning: data/bias QC checks did not fully pass. Output will be marked as rejected.")

    combined, weight_df, diagnostics_df = run_walk_forward_model(model_df, factor_cols, cfg)
    if combined.empty:
        print("No results generated. Try reducing --window-months or min history settings.")
        return

    summary_df = summarize_results(combined, cfg)
    oos_gate_df = evaluate_oos_gate(summary_df, cfg)

    if not qc_pass:
        oos_gate_df.loc[0, "oos_gate_pass"] = False
        prior = str(oos_gate_df.loc[0, "reject_reason"])
        extra = "data_or_bias_qc_failed"
        oos_gate_df.loc[0, "reject_reason"] = f"{prior};{extra}" if prior and prior != "passed" else extra

    save_outputs(
        combined=combined,
        weight_df=weight_df,
        diagnostics_df=diagnostics_df,
        summary_df=summary_df,
        data_qc_df=data_qc_df,
        bias_qc_df=bias_qc_df,
        oos_gate_df=oos_gate_df,
    )
    print_summary(summary_df)
    print("\nOOS Gate")
    print("=" * 90)
    print(oos_gate_df.to_string(index=False))


if __name__ == "__main__":
    main()
