#!/usr/bin/env python3
"""Build two structured HTML reports:
1) Primary Joubert+QuantAtoZ model report
2) Trend Aware (original) model report

Both reports follow the required 7-section structure and export supporting
tables/figures as separate files.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import hilbert


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RAW_DIR = ROOT / "data" / "raw"

RAW_FILES = {
    "spx": RAW_DIR / "spx.xlsx",
    "rates": RAW_DIR / "treasury_10y.xlsx",
    "commodities": RAW_DIR / "bcom.xlsx",
    "credit": RAW_DIR / "corp_bonds.xlsx",
}


@dataclass
class ModelBundle:
    key: str
    name: str
    returns: pd.Series
    side: pd.Series
    target: pd.Series
    spx_ref: pd.Series
    indicators: pd.DataFrame
    concept: str
    rationale: str
    indicator_choice: str


def find_header_row(raw_df: pd.DataFrame) -> int:
    for i, row in raw_df.iterrows():
        values = [str(v).strip().lower() for v in row.tolist() if pd.notna(v)]
        if "date" in values and ("px_last" in values or "close" in values):
            return int(i)
    raise ValueError("Could not find header row with Date and PX_LAST/Close.")


def load_single_price(path: Path, alias: str) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None, dtype=object)
    header_row = find_header_row(raw)

    header = raw.iloc[header_row].tolist()
    data = raw.iloc[header_row + 1 :].copy()
    data.columns = header
    data = data.loc[:, [col for col in data.columns if pd.notna(col)]]
    data.columns = [str(c).strip() for c in data.columns]

    date_col = next((c for c in data.columns if c.lower() == "date"), None)
    price_col = next((c for c in data.columns if c.upper() == "PX_LAST" or "close" in c.lower()), None)
    if date_col is None or price_col is None:
        raise ValueError(f"Missing Date/PX_LAST columns in {path}")

    clean = data[[date_col, price_col]].copy()
    clean.columns = ["Date", alias]
    clean["Date"] = pd.to_datetime(clean["Date"], errors="coerce")
    clean[alias] = pd.to_numeric(clean[alias], errors="coerce")
    clean = clean.dropna(subset=["Date"]).drop_duplicates(subset=["Date"], keep="last")
    clean = clean.sort_values("Date").reset_index(drop=True)
    return clean


def load_prices() -> pd.DataFrame:
    parts = []
    for key, path in RAW_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        parts.append(load_single_price(path, key))

    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on="Date", how="outer")

    merged = merged.sort_values("Date").reset_index(drop=True)
    cols = ["spx", "rates", "commodities", "credit"]
    merged[cols] = merged[cols].ffill()
    merged = merged.dropna(subset=cols).set_index("Date")
    return merged


def compute_trend_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=prices.index)

    def phase(s: pd.Series) -> pd.Series:
        detrended = s - s.rolling(12).mean()
        arr = np.angle(hilbert(detrended.fillna(0)))
        return pd.Series(arr, index=s.index)

    out["rates_phase"] = phase(prices["rates"])
    out["credit_phase"] = phase(prices["credit"])
    out["spx_phase"] = phase(prices["spx"])
    out["commodities_phase"] = phase(prices["commodities"])
    out["spx_roc_3m"] = prices["spx"].pct_change(3)
    return out


def compute_primary_jq_indicators(prices: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=prices.index)
    out["spx_mom_3m"] = prices["spx"].pct_change(3)
    out["spx_mom_12m"] = prices["spx"].pct_change(12)
    out["spx_vs_bcom_6m"] = prices["spx"].pct_change(6) - prices["commodities"].pct_change(6)
    out["rates_change_6m"] = prices["rates"].diff(6)
    out["credit_carry_3m"] = rets["credit"].rolling(3).mean()
    out["vol_regime_3_12"] = rets["spx"].rolling(3).std(ddof=1) / rets["spx"].rolling(12).std(ddof=1)
    out["drawdown_6m"] = prices["spx"] / prices["spx"].rolling(6).max() - 1.0
    out["credit_vs_rates_3m"] = rets["credit"].rolling(3).mean() - rets["rates"].rolling(3).mean()
    return out


def map_to_next_available_dates(source_index: Iterable[pd.Timestamp], ref_index: Iterable[pd.Timestamp]) -> pd.DatetimeIndex:
    src = pd.DatetimeIndex(source_index)
    ref = pd.DatetimeIndex(ref_index).sort_values().unique()
    ref_arr = ref.to_numpy()

    mapped = []
    for d in src:
        pos = ref_arr.searchsorted(d.to_datetime64(), side="right")
        mapped.append(pd.Timestamp(ref_arr[pos]) if pos < len(ref_arr) else pd.NaT)
    return pd.DatetimeIndex(mapped)


def pct(v: float | int | None) -> str:
    if v is None or not np.isfinite(v):
        return "N/A"
    return f"{v * 100:.2f}%"


def num(v: float | int | None, d: int = 3) -> str:
    if v is None or not np.isfinite(v):
        return "N/A"
    return f"{v:.{d}f}"


def clean_binary_side(side: pd.Series) -> pd.Series:
    s = pd.to_numeric(side, errors="coerce").fillna(0.0)
    return (s > 0).astype(int)


def classification_metrics(side: pd.Series, target: pd.Series) -> Dict[str, float]:
    y_pred = clean_binary_side(side)
    y_true = pd.to_numeric(target, errors="coerce")
    valid_idx = y_pred.index.intersection(y_true.dropna().index)
    if len(valid_idx) == 0:
        return {"precision": np.nan, "recall": np.nan, "f1": np.nan}

    yp = y_pred.loc[valid_idx].astype(int)
    yt = y_true.loc[valid_idx].astype(int)

    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def annualized_return(returns: pd.Series) -> float:
    r = returns.dropna().astype(float)
    if r.empty:
        return np.nan
    eq = (1.0 + r).cumprod()
    return float(eq.iloc[-1] ** (12.0 / len(r)) - 1.0)


def annualized_vol(returns: pd.Series) -> float:
    r = returns.dropna().astype(float)
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(12.0))


def max_drawdown(returns: pd.Series) -> float:
    r = returns.dropna().astype(float)
    if r.empty:
        return np.nan
    eq = (1.0 + r).cumprod()
    dd = eq / eq.cummax() - 1.0
    return float(dd.min())


def information_ratio(returns: pd.Series, benchmark: pd.Series) -> float:
    idx = returns.dropna().index.intersection(benchmark.dropna().index)
    if len(idx) < 2:
        return np.nan
    active = returns.loc[idx] - benchmark.loc[idx]
    te = float(active.std(ddof=1) * np.sqrt(12.0))
    if te <= 1e-12:
        return 0.0
    return float((active.mean() * 12.0) / te)


def avg_abs_offdiag_corr(df: pd.DataFrame) -> float:
    d = df.dropna()
    if d.shape[1] < 2 or d.shape[0] < 5:
        return np.nan
    corr = d.corr().to_numpy(dtype=float)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    vals = np.abs(corr[mask])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(vals.mean())


def avg_abs_indicator_to_spx_corr(indicators: pd.DataFrame, spx_ref: pd.Series) -> float:
    cols = []
    for col in indicators.columns:
        pair = pd.concat([indicators[col], spx_ref], axis=1).dropna()
        if len(pair) < 10:
            continue
        c = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
        if np.isfinite(c):
            cols.append(abs(c))
    if not cols:
        return np.nan
    return float(np.mean(cols))


def compute_metric_row(
    label: str,
    returns: pd.Series,
    side: pd.Series,
    target: pd.Series,
    spx_ref: pd.Series,
    indicators: pd.DataFrame | None = None,
) -> Dict[str, float | str]:
    rets = returns.dropna().astype(float)
    if rets.empty:
        return {
            "Model": label,
            "Annualized Return": np.nan,
            "Volatility": np.nan,
            "Sharpe Ratio": np.nan,
            "Max Drawdown": np.nan,
            "Win Rate": np.nan,
            "Cumulative Return": np.nan,
            "Information Ratio": np.nan,
            "Strategy-SPX Correlation": np.nan,
            "Indicator Pair Correlation": np.nan,
            "Indicator-to-SPX Correlation": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1 Score": np.nan,
            "Periods": 0,
        }

    ann_ret = annualized_return(rets)
    ann_vol = annualized_vol(rets)
    sharpe = float(ann_ret / ann_vol) if np.isfinite(ann_ret) and np.isfinite(ann_vol) and ann_vol > 1e-12 else np.nan
    mdd = max_drawdown(rets)
    win_rate = float((rets > 0).mean())
    cumulative = float((1.0 + rets).prod() - 1.0)

    idx = rets.index.intersection(spx_ref.dropna().index)
    corr_spx = float(rets.loc[idx].corr(spx_ref.loc[idx])) if len(idx) >= 3 else np.nan
    ir = information_ratio(rets, spx_ref)

    cls = classification_metrics(side=side, target=target)

    if indicators is None:
        ind_pair = np.nan
        ind_spx = np.nan
    else:
        ind_pair = avg_abs_offdiag_corr(indicators)
        ind_spx = avg_abs_indicator_to_spx_corr(indicators, spx_ref)

    return {
        "Model": label,
        "Annualized Return": ann_ret,
        "Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": mdd,
        "Win Rate": win_rate,
        "Cumulative Return": cumulative,
        "Information Ratio": ir,
        "Strategy-SPX Correlation": corr_spx,
        "Indicator Pair Correlation": ind_pair,
        "Indicator-to-SPX Correlation": ind_spx,
        "Precision": cls["precision"],
        "Recall": cls["recall"],
        "F1 Score": cls["f1"],
        "Periods": int(len(rets)),
    }


def format_metrics_table(df: pd.DataFrame, include_stars: bool = False) -> pd.DataFrame:
    out = df.copy()
    pct_cols = [
        "Annualized Return",
        "Volatility",
        "Max Drawdown",
        "Win Rate",
        "Cumulative Return",
        "Precision",
        "Recall",
        "F1 Score",
    ]
    num_cols = [
        "Sharpe Ratio",
        "Information Ratio",
        "Strategy-SPX Correlation",
        "Indicator Pair Correlation",
        "Indicator-to-SPX Correlation",
    ]
    higher_better = {
        "Annualized Return": True,
        "Volatility": False,
        "Sharpe Ratio": True,
        "Max Drawdown": True,  # less negative is better (max value)
        "Win Rate": True,
        "Cumulative Return": True,
        "Information Ratio": True,
        "Strategy-SPX Correlation": False,
        "Indicator Pair Correlation": False,
        "Indicator-to-SPX Correlation": False,
        "Precision": True,
        "Recall": True,
        "F1 Score": True,
    }

    star_mask: Dict[Tuple[int, str], bool] = {}
    if include_stars:
        for col, hb in higher_better.items():
            vals = pd.to_numeric(out[col], errors="coerce")
            if vals.notna().any():
                best = vals.max() if hb else vals.min()
                for idx in out.index:
                    v = vals.loc[idx]
                    star_mask[(idx, col)] = bool(np.isfinite(v) and np.isclose(v, best, rtol=1e-10, atol=1e-12))

    for col in pct_cols:
        vals = []
        for idx, v in out[col].items():
            text = pct(float(v)) if np.isfinite(float(v)) else "N/A"
            if include_stars and star_mask.get((idx, col), False):
                text = f"{text}*"
            vals.append(text)
        out[col] = vals

    for col in num_cols:
        vals = []
        for idx, v in out[col].items():
            text = num(float(v), 3) if np.isfinite(float(v)) else "N/A"
            if include_stars and star_mask.get((idx, col), False):
                text = f"{text}*"
            vals.append(text)
        out[col] = vals

    out["Periods"] = out["Periods"].astype(int).astype(str)
    return out


def build_trend_bundle(prices: pd.DataFrame, rets: pd.DataFrame) -> ModelBundle:
    weights = pd.read_csv(ROOT / "monthly_allocations.csv", parse_dates=["Date"]).set_index("Date")
    weights = weights[["spx", "rates", "commodities", "credit"]].astype(float)

    common_idx = weights.index.intersection(rets.index)
    weights = weights.loc[common_idx].sort_index()
    aligned = rets.loc[common_idx, ["spx", "rates", "commodities", "credit"]].sort_index()

    strat_ret = (weights.shift(1) * aligned).sum(axis=1).dropna()
    target = (rets["spx"].reindex(strat_ret.index) > 0).astype(int)
    side = pd.Series(1, index=strat_ret.index, dtype=int)
    spx_ref = rets["spx"].reindex(strat_ret.index)

    indicators = compute_trend_indicators(prices)
    indicators_realized = indicators.copy()
    indicators_realized.index = map_to_next_available_dates(indicators_realized.index, rets.index)
    indicators_realized = indicators_realized.loc[indicators_realized.index.notna()]
    if indicators_realized.index.has_duplicates:
        indicators_realized = indicators_realized.groupby(level=0).last()
    indicators_realized = indicators_realized.reindex(strat_ret.index)

    return ModelBundle(
        key="trend_aware",
        name="Trend Aware (Original)",
        returns=strat_ret,
        side=side,
        target=target,
        spx_ref=spx_ref,
        indicators=indicators_realized,
        concept=(
            "Cross-asset allocation model that reads cyclical phase behavior "
            "(rates, credit, equities, commodities) plus equity momentum."
        ),
        rationale=(
            "If macro cycle phases and short-term momentum shift together, "
            "portfolio weights should tilt toward assets with stronger next-period edge."
        ),
        indicator_choice=(
            "Indicators are cycle phase proxies (Hilbert transform) and 3M SPX ROC to capture "
            "regime direction and acceleration."
        ),
    )


def build_primary_jq_bundle(prices: pd.DataFrame, rets: pd.DataFrame) -> ModelBundle:
    df = pd.read_csv(REPORTS / "primary_jq_signal.csv", parse_dates=["Date"]).set_index("Date")
    df = df.sort_index()

    # primary_jq rows are decision-date. Shift to realized month for return comparability.
    realized_idx = map_to_next_available_dates(df.index, rets.index)
    returns = pd.Series(df["net_return"].to_numpy(dtype=float), index=realized_idx, name="primary_jq_net")
    side = pd.Series(df["side"].to_numpy(dtype=float), index=realized_idx, name="side")
    target = pd.Series(df["Target_Up"].to_numpy(dtype=float), index=realized_idx, name="target")
    valid = returns.index.notna()
    returns = returns.loc[valid]
    side = side.loc[valid]
    target = target.loc[valid]
    if returns.index.has_duplicates:
        returns = returns.groupby(level=0).last()
    if side.index.has_duplicates:
        side = side.groupby(level=0).last()
    if target.index.has_duplicates:
        target = target.groupby(level=0).last()
    spx_ref = rets["spx"].reindex(returns.index)
    side = side.reindex(returns.index)
    target = target.reindex(returns.index)

    ind = compute_primary_jq_indicators(prices, rets)
    ind_realized = ind.copy()
    ind_realized.index = map_to_next_available_dates(ind_realized.index, rets.index)
    ind_realized = ind_realized.loc[ind_realized.index.notna()]
    if ind_realized.index.has_duplicates:
        ind_realized = ind_realized.groupby(level=0).last()
    ind_realized = ind_realized.reindex(returns.index)

    return ModelBundle(
        key="primary_jq",
        name="Primary Joubert+QuantAtoZ",
        returns=returns,
        side=side,
        target=target,
        spx_ref=spx_ref,
        indicators=ind_realized,
        concept=(
            "Directional primary model that combines multi-theme factors with "
            "correlation-aware weighting and walk-forward thresholding."
        ),
        rationale=(
            "Design prioritizes high opportunity recall while keeping quality controls "
            "on factor significance, stability, and out-of-sample behavior."
        ),
        indicator_choice=(
            "Indicators span momentum, cross-asset dispersion, rates, credit, volatility regime, "
            "and drawdown to avoid single-regime dependence."
        ),
    )


def build_primary_unified_bundle(prices: pd.DataFrame, rets: pd.DataFrame) -> ModelBundle:
    backtest = pd.read_csv(REPORTS / "primary_v1_backtest.csv", parse_dates=["Date"]).set_index("Date")
    signal = pd.read_csv(REPORTS / "primary_v1_signal.csv", parse_dates=["Date"]).set_index("Date")
    signal = signal.sort_index()
    backtest = backtest.sort_index()

    returns = backtest["net_return"].astype(float)

    side_map = {"BUY": 1.0, "SELL": 0.0, "HOLD": np.nan}
    raw_side = signal["signal"].map(side_map)
    side = raw_side.ffill().fillna(0.0)
    side = side.reindex(returns.index).ffill().fillna(0.0)

    target = (rets["spx"].reindex(returns.index) > 0).astype(int)
    spx_ref = rets["spx"].reindex(returns.index)

    indicators = signal[
        [
            "spx_trend",
            "bcom_trend",
            "credit_vs_rates",
            "risk_breadth",
            "composite_score",
        ]
    ].reindex(returns.index)

    return ModelBundle(
        key="primary_unified",
        name="Primary Unified V1",
        returns=returns,
        side=side,
        target=target,
        spx_ref=spx_ref,
        indicators=indicators,
        concept=(
            "Composite z-score primary signal combining trend, cross-asset, credit, and risk-breadth inputs."
        ),
        rationale=(
            "A compact factor stack seeks robust directional inference with low model complexity."
        ),
        indicator_choice=(
            "Indicators were selected to blend macro-sensitive and risk-sensitive exposures in one score."
        ),
    )


def build_realized_benchmarks(prices: pd.DataFrame, rets: pd.DataFrame) -> Dict[str, Dict[str, pd.Series]]:
    fwd_pos = (prices["spx"].rolling(10).mean() > prices["spx"].rolling(12).mean()).astype(int)
    simple_trend_returns = fwd_pos.shift(1).fillna(0.0) * rets["spx"]
    simple_trend_side = fwd_pos.shift(1).fillna(0.0)

    eq25 = rets[["spx", "rates", "commodities", "credit"]].mean(axis=1)
    bench_6040 = 0.60 * rets["spx"] + 0.40 * rets["rates"]

    return {
        "Buy & Hold SPX": {"returns": rets["spx"], "side": pd.Series(1, index=rets.index)},
        "60/40 SPX-Treasury": {"returns": bench_6040, "side": pd.Series(1, index=rets.index)},
        "EqualWeight 25/25/25/25": {"returns": eq25, "side": pd.Series(1, index=rets.index)},
        "Simple Trend (SMA10/12)": {"returns": simple_trend_returns, "side": simple_trend_side},
    }


def save_plot_equity(
    out_path: Path,
    focus_name: str,
    series_map: Dict[str, pd.Series],
) -> None:
    plt.figure(figsize=(11, 6))
    for name, s in series_map.items():
        r = s.dropna().astype(float)
        if r.empty:
            continue
        eq = (1.0 + r).cumprod()
        lw = 2.8 if name == focus_name else 1.8
        alpha = 1.0 if name == focus_name else 0.85
        plt.plot(eq.index, eq.values, linewidth=lw, alpha=alpha, label=name)
    plt.yscale("log")
    plt.title("Equity Curves")
    plt.ylabel("Growth of $1")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_plot_drawdown(out_path: Path, focus_name: str, series_map: Dict[str, pd.Series]) -> None:
    plt.figure(figsize=(11, 6))
    for name, s in series_map.items():
        r = s.dropna().astype(float)
        if r.empty:
            continue
        eq = (1.0 + r).cumprod()
        dd = eq / eq.cummax() - 1.0
        lw = 2.8 if name == focus_name else 1.8
        alpha = 1.0 if name == focus_name else 0.85
        plt.plot(dd.index, dd.values, linewidth=lw, alpha=alpha, label=name)
    plt.title("Drawdown Curves")
    plt.ylabel("Drawdown")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def rolling_sharpe(returns: pd.Series, window: int = 12) -> pd.Series:
    r = returns.dropna().astype(float)
    if r.empty:
        return r
    return (r.rolling(window).mean() * 12.0) / (r.rolling(window).std(ddof=1) * np.sqrt(12.0))


def save_plot_rolling_sharpe(out_path: Path, focus_name: str, series_map: Dict[str, pd.Series]) -> None:
    plt.figure(figsize=(11, 6))
    for name, s in series_map.items():
        rs = rolling_sharpe(s, window=12).dropna()
        if rs.empty:
            continue
        lw = 2.8 if name == focus_name else 1.8
        alpha = 1.0 if name == focus_name else 0.85
        plt.plot(rs.index, rs.values, linewidth=lw, alpha=alpha, label=name)
    plt.axhline(0.0, color="#374151", linewidth=1.0, linestyle="--")
    plt.title("Rolling 12-Month Sharpe")
    plt.ylabel("Sharpe")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_plot_metric_bars(out_path: Path, rows: pd.DataFrame) -> None:
    display = rows.copy()
    display = display.set_index("Model")
    metrics = ["Annualized Return", "Volatility", "Sharpe Ratio", "F1 Score", "Information Ratio"]
    metrics = [m for m in metrics if m in display.columns]
    if not metrics:
        return

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 2.5 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        vals = pd.to_numeric(display[metric], errors="coerce")
        ax.bar(display.index, vals, color=["#0d9488", "#334155", "#7c3aed"][: len(vals)])
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
            tick.set_ha("right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_plot_wave(out_path: Path, indicators: pd.DataFrame, title: str) -> None:
    # Prefer phase-like columns if available, else normalize first 4 columns and draw as "waves".
    cols = [c for c in indicators.columns if "phase" in c.lower()]
    if len(cols) < 2:
        cols = list(indicators.columns[:4])
    cols = cols[:4]

    data = indicators[cols].copy().dropna()
    if data.empty:
        data = indicators[cols].copy().fillna(0.0)

    if data.shape[0] > 120:
        data = data.iloc[-120:]

    # Normalize each column to roughly [-1, 1] for conceptual wave view.
    for col in data.columns:
        s = data[col]
        sd = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        if sd > 1e-12:
            data[col] = (s - s.mean()) / sd
        data[col] = data[col].clip(-2.5, 2.5)

    x = np.arange(len(data))
    plt.figure(figsize=(11, 5.5))
    base_colors = ["#0284c7", "#16a34a", "#f59e0b", "#ef4444"]
    for i, col in enumerate(data.columns):
        y = data[col].to_numpy(dtype=float)
        plt.plot(x, y, linewidth=2.2, color=base_colors[i % len(base_colors)], label=col)
        plt.fill_between(x, 0, y, alpha=0.08, color=base_colors[i % len(base_colors)])
    plt.axhline(0.0, color="#334155", linestyle="--", linewidth=1.0)
    plt.title(title)
    plt.xlabel("Recent Months")
    plt.ylabel("Normalized Wave Level")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def save_plot_corr_heatmap(out_path: Path, corr_df: pd.DataFrame, title: str) -> None:
    if corr_df.empty:
        return
    vals = corr_df.to_numpy(dtype=float)
    plt.figure(figsize=(8.8, 7.0))
    im = plt.imshow(vals, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr_df.index)), corr_df.index)
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def to_html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=False, border=0, classes="tbl")


def model_score_for_research(row: pd.Series) -> float:
    # Heuristic model ranking score for section-3 highlight.
    ann = float(row.get("Annualized Return", np.nan))
    sharpe = float(row.get("Sharpe Ratio", np.nan))
    f1 = float(row.get("F1 Score", np.nan))
    ir = float(row.get("Information Ratio", np.nan))
    mdd = float(row.get("Max Drawdown", np.nan))

    ann = ann if np.isfinite(ann) else 0.0
    sharpe = sharpe if np.isfinite(sharpe) else 0.0
    f1 = f1 if np.isfinite(f1) else 0.0
    ir = ir if np.isfinite(ir) else 0.0
    mdd = mdd if np.isfinite(mdd) else -1.0
    return ann * 120.0 + sharpe * 30.0 + f1 * 20.0 + ir * 25.0 + mdd * 15.0


def make_report(
    focus: ModelBundle,
    peer: ModelBundle,
    alt3: ModelBundle,
    benchmarks: Dict[str, Dict[str, pd.Series]],
    report_html: Path,
    assets_dir: Path,
) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Core comparison: focus vs peer
    focus_row = compute_metric_row(
        label=focus.name,
        returns=focus.returns,
        side=focus.side,
        target=focus.target,
        spx_ref=focus.spx_ref,
        indicators=focus.indicators,
    )
    peer_row = compute_metric_row(
        label=peer.name,
        returns=peer.returns,
        side=peer.side,
        target=peer.target,
        spx_ref=peer.spx_ref,
        indicators=peer.indicators,
    )
    pair_df = pd.DataFrame([focus_row, peer_row])
    pair_df.to_csv(assets_dir / "focus_vs_peer_metrics.csv", index=False)

    # Benchmarks table on focus period.
    focus_idx = focus.returns.dropna().index
    bench_rows = [focus_row]
    for bname, bundle in benchmarks.items():
        bret = bundle["returns"].reindex(focus_idx)
        bside = bundle["side"].reindex(focus_idx).fillna(0.0)
        btarget = focus.target.reindex(focus_idx)
        b_spx = focus.spx_ref.reindex(focus_idx)
        bench_rows.append(
            compute_metric_row(
                label=bname,
                returns=bret,
                side=bside,
                target=btarget,
                spx_ref=b_spx,
                indicators=None,
            )
        )
    bench_df = pd.DataFrame(bench_rows)
    bench_df.to_csv(assets_dir / "benchmarks_comparison.csv", index=False)

    # Indicators research (3 alternatives).
    alt_rows = [
        focus_row,
        peer_row,
        compute_metric_row(
            label=alt3.name,
            returns=alt3.returns,
            side=alt3.side,
            target=alt3.target,
            spx_ref=alt3.spx_ref,
            indicators=alt3.indicators,
        ),
    ]
    alt_df = pd.DataFrame(alt_rows)
    alt_df["Research Score"] = alt_df.apply(model_score_for_research, axis=1)
    alt_df = alt_df.sort_values("Research Score", ascending=False).reset_index(drop=True)
    alt_df.to_csv(assets_dir / "alternative_models_research.csv", index=False)
    winner_name = str(alt_df.iloc[0]["Model"])

    # Indicator correlation matrix for focus.
    focus_indicator_corr = focus.indicators.dropna().corr()
    focus_indicator_corr.to_csv(assets_dir / "focus_indicator_correlation_matrix.csv")

    # Save plots.
    # Performance visuals: focus vs peer vs 4 benchmarks.
    plot_series = {
        focus.name: focus.returns,
        peer.name: peer.returns,
    }
    for bname, b in benchmarks.items():
        plot_series[bname] = b["returns"].reindex(focus_idx)

    eq_path = assets_dir / "equity_curves.png"
    dd_path = assets_dir / "drawdown_curves.png"
    rs_path = assets_dir / "rolling_sharpe.png"
    mb_path = assets_dir / "executive_metric_bars.png"
    wave_path = assets_dir / "wave_visualization.png"
    heatmap_path = assets_dir / "indicator_corr_heatmap.png"

    save_plot_equity(eq_path, focus.name, plot_series)
    save_plot_drawdown(dd_path, focus.name, plot_series)
    save_plot_rolling_sharpe(rs_path, focus.name, plot_series)
    save_plot_metric_bars(mb_path, pair_df)
    save_plot_wave(
        wave_path,
        focus.indicators,
        "Conceptual Wave View (Herbert/Hilbert-Style Regime Intuition)",
    )
    save_plot_corr_heatmap(
        heatmap_path,
        focus_indicator_corr,
        "Indicator Correlation Matrix",
    )

    # Save supporting headline metrics table.
    focus_exec = pd.DataFrame([focus_row])
    focus_exec.to_csv(assets_dir / "executive_summary_metrics.csv", index=False)

    # Executive display table and benchmark display with stars.
    pair_display = format_metrics_table(pair_df, include_stars=True)
    bench_display = format_metrics_table(bench_df, include_stars=True)
    alt_display = format_metrics_table(alt_df.drop(columns=["Research Score"]), include_stars=True)

    # Benchmark explanations.
    benchmark_explanations = [
        (
            "Buy & Hold SPX",
            "Baseline equity beta. If the primary model cannot beat passive SPX on risk-adjusted terms, timing adds limited value.",
        ),
        (
            "60/40 SPX-Treasury",
            "Institutional allocation baseline. Tests whether dynamic primary signals beat simple diversified policy exposure.",
        ),
        (
            "EqualWeight 25/25/25/25",
            "Neutral cross-asset baseline across equities, rates, commodities, and credit. Useful for checking concentration effects.",
        ),
        (
            "Simple Trend (SMA10/12)",
            "Low-complexity trend filter benchmark. Verifies whether richer indicator stacks outperform a classic trend rule.",
        ),
    ]

    # Deep-dive bullets.
    if focus.key == "primary_jq":
        deep_dive_title = "Primary Joubert+QuantAtoZ - What Goes In"
        deep_dive_bullets = [
            "Input layer: clean monthly cross-asset data (SPX, commodities, rates, credit) with timestamp alignment and forward target construction.",
            "Indicator layer: momentum, cross-asset spread, rate change, credit carry, volatility regime, and drawdown features.",
            "Robustness layer: orientation-consistent normalization and outlier control to keep features comparable.",
            "Combination layer: correlation-aware weighting to avoid double-counting highly similar signals.",
            "Output layer: directional side plus continuous probability score for threshold-driven activation.",
        ]
    else:
        deep_dive_title = "Trend Aware (Original) - What Goes In"
        deep_dive_bullets = [
            "Input layer: monthly prices for SPX, rates, commodities, and credit.",
            "Indicator layer: cycle-phase extraction (Hilbert-based) and 3M SPX momentum.",
            "Model layer: gradient-boosted regressors estimate next-period relative edge per asset.",
            "Allocation layer: equal-weight core with z-score tilts, clipped to practical allocation bounds.",
            "Output layer: monthly cross-asset portfolio weights translated into next-month returns.",
        ]

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{focus.name} - Structured Primary Model Report</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0b0f1a;
      --bg2: #111827;
      --panel: #131c2e;
      --panel2: #1c2840;
      --line: #2a3d5e;
      --ink: #e8edf5;
      --ink2: #a8b8cc;
      --ink3: #6a7f99;
      --teal: #0d9488;
      --orange: #f59e0b;
      --blue: #3b82f6;
      --green: #22c55e;
      --red: #ef4444;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: radial-gradient(circle at 82% -10%, #213a66 0%, var(--bg) 35%);
      color: var(--ink);
      font-family: "DM Sans", sans-serif;
      line-height: 1.45;
    }}
    .wrap {{
      max-width: 1460px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1, h2, h3 {{
      margin: 0 0 10px;
      font-weight: 700;
      letter-spacing: 0.01em;
    }}
    h1 {{
      font-family: "DM Serif Display", serif;
      font-size: 38px;
      letter-spacing: 0.015em;
    }}
    .sub {{
      color: var(--ink2);
      margin-bottom: 18px;
    }}
    .section {{
      background: linear-gradient(180deg, rgba(28,40,64,0.95) 0%, rgba(19,28,46,0.95) 100%);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      margin-bottom: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
      gap: 10px;
    }}
    .tile {{
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid #324869;
      border-radius: 12px;
      padding: 10px;
    }}
    .tile .k {{
      font-size: 12px;
      color: var(--ink2);
      margin-bottom: 5px;
    }}
    .tile .v {{
      font-size: 18px;
      font-weight: 700;
      color: #ecfeff;
      font-family: "JetBrains Mono", monospace;
    }}
    .imgs {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 12px;
      margin-top: 10px;
    }}
    .imgbox {{
      background: #0f172a;
      border: 1px solid #324869;
      border-radius: 12px;
      padding: 8px;
    }}
    .imgbox img {{
      width: 100%;
      border-radius: 8px;
      display: block;
    }}
    .tbl {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12.8px;
      margin-top: 8px;
    }}
    .tbl th, .tbl td {{
      border: 1px solid #2f4667;
      padding: 6px 8px;
      text-align: right;
      white-space: nowrap;
    }}
    .tbl th:first-child, .tbl td:first-child {{
      text-align: left;
    }}
    .tbl th {{
      background: #12213b;
      color: #cde4ff;
    }}
    .mono {{
      font-family: "JetBrains Mono", monospace;
    }}
    .small {{
      color: var(--ink2);
      font-size: 12.5px;
    }}
    .list {{
      margin: 8px 0 0 20px;
      color: var(--ink);
    }}
    .pill {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      background: #102944;
      border: 1px solid #2f4667;
      margin-right: 8px;
      margin-bottom: 8px;
      font-size: 12px;
      color: #d4e9ff;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{focus.name} Structured Report</h1>
    <div class="sub">Primary model comparison report against Trend Aware (original) and benchmark suite. Generated {timestamp}. Values with <span class="mono">*</span> are best in-table.</div>

    <div class="section">
      <h2>1. Executive Summary</h2>
      <div class="small">High-level performance snapshot for the focus primary strategy with direct peer comparison.</div>
      <div class="grid" style="margin-top:10px;">
        <div class="tile"><div class="k">Annualized Return</div><div class="v">{pct(float(focus_row["Annualized Return"]))}</div></div>
        <div class="tile"><div class="k">Volatility</div><div class="v">{pct(float(focus_row["Volatility"]))}</div></div>
        <div class="tile"><div class="k">Sharpe Ratio</div><div class="v">{num(float(focus_row["Sharpe Ratio"]),3)}</div></div>
        <div class="tile"><div class="k">Max Drawdown</div><div class="v">{pct(float(focus_row["Max Drawdown"]))}</div></div>
        <div class="tile"><div class="k">Win Rate</div><div class="v">{pct(float(focus_row["Win Rate"]))}</div></div>
        <div class="tile"><div class="k">Cumulative Return</div><div class="v">{pct(float(focus_row["Cumulative Return"]))}</div></div>
        <div class="tile"><div class="k">Information Ratio</div><div class="v">{num(float(focus_row["Information Ratio"]),3)}</div></div>
        <div class="tile"><div class="k">Strategy-SPX Correlation</div><div class="v">{num(float(focus_row["Strategy-SPX Correlation"]),3)}</div></div>
        <div class="tile"><div class="k">Indicator Pair Correlation</div><div class="v">{num(float(focus_row["Indicator Pair Correlation"]),3)}</div></div>
        <div class="tile"><div class="k">Precision / Recall / F1</div><div class="v">{pct(float(focus_row["Precision"]))} / {pct(float(focus_row["Recall"]))} / {pct(float(focus_row["F1 Score"]))}</div></div>
      </div>
      <div class="imgs">
        <div class="imgbox"><img src="{(assets_dir / 'executive_metric_bars.png').relative_to(REPORTS)}" alt="Executive metric bars"></div>
        <div class="imgbox"><img src="{(assets_dir / 'equity_curves.png').relative_to(REPORTS)}" alt="Executive equity curves"></div>
      </div>
      {to_html_table(pair_display)}
    </div>

    <div class="section">
      <h2>2. Benchmarks Section</h2>
      <div class="small">Four reference baselines and why each is relevant.</div>
      <div style="margin-top:10px;">
"""

    for name, reason in benchmark_explanations:
        html += f'<div class="pill"><b>{name}</b></div><div class="small" style="margin:2px 0 10px 0;">{reason}</div>'

    html += f"""
      </div>
      <div class="small">Benchmark metrics are computed over the same date range as the focus strategy to keep the comparison fair.</div>
      {to_html_table(bench_display)}
    </div>

    <div class="section">
      <h2>3. Indicators Research</h2>
      <div class="small">High-level review of three candidate primary strategies, indicator rationale, and relative performance ranking.</div>
      <div style="margin-top:10px;">
        <h3>{focus.name}</h3>
        <div class="small"><b>Concept:</b> {focus.concept}</div>
        <div class="small"><b>Why it makes sense:</b> {focus.rationale}</div>
        <div class="small"><b>Indicator choice:</b> {focus.indicator_choice}</div>
        <h3 style="margin-top:10px;">{peer.name}</h3>
        <div class="small"><b>Concept:</b> {peer.concept}</div>
        <div class="small"><b>Why it makes sense:</b> {peer.rationale}</div>
        <div class="small"><b>Indicator choice:</b> {peer.indicator_choice}</div>
        <h3 style="margin-top:10px;">{alt3.name}</h3>
        <div class="small"><b>Concept:</b> {alt3.concept}</div>
        <div class="small"><b>Why it makes sense:</b> {alt3.rationale}</div>
        <div class="small"><b>Indicator choice:</b> {alt3.indicator_choice}</div>
      </div>
      <div class="small" style="margin-top:10px;"><b>Current outperformer on the defined research score:</b> {winner_name}</div>
      {to_html_table(alt_display)}
    </div>

    <div class="section">
      <h2>4. Primary Strategy Deep Dive</h2>
      <h3>{deep_dive_title}</h3>
      <ul class="list">
"""
    for bullet in deep_dive_bullets:
        html += f"<li>{bullet}</li>"

    html += f"""
      </ul>
      <div class="small" style="margin-top:8px;">
        Wave view is used to explain the Herbert/Hilbert-style intuition: cyclical indicators move in phases, and the model interprets their relative phase alignment as risk-on/risk-off context.
      </div>
      <div class="imgs">
        <div class="imgbox"><img src="{(assets_dir / 'wave_visualization.png').relative_to(REPORTS)}" alt="Wave visualization"></div>
        <div class="imgbox"><img src="{(assets_dir / 'indicator_corr_heatmap.png').relative_to(REPORTS)}" alt="Indicator correlation heatmap"></div>
      </div>
    </div>

    <div class="section">
      <h2>5. Performance Visualizations</h2>
      <div class="small">Reused visual language from the existing M1 results structure: equity, drawdown, rolling Sharpe, plus explicit metric tables.</div>
      <div class="imgs">
        <div class="imgbox"><img src="{(assets_dir / 'equity_curves.png').relative_to(REPORTS)}" alt="Equity curves"></div>
        <div class="imgbox"><img src="{(assets_dir / 'drawdown_curves.png').relative_to(REPORTS)}" alt="Drawdown curves"></div>
        <div class="imgbox"><img src="{(assets_dir / 'rolling_sharpe.png').relative_to(REPORTS)}" alt="Rolling Sharpe"></div>
        <div class="imgbox"><img src="{(assets_dir / 'executive_metric_bars.png').relative_to(REPORTS)}" alt="Metric bars"></div>
      </div>
      <div class="small" style="margin-top:10px;"><b>Side-by-side:</b> Primary strategy vs 4 benchmarks (all required metrics included).</div>
      {to_html_table(bench_display)}
    </div>

    <div class="section">
      <h2>6. Conclusion</h2>
      <ul class="list">
        <li><b>Backtest summary:</b> {focus.name} produced {pct(float(focus_row["Annualized Return"]))} annualized return with Sharpe {num(float(focus_row["Sharpe Ratio"]),3)} and max drawdown {pct(float(focus_row["Max Drawdown"]))}.</li>
        <li><b>M2 implications:</b> This primary layer can be used as an opportunity generator; M2 can focus on confidence filtering/sizing and turnover-cost control.</li>
        <li><b>Key takeaway:</b> Use the benchmark comparison and classification metrics together. A strong primary model should not only earn returns, but also preserve directional recall quality.</li>
      </ul>
    </div>

    <div class="section">
      <h2>7. Deliverables</h2>
      <div class="small">Complete HTML report and all support files exported separately.</div>
      <ul class="list">
        <li>HTML report: <span class="mono">{report_html.relative_to(ROOT)}</span></li>
        <li>CSV tables: <span class="mono">{(assets_dir / 'executive_summary_metrics.csv').relative_to(ROOT)}</span>, <span class="mono">{(assets_dir / 'focus_vs_peer_metrics.csv').relative_to(ROOT)}</span>, <span class="mono">{(assets_dir / 'benchmarks_comparison.csv').relative_to(ROOT)}</span>, <span class="mono">{(assets_dir / 'alternative_models_research.csv').relative_to(ROOT)}</span>, <span class="mono">{(assets_dir / 'focus_indicator_correlation_matrix.csv').relative_to(ROOT)}</span></li>
        <li>Figures: <span class="mono">{(assets_dir / 'equity_curves.png').relative_to(ROOT)}</span>, <span class="mono">{(assets_dir / 'drawdown_curves.png').relative_to(ROOT)}</span>, <span class="mono">{(assets_dir / 'rolling_sharpe.png').relative_to(ROOT)}</span>, <span class="mono">{(assets_dir / 'executive_metric_bars.png').relative_to(ROOT)}</span>, <span class="mono">{(assets_dir / 'wave_visualization.png').relative_to(ROOT)}</span>, <span class="mono">{(assets_dir / 'indicator_corr_heatmap.png').relative_to(ROOT)}</span></li>
      </ul>
    </div>
  </div>
</body>
</html>
"""

    report_html.write_text(html, encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    prices = load_prices()
    rets = prices.pct_change()

    primary = build_primary_jq_bundle(prices, rets)
    trend = build_trend_bundle(prices, rets)
    unified = build_primary_unified_bundle(prices, rets)
    benchmarks = build_realized_benchmarks(prices, rets)

    primary_html = REPORTS / "primary_jq_report.html"
    trend_html = REPORTS / "trend_aware_report.html"

    primary_assets = REPORTS / "primary_jq_report_assets"
    trend_assets = REPORTS / "trend_aware_report_assets"

    make_report(
        focus=primary,
        peer=trend,
        alt3=unified,
        benchmarks=benchmarks,
        report_html=primary_html,
        assets_dir=primary_assets,
    )
    make_report(
        focus=trend,
        peer=primary,
        alt3=unified,
        benchmarks=benchmarks,
        report_html=trend_html,
        assets_dir=trend_assets,
    )

    print(f"Saved: {primary_html}")
    print(f"Saved: {trend_html}")
    print(f"Saved assets: {primary_assets}")
    print(f"Saved assets: {trend_assets}")


if __name__ == "__main__":
    main()
