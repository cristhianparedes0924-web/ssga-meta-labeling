#!/usr/bin/env python3
"""Build a single-model HTML report for Primary Joubert+QuantAtoZ.

This generator updates:
- reports/primary_model_report.html
- reports/primary_model_report_assets/*

Data source for the focus model is:
- reports/primary_jq_signal.csv (from primary_model_joubert_quantatoz.py)
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

BENCHMARK_EXPLANATIONS = [
    {
        "Benchmark": "Buy & Hold SPX",
        "Why meaningful": "Represents passive equity beta and the default alternative to any timing model.",
        "Why include": "Checks whether the primary strategy adds value above a simple market exposure.",
    },
    {
        "Benchmark": "60/40 SPX-Treasury",
        "Why meaningful": "Institutional policy mix with built-in diversification between growth and duration.",
        "Why include": "Tests if dynamic timing improves on a standard portfolio allocation.",
    },
    {
        "Benchmark": "EqualWeight 25/25/25/25",
        "Why meaningful": "Neutral cross-asset baseline across equities, rates, commodities, and credit.",
        "Why include": "Separates true signal value from concentration in a single risk bucket.",
    },
    {
        "Benchmark": "Simple Trend (SMA10/12)",
        "Why meaningful": "Classic low-complexity trend-following rule.",
        "Why include": "Validates whether richer indicator engineering beats a transparent trend baseline.",
    },
]


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


def map_to_next_available_dates(source_index: Iterable[pd.Timestamp], ref_index: Iterable[pd.Timestamp]) -> pd.DatetimeIndex:
    src = pd.DatetimeIndex(source_index)
    ref = pd.DatetimeIndex(ref_index).sort_values().unique()
    ref_arr = ref.to_numpy()

    mapped = []
    for d in src:
        pos = ref_arr.searchsorted(d.to_datetime64(), side="right")
        mapped.append(pd.Timestamp(ref_arr[pos]) if pos < len(ref_arr) else pd.NaT)
    return pd.DatetimeIndex(mapped)


def compute_trend_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=prices.index)

    def phase(s: pd.Series) -> pd.Series:
        detrended = s - s.rolling(12).mean()
        arr = np.angle(hilbert(detrended.fillna(0.0)))
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

    ind_pair = np.nan if indicators is None else avg_abs_offdiag_corr(indicators)
    ind_spx = np.nan if indicators is None else avg_abs_indicator_to_spx_corr(indicators, spx_ref)

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


def build_primary_jq_bundle(prices: pd.DataFrame, rets: pd.DataFrame) -> ModelBundle:
    signal = pd.read_csv(REPORTS / "primary_jq_signal.csv", parse_dates=["Date"]).set_index("Date")
    signal = signal.sort_index()

    # primary_jq_signal rows are decision-date; shift to next available realized month.
    realized_idx = map_to_next_available_dates(signal.index, rets.index)
    returns = pd.Series(signal["net_return"].to_numpy(dtype=float), index=realized_idx, name="primary_jq_net")
    side = pd.Series(signal["side"].to_numpy(dtype=float), index=realized_idx, name="side")
    target = pd.Series(signal["Target_Up"].to_numpy(dtype=float), index=realized_idx, name="target")

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
            "Directional primary model combining multi-theme macro and market state factors "
            "with correlation-aware aggregation and walk-forward thresholding."
        ),
        rationale=(
            "The design emphasizes opportunity recall without ignoring quality controls on "
            "factor stability, significance, and out-of-sample behavior."
        ),
        indicator_choice=(
            "Indicators cover momentum, cross-asset spreads, rates, credit carry, volatility regime, "
            "and drawdown so the signal is not dependent on a single market regime."
        ),
    )


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
            "from rates, credit, equities, and commodities."
        ),
        rationale=(
            "If cycle phase and short-term momentum align, asset allocation can tilt "
            "toward assets with stronger next-period edge."
        ),
        indicator_choice=(
            "Indicators are Hilbert-style phase proxies plus short-horizon SPX momentum "
            "to capture regime direction and acceleration."
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
            "Compact primary signal that blends trend, cross-asset, credit, and risk-breadth "
            "into one composite directional score."
        ),
        rationale=(
            "Lower model complexity can improve robustness and simplify operational monitoring."
        ),
        indicator_choice=(
            "Indicators were selected to combine macro-sensitive and risk-sensitive information "
            "with minimal feature count."
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


def pct(v: float | int | None) -> str:
    if v is None or not np.isfinite(v):
        return "N/A"
    return f"{v * 100:.2f}%"


def num(v: float | int | None, d: int = 3) -> str:
    if v is None or not np.isfinite(v):
        return "N/A"
    return f"{v:.{d}f}"


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
        "Max Drawdown": True,
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
        for col, higher in higher_better.items():
            vals = pd.to_numeric(out[col], errors="coerce")
            if vals.notna().any():
                best = vals.max() if higher else vals.min()
                for idx in out.index:
                    v = vals.loc[idx]
                    star_mask[(idx, col)] = bool(np.isfinite(v) and np.isclose(v, best, rtol=1e-10, atol=1e-12))

    for col in pct_cols:
        vals = []
        for idx, v in out[col].items():
            f = float(v) if np.isfinite(float(v)) else np.nan
            text = pct(f) if np.isfinite(f) else "N/A"
            if include_stars and star_mask.get((idx, col), False):
                text = f"{text}*"
            vals.append(text)
        out[col] = vals

    for col in num_cols:
        vals = []
        for idx, v in out[col].items():
            f = float(v) if np.isfinite(float(v)) else np.nan
            text = num(f, 3) if np.isfinite(f) else "N/A"
            if include_stars and star_mask.get((idx, col), False):
                text = f"{text}*"
            vals.append(text)
        out[col] = vals

    out["Periods"] = out["Periods"].astype(int).astype(str)
    return out


def to_html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=False, border=0, classes="tbl")


def rolling_sharpe(returns: pd.Series, window: int = 12) -> pd.Series:
    r = returns.dropna().astype(float)
    if r.empty:
        return r
    return (r.rolling(window).mean() * 12.0) / (r.rolling(window).std(ddof=1) * np.sqrt(12.0))


def save_plot_executive_bars(out_path: Path, row: Dict[str, float | str]) -> None:
    labels = [
        "Annualized Return",
        "Volatility",
        "Sharpe Ratio",
        "Information Ratio",
        "Win Rate",
        "F1 Score",
    ]
    vals = [float(row.get(k, np.nan)) for k in labels]
    clean_vals = [v if np.isfinite(v) else 0.0 for v in vals]
    colors = ["#0ea5e9", "#f59e0b", "#22c55e", "#a78bfa", "#14b8a6", "#f97316"]

    plt.figure(figsize=(9.6, 4.8))
    x = np.arange(len(labels))
    bars = plt.bar(x, clean_vals, color=colors)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.title("Executive Metric Snapshot")
    plt.grid(axis="y", alpha=0.25)
    pct_like = {"Annualized Return", "Volatility", "Win Rate", "F1 Score"}
    for b, raw, label in zip(bars, vals, labels):
        if np.isfinite(raw):
            txt = pct(raw) if label in pct_like else num(raw, 3)
            plt.text(
                b.get_x() + b.get_width() / 2.0,
                b.get_height(),
                txt,
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def save_plot_executive_equity(out_path: Path, returns: pd.Series) -> None:
    r = returns.dropna().astype(float)
    if r.empty:
        return
    eq = (1.0 + r).cumprod()
    plt.figure(figsize=(9.6, 4.8))
    plt.plot(eq.index, eq.values, linewidth=2.4, color="#0ea5e9")
    plt.yscale("log")
    plt.title("Primary Strategy Equity Curve")
    plt.ylabel("Growth of $1")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def save_plot_equity(out_path: Path, focus_name: str, series_map: Dict[str, pd.Series]) -> None:
    plt.figure(figsize=(11.2, 6.2))
    for name, s in series_map.items():
        r = s.dropna().astype(float)
        if r.empty:
            continue
        eq = (1.0 + r).cumprod()
        lw = 2.9 if name == focus_name else 1.8
        alpha = 1.0 if name == focus_name else 0.85
        plt.plot(eq.index, eq.values, linewidth=lw, alpha=alpha, label=name)
    plt.yscale("log")
    plt.title("Equity Curves - Primary vs Benchmarks")
    plt.ylabel("Growth of $1")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def save_plot_drawdown(out_path: Path, focus_name: str, series_map: Dict[str, pd.Series]) -> None:
    plt.figure(figsize=(11.2, 6.2))
    for name, s in series_map.items():
        r = s.dropna().astype(float)
        if r.empty:
            continue
        eq = (1.0 + r).cumprod()
        dd = eq / eq.cummax() - 1.0
        lw = 2.9 if name == focus_name else 1.8
        alpha = 1.0 if name == focus_name else 0.85
        plt.plot(dd.index, dd.values, linewidth=lw, alpha=alpha, label=name)
    plt.title("Drawdown Curves - Primary vs Benchmarks")
    plt.ylabel("Drawdown")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def save_plot_rolling_sharpe(out_path: Path, focus_name: str, series_map: Dict[str, pd.Series]) -> None:
    plt.figure(figsize=(11.2, 6.2))
    for name, s in series_map.items():
        rs = rolling_sharpe(s, window=12).dropna()
        if rs.empty:
            continue
        lw = 2.9 if name == focus_name else 1.8
        alpha = 1.0 if name == focus_name else 0.85
        plt.plot(rs.index, rs.values, linewidth=lw, alpha=alpha, label=name)
    plt.axhline(0.0, color="#334155", linewidth=1.0, linestyle="--")
    plt.title("Rolling 12-Month Sharpe")
    plt.ylabel("Sharpe")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def save_plot_wave(out_path: Path, indicators: pd.DataFrame) -> None:
    cols = [c for c in indicators.columns if "phase" in c.lower()]
    if len(cols) < 2:
        cols = list(indicators.columns[:4])
    cols = cols[:4]

    data = indicators[cols].copy().dropna()
    if data.empty:
        data = indicators[cols].copy().fillna(0.0)
    if data.shape[0] > 120:
        data = data.iloc[-120:]

    for col in data.columns:
        s = data[col]
        sd = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        if sd > 1e-12:
            data[col] = (s - s.mean()) / sd
        data[col] = data[col].clip(-2.5, 2.5)

    x = np.arange(len(data))
    plt.figure(figsize=(11.2, 5.8))
    colors = ["#0284c7", "#22c55e", "#f59e0b", "#ef4444"]
    for i, col in enumerate(data.columns):
        y = data[col].to_numpy(dtype=float)
        plt.plot(x, y, linewidth=2.2, color=colors[i % len(colors)], label=col)
        plt.fill_between(x, 0, y, alpha=0.08, color=colors[i % len(colors)])
    plt.axhline(0.0, color="#334155", linestyle="--", linewidth=1.0)
    plt.title("Wave View of Herbert/Hilbert-Style Indicator Behavior")
    plt.xlabel("Recent Months")
    plt.ylabel("Normalized Wave Level")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def save_plot_corr_heatmap(out_path: Path, corr_df: pd.DataFrame) -> None:
    if corr_df.empty:
        return
    vals = corr_df.to_numpy(dtype=float)
    plt.figure(figsize=(8.8, 7.0))
    im = plt.imshow(vals, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr_df.index)), corr_df.index)
    plt.title("Indicator Correlation Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def model_score_for_research(row: pd.Series) -> float:
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


def build_report() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    assets_dir = REPORTS / "primary_model_report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    prices = load_prices()
    rets = prices.pct_change()

    focus = build_primary_jq_bundle(prices, rets)
    alt_trend = build_trend_bundle(prices, rets)
    alt_unified = build_primary_unified_bundle(prices, rets)
    benchmarks = build_realized_benchmarks(prices, rets)

    focus_row = compute_metric_row(
        label=focus.name,
        returns=focus.returns,
        side=focus.side,
        target=focus.target,
        spx_ref=focus.spx_ref,
        indicators=focus.indicators,
    )

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

    alt_rows = [
        focus_row,
        compute_metric_row(
            label=alt_trend.name,
            returns=alt_trend.returns,
            side=alt_trend.side,
            target=alt_trend.target,
            spx_ref=alt_trend.spx_ref,
            indicators=alt_trend.indicators,
        ),
        compute_metric_row(
            label=alt_unified.name,
            returns=alt_unified.returns,
            side=alt_unified.side,
            target=alt_unified.target,
            spx_ref=alt_unified.spx_ref,
            indicators=alt_unified.indicators,
        ),
    ]
    alt_df = pd.DataFrame(alt_rows)
    alt_df["Research Score"] = alt_df.apply(model_score_for_research, axis=1)
    alt_df = alt_df.sort_values("Research Score", ascending=False).reset_index(drop=True)
    winner_name = str(alt_df.iloc[0]["Model"])

    focus_indicator_corr = focus.indicators.dropna().corr()

    # Export supporting tables.
    pd.DataFrame([focus_row]).to_csv(assets_dir / "executive_summary_metrics.csv", index=False)
    bench_df.to_csv(assets_dir / "benchmarks_comparison.csv", index=False)
    bench_df.to_csv(assets_dir / "side_by_side_primary_vs_benchmarks.csv", index=False)
    alt_df.to_csv(assets_dir / "alternative_models_research.csv", index=False)
    focus_indicator_corr.to_csv(assets_dir / "focus_indicator_correlation_matrix.csv")
    pd.DataFrame(BENCHMARK_EXPLANATIONS).to_csv(assets_dir / "benchmark_explanations.csv", index=False)

    # Export supporting figures.
    save_plot_executive_bars(assets_dir / "executive_metric_bars.png", focus_row)
    save_plot_executive_equity(assets_dir / "executive_equity_curve.png", focus.returns)
    save_plot_wave(assets_dir / "wave_visualization.png", focus.indicators)
    save_plot_corr_heatmap(assets_dir / "indicator_corr_heatmap.png", focus_indicator_corr)

    plot_series = {focus.name: focus.returns}
    for bname, bundle in benchmarks.items():
        plot_series[bname] = bundle["returns"].reindex(focus_idx)
    save_plot_equity(assets_dir / "equity_curves.png", focus.name, plot_series)
    save_plot_drawdown(assets_dir / "drawdown_curves.png", focus.name, plot_series)
    save_plot_rolling_sharpe(assets_dir / "rolling_sharpe.png", focus.name, plot_series)

    exec_display = format_metrics_table(pd.DataFrame([focus_row]), include_stars=False)
    bench_display = format_metrics_table(bench_df, include_stars=True)
    alt_display = format_metrics_table(alt_df.drop(columns=["Research Score"]), include_stars=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    report_html = REPORTS / "primary_model_report.html"

    deep_dive_bullets = [
        "Input layer: monthly SPX, commodities, rates, and credit data aligned to a consistent timestamp grid.",
        "Feature layer: momentum, cross-asset spread, rate change, credit carry, volatility regime, and drawdown indicators.",
        "Stability layer: orientation-aware normalization and outlier control make factors comparable through time.",
        "Combination layer: correlation-aware weighting reduces double-counting of similar indicators.",
        "Decision layer: probability thresholding with hysteresis converts factor evidence into a practical side signal.",
    ]

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Primary Joubert+QuantAtoZ - Primary Model Report</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0b1220;
      --bg2: #111a2d;
      --panel: #142039;
      --panel2: #1d2b46;
      --line: #2a3b5d;
      --ink: #e6edf7;
      --ink2: #9fb3cf;
      --cyan: #0ea5e9;
      --teal: #14b8a6;
      --green: #22c55e;
      --orange: #f59e0b;
      --red: #ef4444;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at 15% -10%, #17335e 0%, transparent 45%),
        radial-gradient(circle at 90% 0%, #134e4a 0%, transparent 30%),
        var(--bg);
      color: var(--ink);
      font-family: "DM Sans", sans-serif;
      line-height: 1.45;
    }}
    .wrap {{ max-width: 1440px; margin: 0 auto; padding: 24px; }}
    h1, h2, h3 {{ margin: 0 0 10px; letter-spacing: 0.01em; }}
    h1 {{ font-family: "DM Serif Display", serif; font-size: 42px; line-height: 1.08; }}
    h2 {{ font-size: 28px; }}
    h3 {{ font-size: 19px; }}
    .sub {{ color: var(--ink2); margin-bottom: 18px; }}
    .section {{
      background: linear-gradient(180deg, rgba(29,43,70,0.94) 0%, rgba(20,32,57,0.94) 100%);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      margin-bottom: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
      gap: 10px;
      margin-top: 10px;
    }}
    .tile {{
      background: rgba(11,18,32,0.72);
      border: 1px solid #334969;
      border-radius: 12px;
      padding: 10px;
    }}
    .tile .k {{ font-size: 12px; color: var(--ink2); margin-bottom: 5px; }}
    .tile .v {{ font-size: 18px; font-weight: 700; color: #f3faff; font-family: "JetBrains Mono", monospace; }}
    .imgs {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 12px;
      margin-top: 10px;
    }}
    .imgbox {{
      background: #0f172a;
      border: 1px solid #334969;
      border-radius: 12px;
      padding: 8px;
    }}
    .imgbox img {{ width: 100%; border-radius: 8px; display: block; }}
    .tbl {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12.5px;
      margin-top: 8px;
    }}
    .tbl th, .tbl td {{
      border: 1px solid #2f4467;
      padding: 6px 8px;
      text-align: right;
      white-space: nowrap;
    }}
    .tbl th:first-child, .tbl td:first-child {{ text-align: left; }}
    .tbl th {{ background: #14213a; color: #d5e7ff; }}
    .small {{ color: var(--ink2); font-size: 12.5px; }}
    .list {{ margin: 8px 0 0 20px; color: var(--ink); }}
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
    .bench-cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 10px;
      margin-top: 10px;
    }}
    .bench-card {{
      background: rgba(11,18,32,0.68);
      border: 1px solid #334969;
      border-radius: 10px;
      padding: 10px;
    }}
    .dl-list {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 8px 20px;
      margin-top: 8px;
      font-size: 12.5px;
    }}
    .mono {{ font-family: "JetBrains Mono", monospace; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Primary Joubert+QuantAtoZ</h1>
    <div class="sub">Single-model report generated from <span class="mono">primary_model_joubert_quantatoz.py</span> outputs only. Generated {timestamp}.</div>

    <div class="section">
      <h2>1. Executive Summary</h2>
      <div class="small">High-level snapshot of the primary strategy, with explicit required metrics.</div>
      <div class="grid">
        <div class="tile"><div class="k">Annualized Return</div><div class="v">{pct(float(focus_row["Annualized Return"]))}</div></div>
        <div class="tile"><div class="k">Volatility</div><div class="v">{pct(float(focus_row["Volatility"]))}</div></div>
        <div class="tile"><div class="k">Sharpe Ratio</div><div class="v">{num(float(focus_row["Sharpe Ratio"]),3)}</div></div>
        <div class="tile"><div class="k">Max Drawdown</div><div class="v">{pct(float(focus_row["Max Drawdown"]))}</div></div>
        <div class="tile"><div class="k">Win Rate</div><div class="v">{pct(float(focus_row["Win Rate"]))}</div></div>
        <div class="tile"><div class="k">Cumulative Return</div><div class="v">{pct(float(focus_row["Cumulative Return"]))}</div></div>
        <div class="tile"><div class="k">Information Ratio</div><div class="v">{num(float(focus_row["Information Ratio"]),3)}</div></div>
        <div class="tile"><div class="k">Strategy-SPX Correlation</div><div class="v">{num(float(focus_row["Strategy-SPX Correlation"]),3)}</div></div>
        <div class="tile"><div class="k">Indicator Pair Correlation</div><div class="v">{num(float(focus_row["Indicator Pair Correlation"]),3)}</div></div>
        <div class="tile"><div class="k">Indicator-to-SPX Correlation</div><div class="v">{num(float(focus_row["Indicator-to-SPX Correlation"]),3)}</div></div>
        <div class="tile"><div class="k">Precision</div><div class="v">{pct(float(focus_row["Precision"]))}</div></div>
        <div class="tile"><div class="k">Recall</div><div class="v">{pct(float(focus_row["Recall"]))}</div></div>
        <div class="tile"><div class="k">F1 Score</div><div class="v">{pct(float(focus_row["F1 Score"]))}</div></div>
      </div>
      <div class="imgs">
        <div class="imgbox"><img src="{(assets_dir / 'executive_metric_bars.png').relative_to(REPORTS)}" alt="Executive metric bars"></div>
        <div class="imgbox"><img src="{(assets_dir / 'executive_equity_curve.png').relative_to(REPORTS)}" alt="Executive equity curve"></div>
      </div>
      {to_html_table(exec_display)}
    </div>

    <div class="section">
      <h2>2. Benchmarks Section</h2>
      <div class="small">Four benchmarks, each with relevance and inclusion rationale.</div>
      <div class="bench-cards">
"""

    for row in BENCHMARK_EXPLANATIONS:
        html += f"""
        <div class="bench-card">
          <div class="pill"><b>{row["Benchmark"]}</b></div>
          <div class="small"><b>Why meaningful:</b> {row["Why meaningful"]}</div>
          <div class="small" style="margin-top:6px;"><b>Why include:</b> {row["Why include"]}</div>
        </div>
"""

    html += f"""
      </div>
      <div class="small" style="margin-top:10px;">Benchmark metrics are aligned to the primary strategy date range for fair side-by-side evaluation.</div>
      {to_html_table(bench_display)}
    </div>

    <div class="section">
      <h2>3. Indicators Research</h2>
      <div class="small">Three candidate strategies reviewed at conceptual and analytical level.</div>
      <div style="margin-top:10px;">
        <h3>{focus.name}</h3>
        <div class="small"><b>Concept:</b> {focus.concept}</div>
        <div class="small"><b>Why it can make sense:</b> {focus.rationale}</div>
        <div class="small"><b>Indicator selection:</b> {focus.indicator_choice}</div>

        <h3 style="margin-top:10px;">{alt_trend.name}</h3>
        <div class="small"><b>Concept:</b> {alt_trend.concept}</div>
        <div class="small"><b>Why it can make sense:</b> {alt_trend.rationale}</div>
        <div class="small"><b>Indicator selection:</b> {alt_trend.indicator_choice}</div>

        <h3 style="margin-top:10px;">{alt_unified.name}</h3>
        <div class="small"><b>Concept:</b> {alt_unified.concept}</div>
        <div class="small"><b>Why it can make sense:</b> {alt_unified.rationale}</div>
        <div class="small"><b>Indicator selection:</b> {alt_unified.indicator_choice}</div>
      </div>
      <div class="small" style="margin-top:10px;"><b>Outperformer on the defined research score:</b> {winner_name}</div>
      {to_html_table(alt_display)}
    </div>

    <div class="section">
      <h2>4. Primary Strategy Deep Dive</h2>
      <h3>What Goes Into Primary Joubert+QuantAtoZ</h3>
      <ul class="list">
"""

    for bullet in deep_dive_bullets:
        html += f"<li>{bullet}</li>"

    html += f"""
      </ul>
      <div class="small" style="margin-top:8px;">
        Intuition: in the wave view, indicators move in and out of phase. Consistent phase alignment usually signals stronger regime conviction, while mixed phase states suggest lower conviction.
      </div>
      <div class="imgs">
        <div class="imgbox"><img src="{(assets_dir / 'wave_visualization.png').relative_to(REPORTS)}" alt="Wave visualization"></div>
        <div class="imgbox"><img src="{(assets_dir / 'indicator_corr_heatmap.png').relative_to(REPORTS)}" alt="Indicator correlation heatmap"></div>
      </div>
    </div>

    <div class="section">
      <h2>5. Performance Visualizations</h2>
      <div class="small">Visual diagnostics and explicit metric table for Primary vs the 4 benchmarks.</div>
      <div class="imgs">
        <div class="imgbox"><img src="{(assets_dir / 'equity_curves.png').relative_to(REPORTS)}" alt="Equity curves"></div>
        <div class="imgbox"><img src="{(assets_dir / 'drawdown_curves.png').relative_to(REPORTS)}" alt="Drawdown curves"></div>
        <div class="imgbox"><img src="{(assets_dir / 'rolling_sharpe.png').relative_to(REPORTS)}" alt="Rolling Sharpe"></div>
        <div class="imgbox"><img src="{(assets_dir / 'executive_metric_bars.png').relative_to(REPORTS)}" alt="Metric bars"></div>
      </div>
      <div class="small" style="margin-top:10px;"><b>Side-by-side comparison:</b> Primary strategy vs the 4 benchmarks.</div>
      {to_html_table(bench_display)}
    </div>

    <div class="section">
      <h2>6. Conclusion</h2>
      <ul class="list">
        <li><b>Backtest summary:</b> {focus.name} delivered {pct(float(focus_row["Annualized Return"]))} annualized return, Sharpe {num(float(focus_row["Sharpe Ratio"]),3)}, and max drawdown {pct(float(focus_row["Max Drawdown"]))}.</li>
        <li><b>Implications for M2:</b> Use this primary layer as an opportunity generator; M2 should focus on confidence filtering, sizing, and cost-aware execution.</li>
        <li><b>Key takeaway:</b> Evaluate return metrics together with precision/recall/F1 and correlation structure, not in isolation.</li>
      </ul>
    </div>

    <div class="section">
      <h2>7. Deliverables</h2>
      <div class="small">Complete HTML plus all supporting tables and figures exported separately.</div>
      <div class="dl-list">
        <div>HTML: <span class="mono">{report_html.relative_to(ROOT)}</span></div>
        <div>CSV: <span class="mono">{(assets_dir / 'executive_summary_metrics.csv').relative_to(ROOT)}</span></div>
        <div>CSV: <span class="mono">{(assets_dir / 'benchmarks_comparison.csv').relative_to(ROOT)}</span></div>
        <div>CSV: <span class="mono">{(assets_dir / 'side_by_side_primary_vs_benchmarks.csv').relative_to(ROOT)}</span></div>
        <div>CSV: <span class="mono">{(assets_dir / 'alternative_models_research.csv').relative_to(ROOT)}</span></div>
        <div>CSV: <span class="mono">{(assets_dir / 'focus_indicator_correlation_matrix.csv').relative_to(ROOT)}</span></div>
        <div>CSV: <span class="mono">{(assets_dir / 'benchmark_explanations.csv').relative_to(ROOT)}</span></div>
        <div>PNG: <span class="mono">{(assets_dir / 'executive_metric_bars.png').relative_to(ROOT)}</span></div>
        <div>PNG: <span class="mono">{(assets_dir / 'executive_equity_curve.png').relative_to(ROOT)}</span></div>
        <div>PNG: <span class="mono">{(assets_dir / 'equity_curves.png').relative_to(ROOT)}</span></div>
        <div>PNG: <span class="mono">{(assets_dir / 'drawdown_curves.png').relative_to(ROOT)}</span></div>
        <div>PNG: <span class="mono">{(assets_dir / 'rolling_sharpe.png').relative_to(ROOT)}</span></div>
        <div>PNG: <span class="mono">{(assets_dir / 'wave_visualization.png').relative_to(ROOT)}</span></div>
        <div>PNG: <span class="mono">{(assets_dir / 'indicator_corr_heatmap.png').relative_to(ROOT)}</span></div>
      </div>
    </div>
  </div>
</body>
</html>
"""

    report_html.write_text(html, encoding="utf-8")
    print(f"Saved: {report_html}")
    print(f"Saved assets: {assets_dir}")


if __name__ == "__main__":
    build_report()
