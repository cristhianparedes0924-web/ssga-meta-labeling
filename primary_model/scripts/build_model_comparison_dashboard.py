#!/usr/bin/env python3
"""Build a consolidated HTML dashboard comparing all primary-model variants."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
ASSETS_DIR = REPORTS_DIR / "assets"

RAW_FILES = {
    "spx": ROOT / "data" / "raw" / "spx.xlsx",
    "rates": ROOT / "data" / "raw" / "treasury_10y.xlsx",
    "commodities": ROOT / "data" / "raw" / "bcom.xlsx",
    "credit": ROOT / "data" / "raw" / "corp_bonds.xlsx",
}


def load_raw_prices(file_map: dict[str, Path]) -> pd.DataFrame:
    prices = pd.DataFrame()

    for name, filepath in file_map.items():
        raw = pd.read_excel(filepath, header=None, dtype=object)

        header_row = -1
        for i, row in raw.iterrows():
            values = [str(v).strip().lower() for v in row.tolist() if pd.notna(v)]
            if "date" in values and ("px_last" in values or "close" in values):
                header_row = int(i)
                break

        if header_row == -1:
            raise ValueError(f"Header not found in {filepath}")

        header = raw.iloc[header_row].tolist()
        df = raw.iloc[header_row + 1 :].copy()
        df.columns = header
        df = df.loc[:, [col for col in df.columns if pd.notna(col)]]
        df.columns = [str(c).strip() for c in df.columns]

        date_col = next((c for c in df.columns if c.lower() == "date"), None)
        close_col = next((c for c in df.columns if c.upper() == "PX_LAST" or "close" in c.lower()), None)
        if date_col is None or close_col is None:
            raise ValueError(f"Required columns missing in {filepath}")

        series = (
            df[[date_col, close_col]]
            .rename(columns={date_col: "Date", close_col: name})
            .assign(Date=lambda x: pd.to_datetime(x["Date"], errors="coerce"))
            .assign(**{name: lambda x: pd.to_numeric(x[name], errors="coerce")})
            .dropna(subset=["Date"])
            .drop_duplicates(subset=["Date"], keep="last")
            .set_index("Date")
            .sort_index()
        )

        if prices.empty:
            prices = series
        else:
            prices = prices.join(series, how="outer")

    return prices.sort_index().ffill().dropna()


def compute_metrics(returns: pd.Series) -> dict[str, float | int | str]:
    rets = returns.dropna().astype(float)
    if rets.empty:
        return {
            "periods": 0,
            "start": "",
            "end": "",
            "cumulative_return": np.nan,
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
            "win_rate": np.nan,
            "best_month": np.nan,
            "worst_month": np.nan,
        }

    equity = (1 + rets).cumprod()
    periods = len(rets)
    ann_return = float((equity.iloc[-1]) ** (12 / periods) - 1)
    ann_vol = float(rets.std(ddof=1) * np.sqrt(12))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else np.nan
    drawdown = equity / equity.cummax() - 1
    max_drawdown = float(drawdown.min())
    calmar = float(ann_return / abs(max_drawdown)) if max_drawdown < 0 else np.nan

    return {
        "periods": periods,
        "start": rets.index.min().strftime("%Y-%m-%d"),
        "end": rets.index.max().strftime("%Y-%m-%d"),
        "cumulative_return": float(equity.iloc[-1] - 1),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": float((rets > 0).mean()),
        "best_month": float(rets.max()),
        "worst_month": float(rets.min()),
    }


def rolling_sharpe(returns: pd.Series, window: int = 12) -> pd.Series:
    rets = returns.dropna().astype(float)
    if rets.empty:
        return rets
    return (rets.rolling(window).mean() * 12) / (rets.rolling(window).std(ddof=1) * np.sqrt(12))


def pct(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:.2f}%"


def num(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x:.{digits}f}"


def format_value(value: object, fmt: str) -> str:
    if fmt == "pct":
        return pct(float(value))
    if fmt == "num":
        return num(float(value))
    if fmt == "int":
        if pd.isna(value):
            return "N/A"
        return str(int(value))
    if fmt == "text":
        return str(value)
    return str(value)


def format_table_with_stars(
    df: pd.DataFrame,
    schema: dict[str, dict[str, str | None]],
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    for col in df.columns:
        meta = schema.get(col, {"fmt": "text", "best": None})
        fmt = str(meta.get("fmt", "text"))
        best_rule = meta.get("best")

        numeric_col = pd.to_numeric(df[col], errors="coerce")
        star_mask = pd.Series(False, index=df.index)

        if best_rule in {"max", "min"} and numeric_col.notna().any():
            best_val = numeric_col.max() if best_rule == "max" else numeric_col.min()
            star_mask = numeric_col.notna() & np.isclose(numeric_col, best_val, rtol=1e-10, atol=1e-12)

        display_vals: list[str] = []
        for idx, val in df[col].items():
            if fmt in {"pct", "num", "int"}:
                if pd.isna(val):
                    text = "N/A"
                else:
                    text = format_value(val, fmt)
            else:
                text = format_value(val, fmt)

            if star_mask.loc[idx]:
                text = f"{text}*"
            display_vals.append(text)

        out[col] = display_vals

    return out


def compute_strategy_returns_from_weights(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    transaction_cost_bps: float = 0.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    common_idx = weights.index.intersection(returns.index)
    weights = weights.loc[common_idx].sort_index()
    returns = returns.loc[common_idx, weights.columns].sort_index()

    gross = (weights.shift(1) * returns).sum(axis=1).dropna()
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    cost_drag = turnover.shift(1).fillna(0.0) * (transaction_cost_bps / 10000.0)
    cost_drag = cost_drag.loc[gross.index]
    net = gross - cost_drag

    return gross, net, turnover.loc[gross.index], cost_drag


def build_primary_unified_bundle() -> dict[str, object]:
    backtest = pd.read_csv(REPORTS_DIR / "primary_v1_backtest.csv", parse_dates=["Date"]).set_index("Date")
    signals = pd.read_csv(REPORTS_DIR / "primary_v1_signal.csv", parse_dates=["Date"])
    weights = pd.read_csv(REPORTS_DIR / "primary_v1_weights.csv", parse_dates=["Date"]).set_index("Date")
    benchmarks = pd.read_csv(REPORTS_DIR / "benchmarks_summary.csv").rename(columns={"Unnamed: 0": "strategy"})

    strategy_returns = backtest["net_return"].astype(float)
    metrics = compute_metrics(strategy_returns)

    return {
        "name": "Primary Unified",
        "returns": strategy_returns,
        "metrics": metrics,
        "avg_turnover": float(backtest["turnover"].fillna(0.0).mean()),
        "time_in_market": 1.0,
        "signal_counts": signals["signal"].value_counts(dropna=True).to_dict(),
        "weights_avg": weights.mean(numeric_only=True).to_dict(),
        "benchmark_table": benchmarks,
    }


def build_trend_aware_bundle(prices: pd.DataFrame, returns: pd.DataFrame) -> dict[str, object]:
    weights = pd.read_csv(ROOT / "monthly_allocations.csv", parse_dates=["Date"]).set_index("Date")
    weights = weights[["spx", "rates", "commodities", "credit"]].astype(float)

    gross, net, turnover, _ = compute_strategy_returns_from_weights(weights, returns, transaction_cost_bps=0.0)
    spx_returns = returns.loc[net.index, "spx"]
    ew_returns = returns.loc[net.index, ["spx", "rates", "commodities", "credit"]].mean(axis=1)

    benchmark_table = pd.DataFrame(
        [
            {"strategy": "TrendAware", **compute_metrics(net)},
            {"strategy": "SPX BuyHold", **compute_metrics(spx_returns)},
            {"strategy": "EqualWeight 4-Asset", **compute_metrics(ew_returns)},
        ]
    )

    return {
        "name": "Trend Aware",
        "returns": net,
        "metrics": compute_metrics(net),
        "avg_turnover": float(turnover.mean()),
        "time_in_market": 1.0,
        "signal_counts": {"BUY": np.nan, "HOLD": np.nan, "SELL": np.nan},
        "weights_avg": weights.mean(numeric_only=True).to_dict(),
        "benchmark_table": benchmark_table,
    }


def build_trend_aware_enhanced_bundle(prices: pd.DataFrame, returns: pd.DataFrame) -> dict[str, object]:
    weights = pd.read_csv(ROOT / "monthly_allocations_enhanced.csv", parse_dates=["Date"]).set_index("Date")
    weights = weights[["spx", "rates", "commodities", "credit"]].astype(float)

    enhanced_rets = pd.read_csv(ROOT / "strategy_returns_enhanced.csv", parse_dates=["Date"]).set_index("Date")
    net_returns = enhanced_rets["net_return"].astype(float)

    spx_returns = returns.loc[net_returns.index, "spx"]
    ew_returns = returns.loc[net_returns.index, ["spx", "rates", "commodities", "credit"]].mean(axis=1)

    benchmark_table = pd.DataFrame(
        [
            {"strategy": "TrendAware Enhanced (Net)", **compute_metrics(net_returns)},
            {"strategy": "SPX BuyHold", **compute_metrics(spx_returns)},
            {"strategy": "EqualWeight 4-Asset", **compute_metrics(ew_returns)},
        ]
    )

    return {
        "name": "Trend Aware Enhanced",
        "returns": net_returns,
        "metrics": compute_metrics(net_returns),
        "avg_turnover": float(enhanced_rets["turnover"].mean()),
        "time_in_market": 1.0,
        "signal_counts": {"BUY": np.nan, "HOLD": np.nan, "SELL": np.nan},
        "weights_avg": weights.mean(numeric_only=True).to_dict(),
        "benchmark_table": benchmark_table,
    }


def build_sam_benchmark_bundle() -> dict[str, object]:
    df = pd.read_csv(REPORTS_DIR / "backtest_results.csv", parse_dates=["Date"]).set_index("Date")
    mask = df["Forward_Return"].notna()

    strategy_returns = df.loc[mask, "Strategy_Return"].astype(float)
    bh_returns = df.loc[mask, "BH_Return"].astype(float)
    bench_6040 = df.loc[mask, "Bench_6040_Return"].astype(float)
    bench_sma = df.loc[mask, "Bench_SMA_Return"].astype(float)
    bench_random = df.loc[mask, "Bench_Random_Return"].astype(float)

    test_mask = mask & (df["Set"] == "test")
    strategy_test_returns = df.loc[test_mask, "Strategy_Return"].astype(float)

    benchmark_table = pd.DataFrame(
        [
            {"strategy": "SamBenchmark_M1", **compute_metrics(strategy_returns)},
            {"strategy": "BuyHold SPX", **compute_metrics(bh_returns)},
            {"strategy": "60/40", **compute_metrics(bench_6040)},
            {"strategy": "SMA 10/12", **compute_metrics(bench_sma)},
            {"strategy": "Random Median", **compute_metrics(bench_random)},
        ]
    )

    return {
        "name": "Benchmark (Samwegs)",
        "returns": strategy_returns,
        "metrics": compute_metrics(strategy_returns),
        "metrics_test": compute_metrics(strategy_test_returns),
        "avg_turnover": float(df.loc[mask, "Position"].diff().abs().fillna(0.0).mean()),
        "time_in_market": float((df.loc[mask, "Position"] != 0).mean()),
        "signal_counts": df["Signal"].value_counts(dropna=True).to_dict(),
        "weights_avg": {},
        "benchmark_table": benchmark_table,
    }


def save_comparison_plots(returns_map: dict[str, pd.Series]) -> dict[str, Path]:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    equity_path = ASSETS_DIR / "model_compare_equity.png"
    dd_path = ASSETS_DIR / "model_compare_drawdown.png"
    sharpe_path = ASSETS_DIR / "model_compare_rolling_sharpe.png"

    plt.figure(figsize=(11, 6))
    for name, rets in returns_map.items():
        eq = (1 + rets.dropna()).cumprod()
        plt.plot(eq.index, eq.values, label=name, linewidth=2)
    plt.title("Equity Curves: Four Model Strategies")
    plt.ylabel("Growth of $1")
    plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(equity_path, dpi=150)
    plt.close()

    plt.figure(figsize=(11, 6))
    for name, rets in returns_map.items():
        eq = (1 + rets.dropna()).cumprod()
        dd = eq / eq.cummax() - 1
        plt.plot(dd.index, dd.values, label=name, linewidth=2)
    plt.title("Drawdowns: Four Model Strategies")
    plt.ylabel("Drawdown")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(dd_path, dpi=150)
    plt.close()

    plt.figure(figsize=(11, 6))
    for name, rets in returns_map.items():
        rs = rolling_sharpe(rets)
        plt.plot(rs.index, rs.values, label=name, linewidth=2)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.title("Rolling 12-Month Sharpe")
    plt.ylabel("Sharpe")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(sharpe_path, dpi=150)
    plt.close()

    return {"equity": equity_path, "drawdown": dd_path, "rolling_sharpe": sharpe_path}


def make_signal_table(model_bundles: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bundle in model_bundles:
        counts = bundle["signal_counts"]
        if any(pd.isna(v) for v in counts.values()):
            rows.append(
                {
                    "Model": bundle["name"],
                    "BUY": "N/A",
                    "HOLD": "N/A",
                    "SELL": "N/A",
                    "Coverage": "No explicit BUY/HOLD/SELL signals",
                }
            )
            continue

        total = int(sum(int(v) for v in counts.values()))
        rows.append(
            {
                "Model": bundle["name"],
                "BUY": int(counts.get("BUY", 0)),
                "HOLD": int(counts.get("HOLD", 0)),
                "SELL": int(counts.get("SELL", 0)),
                "Coverage": f"{total} labeled periods",
            }
        )

    return pd.DataFrame(rows)


def make_weight_table(model_bundles: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for bundle in model_bundles:
        avg_weights = bundle["weights_avg"]
        if not avg_weights:
            rows.append(
                {
                    "Model": bundle["name"],
                    "SPX": "N/A",
                    "Commodities": "N/A",
                    "Rates/Treasury": "N/A",
                    "Credit": "N/A",
                }
            )
            continue

        rows.append(
            {
                "Model": bundle["name"],
                "SPX": pct(float(avg_weights.get("spx", np.nan))),
                "Commodities": pct(float(avg_weights.get("commodities", avg_weights.get("bcom", np.nan)))),
                "Rates/Treasury": pct(float(avg_weights.get("rates", avg_weights.get("treasury_10y", np.nan)))),
                "Credit": pct(float(avg_weights.get("credit", avg_weights.get("corp_bonds", np.nan)))),
            }
        )

    return pd.DataFrame(rows)


def render_dashboard(model_bundles: list[dict[str, object]], plots: dict[str, Path], output_path: Path) -> None:
    core_rows: list[dict[str, object]] = []
    risk_rows: list[dict[str, object]] = []

    for bundle in model_bundles:
        m = bundle["metrics"]
        core_rows.append(
            {
                "Model": bundle["name"],
                "Periods": int(m["periods"]),
                "Start": m["start"],
                "End": m["end"],
                "Cumulative Return": float(m["cumulative_return"]),
                "Annual Return": float(m["ann_return"]),
                "Annual Volatility": float(m["ann_vol"]),
                "Sharpe": float(m["sharpe"]),
                "Max Drawdown": float(m["max_drawdown"]),
                "Calmar": float(m["calmar"]),
            }
        )

        risk_rows.append(
            {
                "Model": bundle["name"],
                "Win Rate": float(m["win_rate"]),
                "Best Month": float(m["best_month"]),
                "Worst Month": float(m["worst_month"]),
                "Avg Turnover (monthly)": float(bundle["avg_turnover"]),
                "Time in Market": float(bundle["time_in_market"]),
            }
        )

    core_numeric = pd.DataFrame(core_rows)
    risk_numeric = pd.DataFrame(risk_rows)

    core_schema = {
        "Model": {"fmt": "text", "best": None},
        "Periods": {"fmt": "int", "best": None},
        "Start": {"fmt": "text", "best": None},
        "End": {"fmt": "text", "best": None},
        "Cumulative Return": {"fmt": "pct", "best": "max"},
        "Annual Return": {"fmt": "pct", "best": "max"},
        "Annual Volatility": {"fmt": "pct", "best": "min"},
        "Sharpe": {"fmt": "num", "best": "max"},
        "Max Drawdown": {"fmt": "pct", "best": "max"},
        "Calmar": {"fmt": "num", "best": "max"},
    }

    risk_schema = {
        "Model": {"fmt": "text", "best": None},
        "Win Rate": {"fmt": "pct", "best": "max"},
        "Best Month": {"fmt": "pct", "best": "max"},
        "Worst Month": {"fmt": "pct", "best": "max"},
        "Avg Turnover (monthly)": {"fmt": "pct", "best": "min"},
        "Time in Market": {"fmt": "pct", "best": None},
    }

    core_df = format_table_with_stars(core_numeric, core_schema)
    risk_df = format_table_with_stars(risk_numeric, risk_schema)

    signal_df = make_signal_table(model_bundles)
    weights_df = make_weight_table(model_bundles)

    sam_bundle = next(bundle for bundle in model_bundles if bundle["name"] == "Benchmark (Samwegs)")
    sam_test = sam_bundle["metrics_test"]
    sam_test_df = pd.DataFrame(
        [
            {
                "Model": "Benchmark (Samwegs) - Test Only",
                "Periods": int(sam_test["periods"]),
                "Start": sam_test["start"],
                "End": sam_test["end"],
                "Cumulative Return": pct(float(sam_test["cumulative_return"])),
                "Annual Return": pct(float(sam_test["ann_return"])),
                "Annual Volatility": pct(float(sam_test["ann_vol"])),
                "Sharpe": num(float(sam_test["sharpe"])),
                "Max Drawdown": pct(float(sam_test["max_drawdown"])),
                "Calmar": num(float(sam_test["calmar"])),
            }
        ]
    )

    unified_bench = next(bundle for bundle in model_bundles if bundle["name"] == "Primary Unified")["benchmark_table"].copy()
    unified_bench = unified_bench.rename(columns={"strategy": "Strategy"})
    for col in ["ann_return", "ann_vol", "max_drawdown", "avg_turnover"]:
        unified_bench[col] = unified_bench[col].apply(pct)
    for col in ["sharpe", "calmar"]:
        unified_bench[col] = unified_bench[col].apply(num)

    trend_bench = next(bundle for bundle in model_bundles if bundle["name"] == "Trend Aware")["benchmark_table"].copy()
    trend_bench = trend_bench.rename(columns={"strategy": "Strategy"})
    for col in ["cumulative_return", "ann_return", "ann_vol", "max_drawdown", "win_rate"]:
        trend_bench[col] = trend_bench[col].apply(pct)
    for col in ["sharpe", "calmar"]:
        trend_bench[col] = trend_bench[col].apply(num)
    trend_bench = trend_bench[
        [
            "Strategy",
            "periods",
            "start",
            "end",
            "cumulative_return",
            "ann_return",
            "ann_vol",
            "sharpe",
            "max_drawdown",
            "calmar",
            "win_rate",
        ]
    ]

    trend_enh_bench = next(bundle for bundle in model_bundles if bundle["name"] == "Trend Aware Enhanced")["benchmark_table"].copy()
    trend_enh_bench = trend_enh_bench.rename(columns={"strategy": "Strategy"})
    for col in ["cumulative_return", "ann_return", "ann_vol", "max_drawdown", "win_rate"]:
        trend_enh_bench[col] = trend_enh_bench[col].apply(pct)
    for col in ["sharpe", "calmar"]:
        trend_enh_bench[col] = trend_enh_bench[col].apply(num)
    trend_enh_bench = trend_enh_bench[
        [
            "Strategy",
            "periods",
            "start",
            "end",
            "cumulative_return",
            "ann_return",
            "ann_vol",
            "sharpe",
            "max_drawdown",
            "calmar",
            "win_rate",
        ]
    ]

    sam_bench = sam_bundle["benchmark_table"].copy()
    sam_bench = sam_bench.rename(columns={"strategy": "Strategy"})
    for col in ["cumulative_return", "ann_return", "ann_vol", "max_drawdown", "win_rate"]:
        sam_bench[col] = sam_bench[col].apply(pct)
    for col in ["sharpe", "calmar"]:
        sam_bench[col] = sam_bench[col].apply(num)
    sam_bench = sam_bench[
        [
            "Strategy",
            "periods",
            "start",
            "end",
            "cumulative_return",
            "ann_return",
            "ann_vol",
            "sharpe",
            "max_drawdown",
            "calmar",
            "win_rate",
        ]
    ]

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Model Comparison Dashboard</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #0b1220;
      --muted: #495a73;
      --line: #d8dfec;
      --accent: #005f73;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: radial-gradient(circle at top right, #e6f7ff 0%, var(--bg) 40%);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1460px;
      margin: 0 auto;
      padding: 28px 22px 36px;
    }}
    h1 {{
      margin: 0 0 8px;
      letter-spacing: 0.01em;
    }}
    .sub {{
      color: var(--muted);
      margin-bottom: 18px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}
    .card {{
      background: linear-gradient(165deg, #ffffff 0%, #f7fbff 100%);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 2px 10px rgba(8, 19, 38, 0.04);
    }}
    .card h3 {{
      margin: 0 0 10px;
      font-size: 16px;
      color: var(--accent);
    }}
    .section {{
      margin-top: 18px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
    }}
    .section h2 {{
      margin: 0 0 12px;
      font-size: 19px;
    }}
    .imgbox {{
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px;
      background: #fff;
    }}
    .imgbox img {{
      width: 100%;
      height: auto;
      border-radius: 8px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 6px 8px;
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    th {{
      background: #edf4ff;
      color: #0e1f3a;
    }}
    .note {{
      font-size: 12px;
      color: var(--muted);
      line-height: 1.45;
    }}
    .pill {{
      display: inline-block;
      background: #eaf3ff;
      border: 1px solid #cfe2ff;
      color: #19436a;
      border-radius: 999px;
      padding: 3px 10px;
      margin-right: 8px;
      margin-bottom: 8px;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Primary-Model Comparison Dashboard</h1>
    <div class="sub">Primary Unified vs Trend Aware vs Trend Aware Enhanced vs Benchmark (Samwegs) | Generated {timestamp}</div>

    <div class="grid">
      <div class="card">
        <h3>Run Status</h3>
        <div><span class="pill">primary_model_unified.py run-all: OK</span></div>
        <div><span class="pill">Trend Aware (1) (1).py: OK</span></div>
        <div><span class="pill">Trend Aware (1) (1) - Enhanced.py: OK</span></div>
        <div><span class="pill">Benchmark (2).py: OK</span></div>
      </div>
      <div class="card">
        <h3>Star Marker</h3>
        <div class="note">Any value marked with <b>*</b> is the best value for that tested metric across the four models in that table.</div>
      </div>
      <div class="card">
        <h3>Included Aspects</h3>
        <div class="note">Return, volatility, Sharpe, drawdown, Calmar, win-rate, turnover, time-in-market, signal mix, allocation profile, equity curve, drawdown curve, rolling Sharpe, and benchmark context.</div>
      </div>
    </div>

    <div class="section">
      <h2>Core Performance</h2>
      {core_df.to_html(index=False, escape=False)}
    </div>

    <div class="section">
      <h2>Risk and Trading Behavior</h2>
      {risk_df.to_html(index=False, escape=False)}
    </div>

    <div class="section">
      <h2>Signal Profile</h2>
      {signal_df.to_html(index=False, escape=False)}
    </div>

    <div class="section">
      <h2>Average Allocation Profile</h2>
      {weights_df.to_html(index=False, escape=False)}
    </div>

    <div class="section">
      <h2>Equity Curves</h2>
      <div class="imgbox"><img src="{plots['equity'].relative_to(REPORTS_DIR)}" alt="Equity curves"></div>
    </div>

    <div class="section">
      <h2>Drawdown Curves</h2>
      <div class="imgbox"><img src="{plots['drawdown'].relative_to(REPORTS_DIR)}" alt="Drawdown curves"></div>
    </div>

    <div class="section">
      <h2>Rolling 12-Month Sharpe</h2>
      <div class="imgbox"><img src="{plots['rolling_sharpe'].relative_to(REPORTS_DIR)}" alt="Rolling Sharpe"></div>
    </div>

    <div class="section">
      <h2>Benchmark Context: Primary Unified</h2>
      {unified_bench.to_html(index=False, escape=False)}
    </div>

    <div class="section">
      <h2>Benchmark Context: Trend Aware</h2>
      {trend_bench.to_html(index=False, escape=False)}
    </div>

    <div class="section">
      <h2>Benchmark Context: Trend Aware Enhanced</h2>
      {trend_enh_bench.to_html(index=False, escape=False)}
    </div>

    <div class="section">
      <h2>Benchmark Context: Benchmark (Samwegs)</h2>
      {sam_bench.to_html(index=False, escape=False)}
    </div>

    <div class="section">
      <h2>Benchmark (Samwegs) Test-Only Slice</h2>
      {sam_test_df.to_html(index=False, escape=False)}
    </div>

    <div class="section">
      <h2>Method Notes</h2>
      <div class="note">
        <p>1. Annualized metrics assume monthly periodicity.</p>
        <p>2. Primary Unified uses net returns from <code>reports/primary_v1_backtest.csv</code>.</p>
        <p>3. Trend Aware uses monthly allocation outputs from <code>monthly_allocations.csv</code>.</p>
        <p>4. Trend Aware Enhanced uses net returns from <code>strategy_returns_enhanced.csv</code>.</p>
        <p>5. Benchmark (Samwegs) uses strategy columns from <code>reports/backtest_results.csv</code>.</p>
      </div>
    </div>
  </div>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    prices = load_raw_prices(RAW_FILES)
    returns = prices.pct_change()

    primary_bundle = build_primary_unified_bundle()
    trend_bundle = build_trend_aware_bundle(prices, returns)
    trend_enh_bundle = build_trend_aware_enhanced_bundle(prices, returns)
    sam_bundle = build_sam_benchmark_bundle()

    model_bundles = [primary_bundle, trend_bundle, trend_enh_bundle, sam_bundle]

    returns_map = {
        primary_bundle["name"]: primary_bundle["returns"],
        trend_bundle["name"]: trend_bundle["returns"],
        trend_enh_bundle["name"]: trend_enh_bundle["returns"],
        sam_bundle["name"]: sam_bundle["returns"],
    }
    plot_paths = save_comparison_plots(returns_map)

    output_path = REPORTS_DIR / "model_comparison_dashboard.html"
    render_dashboard(model_bundles, plot_paths, output_path)
    print(f"Saved dashboard: {output_path}")


if __name__ == "__main__":
    main()
