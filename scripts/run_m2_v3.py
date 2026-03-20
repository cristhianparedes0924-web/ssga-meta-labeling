"""M2 v3: Position sizing instead of binary gate.

Instead of approve/reject, M2's probability scales position size:
    m2_prob = 0.80  ->  trade at 1.28x normal size  (normalized)
    m2_prob = 0.55  ->  trade at 0.88x normal size
    m2_prob = 0.35  ->  trade at 0.56x normal size

Compares four setups side by side:
    M1 baseline          : all 78 trades at size=1
    M2 binary t=0.51     : 28 trades at size=1, 50 earn 0
    M2 sizing (raw)      : all 78 trades, size = m2_prob (0 to 1)
    M2 sizing (normalised): all 78 trades, size = m2_prob / mean(m2_prob)

Run from repo root:
    python scripts/run_m2_v3.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metalabel.secondary.model import (
    M2_FEATURES_CORE,
    apply_position_sizing,
    run_walk_forward,
)

# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "reports" / "results" / "secondary_dataset.csv"
OUT_DIR   = ROOT / "reports" / "assets" / "m2_v3_results"
METRICS_PATH = ROOT / "reports" / "results" / "m2_v3_metrics.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TRAIN = 60

# ---------------------------------------------------------------------------
# Load and run walk-forward (core features, no threshold needed for sizing)
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

preds = run_walk_forward(df, min_train_size=MIN_TRAIN,
                         threshold=0.5, features=M2_FEATURES_CORE)
preds = preds.sort_values("date").reset_index(drop=True)

# ---------------------------------------------------------------------------
# Build the four return streams
# ---------------------------------------------------------------------------
m1_returns   = preds["meta_target_return"].values

# 1. M1 baseline: size=1 on every trade
m1_stream = m1_returns.copy()

# 2. M2 binary gate at 0.51: approved trades at size=1, rejected earn 0
binary_stream = np.where(preds["m2_prob"] >= 0.51, m1_returns, 0.0)

# 3. M2 raw sizing: size = m2_prob (always < 1, de-risked vs M1)
sized_raw = apply_position_sizing(preds, normalize=False)
raw_stream = sized_raw["sized_return"].values

# 4. M2 normalised sizing: size = m2_prob / mean(m2_prob), avg exposure = M1
sized_norm = apply_position_sizing(preds, normalize=True)
norm_stream = sized_norm["sized_return"].values

print("Position size distribution (normalised):")
sizes = sized_norm["position_size"].values
print(f"  Min:  {sizes.min():.3f}x")
print(f"  Mean: {sizes.mean():.3f}x")
print(f"  Max:  {sizes.max():.3f}x")
print(f"  Std:  {sizes.std():.3f}")

# ---------------------------------------------------------------------------
# Stats function
# ---------------------------------------------------------------------------
def stats(portfolio, benchmark):
    mu      = np.mean(portfolio)
    std     = np.std(portfolio)
    sharpe  = mu / std * np.sqrt(12) if std > 0 else np.nan
    ann_ret = (1 + mu) ** 12 - 1
    active  = portfolio - benchmark
    te      = np.std(active)
    ir      = np.mean(active) / te * np.sqrt(12) if te > 0 else 0.0
    return ann_ret, sharpe, ir

m1_ann,  m1_sh,  m1_ir  = stats(m1_stream,     m1_stream)
bi_ann,  bi_sh,  bi_ir  = stats(binary_stream,  m1_stream)
raw_ann, raw_sh, raw_ir = stats(raw_stream,      m1_stream)
nor_ann, nor_sh, nor_ir = stats(norm_stream,     m1_stream)

# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------
print(f"\n{'':=<72}")
print(f"{'POSITION SIZING vs BINARY GATE — FULL COMPARISON':^72}")
print(f"{'':=<72}")
print(f"{'Setup':<32} {'N':>4}  {'Ann Return':>10}  {'Sharpe':>7}  {'Info Ratio':>11}")
print("-" * 72)
print(f"{'M1 baseline (no filter)':<32} {'78':>4}  {m1_ann:>9.2%}  {m1_sh:>7.3f}  {m1_ir:>+11.3f}")
print(f"{'M2 binary gate t=0.51':<32} {'28':>4}  {bi_ann:>9.2%}  {bi_sh:>7.3f}  {bi_ir:>+11.3f}")
print(f"{'M2 raw sizing (prob direct)':<32} {'78':>4}  {raw_ann:>9.2%}  {raw_sh:>7.3f}  {raw_ir:>+11.3f}")
print(f"{'M2 norm sizing (avg=1.0x)':<32} {'78':>4}  {nor_ann:>9.2%}  {nor_sh:>7.3f}  {nor_ir:>+11.3f}")
print()
print("IR benchmark: M1 full return stream.")
print("Binary gate rejected months earn 0. Sizing trades always participate.")

# ---------------------------------------------------------------------------
# Save metrics
# ---------------------------------------------------------------------------
metrics = {
    "version": "v3 - position sizing",
    "position_size_stats": {
        "min":  round(float(sizes.min()), 4),
        "mean": round(float(sizes.mean()), 4),
        "max":  round(float(sizes.max()), 4),
        "std":  round(float(sizes.std()), 4),
    },
    "results": {
        "m1_baseline":      {"n": 78, "ann_return": round(m1_ann, 4), "sharpe": round(m1_sh, 3), "ir": round(m1_ir, 3)},
        "m2_binary_t051":   {"n": 28, "ann_return": round(bi_ann, 4), "sharpe": round(bi_sh, 3), "ir": round(bi_ir, 3)},
        "m2_sizing_raw":    {"n": 78, "ann_return": round(raw_ann, 4), "sharpe": round(raw_sh, 3), "ir": round(raw_ir, 3)},
        "m2_sizing_norm":   {"n": 78, "ann_return": round(nor_ann, 4), "sharpe": round(nor_sh, 3), "ir": round(nor_ir, 3)},
    }
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved -> {METRICS_PATH.relative_to(ROOT)}")

# ---------------------------------------------------------------------------
# Chart 1: Cumulative return — all four strategies
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 6))
fig.suptitle("M2 v3: Position Sizing vs Binary Gate vs M1 Baseline",
             fontsize=13, fontweight="bold")

dates = pd.to_datetime(preds["date"])

cum = {
    "M1 Baseline (size=1 always)":         (1 + m1_stream).cumprod(),
    "M2 Binary Gate t=0.51 (28 trades)":   (1 + binary_stream).cumprod(),
    "M2 Raw Sizing (prob direct)":          (1 + raw_stream).cumprod(),
    "M2 Normalised Sizing (avg=1.0x)":     (1 + norm_stream).cumprod(),
}
colors = ["#3498db", "#e74c3c", "#f39c12", "#2ecc71"]
styles = ["-", "--", "-.", "-"]
widths = [2.5, 2, 1.8, 2.2]

for (label, series), col, ls, lw in zip(cum.items(), colors, styles, widths):
    ax.plot(dates, series, label=label, color=col, linestyle=ls, linewidth=lw)

ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8)
ax.set_ylabel("Cumulative Growth ($1 invested)")
ax.set_xlabel("Date")
ax.set_title(f"OOS: {dates.iloc[0].date()} to {dates.iloc[-1].date()}")
ax.legend(fontsize=9)
ax.grid(alpha=0.2)

plt.tight_layout()
fig.savefig(OUT_DIR / "1_cumulative_return_sizing.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[Chart 1] Cumulative return saved.")

# ---------------------------------------------------------------------------
# Chart 2: Position sizes over time
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(13, 8))
fig.suptitle("M2 v3: Position Sizes Over Time (Normalised)", fontsize=13, fontweight="bold")

axes[0].bar(dates, sized_norm["position_size"], color=np.where(
    sized_norm["position_size"] >= 1.0, "#2ecc71", "#e74c3c"), alpha=0.8, width=20)
axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="M1 full size (1.0x)")
axes[0].axhline(sized_norm["position_size"].mean(), color="#3498db",
                linestyle=":", linewidth=1, label=f"Mean ({sized_norm['position_size'].mean():.2f}x)")
axes[0].set_ylabel("Position Size (x)")
axes[0].set_title("Green = above full size (M2 confident), Red = below (M2 uncertain)")
axes[0].legend()
axes[0].grid(alpha=0.2)

# Rolling Sharpe: M1 vs normalised sizing
window = 24
roll_m1  = pd.Series(m1_stream).rolling(window).apply(
    lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else np.nan)
roll_nor = pd.Series(norm_stream).rolling(window).apply(
    lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else np.nan)

axes[1].plot(dates, roll_m1,  label=f"M1 baseline ({window}m rolling Sharpe)",
             color="#3498db", linewidth=1.8)
axes[1].plot(dates, roll_nor, label=f"M2 norm sizing ({window}m rolling Sharpe)",
             color="#2ecc71", linewidth=1.8)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_ylabel("Rolling Sharpe")
axes[1].set_title(f"{window}-Month Rolling Sharpe: M1 vs M2 Normalised Sizing")
axes[1].legend()
axes[1].grid(alpha=0.2)

plt.tight_layout()
fig.savefig(OUT_DIR / "2_position_sizes.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[Chart 2] Position sizes saved.")

# ---------------------------------------------------------------------------
# Chart 3: Bar chart summary
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("M2 v3: Summary Comparison", fontsize=13, fontweight="bold")

labels   = ["M1\nBaseline", "M2 Binary\nt=0.51", "M2 Raw\nSizing", "M2 Norm\nSizing"]
ann_rets = [m1_ann, bi_ann, raw_ann, nor_ann]
sharpes  = [m1_sh,  bi_sh,  raw_sh,  nor_sh]
irs      = [m1_ir,  bi_ir,  raw_ir,  nor_ir]
colors   = ["#3498db", "#e74c3c", "#f39c12", "#2ecc71"]

for ax, vals, title, ylabel, fmt in zip(
    axes,
    [ann_rets, sharpes, irs],
    ["Annualised Return", "Sharpe Ratio", "Information Ratio"],
    ["Ann. Return", "Sharpe", "IR"],
    ["{:.1%}", "{:.3f}", "{:+.3f}"],
):
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, vals):
        ypos = bar.get_height() + (0.002 if val >= 0 else -0.015)
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                fmt.format(val), ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)

plt.tight_layout()
fig.savefig(OUT_DIR / "3_summary_bars.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[Chart 3] Summary bars saved.")

print(f"\n{'':=<72}")
print(f"All outputs in: {OUT_DIR.relative_to(ROOT)}/")
print(f"{'':=<72}")
