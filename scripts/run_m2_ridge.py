"""M2 Ridge vs Baseline comparison.

Runs two walk-forward backtests side by side:
  1. Baseline  - logistic regression, C=0.5 fixed
  2. Ridge CV  - logistic regression, C tuned by inner 5-fold CV each step

Compares on:
  - ROC-AUC (classification skill)
  - Position sizing: Ann Return, Sharpe, Info Ratio
  - C values chosen across walk-forward steps (Ridge only)

Run from repo root:
    python scripts/run_m2_ridge.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from metalabel.secondary.model import (
    M2_FEATURES_CORE,
    apply_position_sizing,
    compute_carry_returns,
    run_walk_forward,
    run_walk_forward_ridge,
)

ROOT     = Path(__file__).resolve().parent.parent
DATA     = ROOT / "reports" / "results" / "secondary_dataset.csv"
OUT_DIR  = ROOT / "reports" / "assets" / "m2_ridge"
METRICS  = ROOT / "reports" / "results" / "m2_ridge_metrics.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA, parse_dates=["date", "realized_date"])
df = df.sort_values("date").reset_index(drop=True)

# ---------------------------------------------------------------------------
# Load asset returns for carry-forward calculation
# ---------------------------------------------------------------------------
def load_asset(name):
    a = pd.read_csv(ROOT / "data" / "clean" / f"{name}.csv", parse_dates=["Date"])
    return a.set_index("Date")["Return"].rename(name)

asset_rets = pd.concat([
    load_asset("spx"),
    load_asset("bcom"),
    load_asset("corp_bonds"),
    load_asset("treasury_10y"),
], axis=1).sort_index()

w_cols = ["date", "realized_date", "weight_spx", "weight_bcom",
          "weight_treasury_10y", "weight_corp_bonds"]

# ---------------------------------------------------------------------------
# Run both models
# ---------------------------------------------------------------------------
print("Running baseline (C=0.5 fixed)...")
base_preds = run_walk_forward(df, min_train_size=60, threshold=0.5,
                              features=M2_FEATURES_CORE)
base_preds = base_preds.merge(df[w_cols], on="date", how="left")
base_carry = compute_carry_returns(base_preds, asset_rets)

print("Running Ridge CV (C tuned per step)...")
ridge_preds, c_vals = run_walk_forward_ridge(df, min_train_size=60,
                                             threshold=0.5,
                                             features=M2_FEATURES_CORE)
ridge_preds = ridge_preds.merge(df[w_cols], on="date", how="left")
ridge_carry = compute_carry_returns(ridge_preds, asset_rets)

# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------
base_auc  = roc_auc_score(base_preds["meta_label"],  base_preds["m2_prob"])
ridge_auc = roc_auc_score(ridge_preds["meta_label"], ridge_preds["m2_prob"])

fpr_b, tpr_b, _ = roc_curve(base_preds["meta_label"],  base_preds["m2_prob"])
fpr_r, tpr_r, _ = roc_curve(ridge_preds["meta_label"], ridge_preds["m2_prob"])

# ---------------------------------------------------------------------------
# Position sizing metrics
# ---------------------------------------------------------------------------
def sizing_stats(preds, carry):
    sized  = apply_position_sizing(preds, normalize=True, carry_returns=carry)
    rets   = sized["sized_return"].values
    m1     = sized["meta_target_return"].values
    mu     = np.mean(rets)
    std    = np.std(rets)
    sharpe = mu / std * np.sqrt(12) if std > 0 else np.nan
    ann    = (1 + mu) ** 12 - 1
    active = rets - m1
    te     = np.std(active)
    ir     = np.mean(active) / te * np.sqrt(12) if te > 0 else 0.0
    return ann, sharpe, ir, sized

m1_rets    = base_preds["meta_target_return"].values
m1_mu      = np.mean(m1_rets)
m1_ann     = (1 + m1_mu) ** 12 - 1
m1_sharpe  = m1_mu / np.std(m1_rets) * np.sqrt(12)

base_ann,  base_sh,  base_ir,  base_sized  = sizing_stats(base_preds,  base_carry)
ridge_ann, ridge_sh, ridge_ir, ridge_sized = sizing_stats(ridge_preds, ridge_carry)

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
print(f"\n{'':=<72}")
print(f"{'M2 RIDGE CV vs BASELINE COMPARISON':^72}")
print(f"{'':=<72}")
print(f"\n{'Metric':<35} {'Baseline':>15}  {'Ridge CV':>15}")
print("-" * 72)
print(f"{'ROC-AUC (OOS)':<35} {base_auc:>15.4f}  {ridge_auc:>15.4f}")
print(f"{'Ann Return (position sizing)':<35} {base_ann:>15.2%}  {ridge_ann:>15.2%}")
print(f"{'Sharpe (position sizing)':<35} {base_sh:>15.3f}  {ridge_sh:>15.3f}")
print(f"{'Info Ratio (position sizing)':<35} {base_ir:>+15.3f}  {ridge_ir:>+15.3f}")
print(f"\nM1 baseline: Ann={m1_ann:.2%}  Sharpe={m1_sharpe:.3f}  IR=+0.000")
print(f"\nRidge C values chosen across {len(c_vals)} walk-forward steps:")
from collections import Counter
c_counts = Counter(c_vals)
for c, cnt in sorted(c_counts.items()):
    print(f"  C = {c:.2f}  ->  selected {cnt} times ({cnt/len(c_vals):.0%})")
print(f"{'':=<72}")

# ---------------------------------------------------------------------------
# Save metrics
# ---------------------------------------------------------------------------
metrics = {
    "baseline": {
        "model": "LogisticRegression(C=0.5, penalty=l2)",
        "auc": round(base_auc, 4),
        "ann_return": round(base_ann, 4),
        "sharpe": round(base_sh, 3),
        "ir": round(base_ir, 3),
    },
    "ridge_cv": {
        "model": "LogisticRegressionCV(Cs=[0.01,0.05,0.1,0.5,1,5,10], cv=5)",
        "auc": round(ridge_auc, 4),
        "ann_return": round(ridge_ann, 4),
        "sharpe": round(ridge_sh, 3),
        "ir": round(ridge_ir, 3),
        "c_values_chosen": dict(sorted({str(k): v for k, v in c_counts.items()}.items())),
    },
    "m1_baseline": {
        "ann_return": round(m1_ann, 4),
        "sharpe": round(m1_sharpe, 3),
        "ir": 0.0,
    },
}
with open(METRICS, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved -> {METRICS.relative_to(ROOT)}")

# Save ridge predictions for use in other scripts
ridge_out = ROOT / "reports" / "results" / "m2_ridge_predictions.csv"
ridge_preds[["date", "meta_label", "meta_target_return", "m2_prob"]].to_csv(
    ridge_out, index=False)
print(f"Ridge predictions saved -> {ridge_out.relative_to(ROOT)}")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
BG    = "#0B1520"; CARD  = "#111E2D"; PANEL = "#162230"
TEAL  = "#00C4A8"; GOLD  = "#FFC432"; GREEN = "#2ecc71"
RED   = "#e74c3c"; BLUE  = "#4A90D9"; WHITE = "#FFFFFF"
GREY  = "#8CA0B8"; LGREY = "#C8D2DC"

fig = plt.figure(figsize=(16, 10), facecolor=BG)
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35,
                       left=0.07, right=0.97, top=0.90, bottom=0.08)

fig.suptitle("M2 Ridge CV vs Baseline: Does Tuning C Help?",
             fontsize=14, fontweight="bold", color=WHITE, y=0.97)

# ---- Panel 1: ROC curves ----
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(PANEL)
ax1.plot(fpr_b, tpr_b, color=BLUE,  lw=2.2, label=f"Baseline  (AUC={base_auc:.3f})")
ax1.plot(fpr_r, tpr_r, color=TEAL,  lw=2.2, label=f"Ridge CV  (AUC={ridge_auc:.3f})", ls="--")
ax1.plot([0,1],[0,1], color=GREY,   lw=1.2, ls=":", label="Random (0.500)")
ax1.set_title("ROC Curve", color=WHITE, fontsize=10, fontweight="bold")
ax1.set_xlabel("False Positive Rate", color=LGREY, fontsize=8.5)
ax1.set_ylabel("True Positive Rate", color=LGREY, fontsize=8.5)
ax1.tick_params(colors=GREY, labelsize=8)
ax1.legend(fontsize=8, facecolor=CARD, edgecolor="#2A4060", labelcolor=LGREY)
ax1.grid(alpha=0.15, color=GREY)
for sp in ax1.spines.values(): sp.set_edgecolor("#1A2A3A")

# ---- Panel 2: Cumulative returns ----
ax2 = fig.add_subplot(gs[0, 1:])
ax2.set_facecolor(PANEL)
dates = pd.to_datetime(base_preds["date"])
ax2.plot(dates, (1 + m1_rets).cumprod(),
         color=GREY, lw=2.0, ls="--", label="M1 Baseline", alpha=0.8)
ax2.plot(dates, (1 + base_sized["sized_return"]).cumprod(),
         color=BLUE, lw=2.2, label=f"Baseline sizing  (IR={base_ir:+.3f})")
ax2.plot(dates, (1 + ridge_sized["sized_return"]).cumprod(),
         color=TEAL, lw=2.2, ls="--", label=f"Ridge CV sizing  (IR={ridge_ir:+.3f})")
ax2.axhline(1.0, color=WHITE, lw=0.6, ls=":")
ax2.set_title("Cumulative Return - Position Sizing", color=WHITE, fontsize=10, fontweight="bold")
ax2.set_ylabel("Growth of $1", color=LGREY, fontsize=8.5)
ax2.tick_params(colors=GREY, labelsize=8)
ax2.legend(fontsize=8.5, facecolor=CARD, edgecolor="#2A4060", labelcolor=LGREY)
ax2.grid(alpha=0.15, color=GREY)
for sp in ax2.spines.values(): sp.set_edgecolor("#1A2A3A")

# ---- Panel 3: Bar comparison ----
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(PANEL)
metrics_labels = ["Ann Return", "Sharpe", "Info Ratio"]
m1_vals    = [m1_ann * 100,    m1_sharpe,  0.0]
base_vals  = [base_ann * 100,  base_sh,    base_ir]
ridge_vals = [ridge_ann * 100, ridge_sh,   ridge_ir]
x = np.arange(3)
w = 0.25
ax3.bar(x - w, m1_vals,    w, label="M1",        color=GREY,  alpha=0.8)
ax3.bar(x,     base_vals,  w, label="Baseline",  color=BLUE,  alpha=0.9)
ax3.bar(x + w, ridge_vals, w, label="Ridge CV",  color=TEAL,  alpha=0.9)
ax3.set_xticks(x)
ax3.set_xticklabels(metrics_labels, color=LGREY, fontsize=8.5)
ax3.axhline(0, color=WHITE, lw=0.6)
ax3.set_title("Metric Comparison", color=WHITE, fontsize=10, fontweight="bold")
ax3.tick_params(colors=GREY, labelsize=8)
ax3.legend(fontsize=8, facecolor=CARD, edgecolor="#2A4060", labelcolor=LGREY)
ax3.grid(alpha=0.15, color=GREY, axis="y")
for sp in ax3.spines.values(): sp.set_edgecolor("#1A2A3A")

# ---- Panel 4: C value distribution ----
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(PANEL)
c_options = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
c_freqs   = [c_counts.get(c, 0) for c in c_options]
bars = ax4.bar([str(c) for c in c_options], c_freqs,
               color=TEAL, alpha=0.85, edgecolor=BG)
for bar, freq in zip(bars, c_freqs):
    if freq > 0:
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 str(freq), ha="center", fontsize=8, color=WHITE, fontweight="bold")
ax4.set_title("C Values Chosen by CV", color=WHITE, fontsize=10, fontweight="bold")
ax4.set_xlabel("Regularisation Strength (C)", color=LGREY, fontsize=8.5)
ax4.set_ylabel("Times selected", color=LGREY, fontsize=8.5)
ax4.tick_params(colors=GREY, labelsize=8)
ax4.grid(alpha=0.15, color=GREY, axis="y")
for sp in ax4.spines.values(): sp.set_edgecolor("#1A2A3A")

# ---- Panel 5: Probability distributions ----
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor(PANEL)
ax5.hist(base_preds["m2_prob"],  bins=20, alpha=0.6, color=BLUE,
         label="Baseline", density=True)
ax5.hist(ridge_preds["m2_prob"], bins=20, alpha=0.6, color=TEAL,
         label="Ridge CV", density=True)
ax5.axvline(0.5, color=GOLD, lw=1.5, ls="--", label="t=0.50")
ax5.set_title("M2 Probability Distribution", color=WHITE, fontsize=10, fontweight="bold")
ax5.set_xlabel("m2_prob", color=LGREY, fontsize=8.5)
ax5.set_ylabel("Density", color=LGREY, fontsize=8.5)
ax5.tick_params(colors=GREY, labelsize=8)
ax5.legend(fontsize=8, facecolor=CARD, edgecolor="#2A4060", labelcolor=LGREY)
ax5.grid(alpha=0.15, color=GREY)
for sp in ax5.spines.values(): sp.set_edgecolor("#1A2A3A")

out = OUT_DIR / "ridge_vs_baseline.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Chart saved -> {out.relative_to(ROOT)}")
