"""M2 v2: Core features only (no VIX/OAS) + threshold sweep.

Changes vs v1:
  1. Uses M2_FEATURES_CORE (12 features, drops VIX and OAS)
  2. Sweeps thresholds 0.10 -> 0.90 to find best Sharpe
  3. Compares v2 vs v1 side by side

Run from repo root:
    python scripts/run_m2_v2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

from metalabel.secondary.model import (
    M2_FEATURES_CORE,
    run_walk_forward,
    save_predictions,
    sweep_thresholds,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parent.parent
DATA_PATH    = ROOT / "reports" / "results" / "secondary_dataset.csv"
PRED_PATH    = ROOT / "reports" / "results" / "m2_v2_predictions.csv"
METRICS_PATH = ROOT / "reports" / "results" / "m2_v2_metrics.json"
OUT_DIR      = ROOT / "reports" / "assets" / "m2_v2_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

V1_METRICS   = ROOT / "reports" / "results" / "m2_metrics.json"

MIN_TRAIN = 60

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

print("=" * 60)
print("M2 V2 — CORE FEATURES + THRESHOLD SWEEP")
print("=" * 60)
print(f"Features used      : {len(M2_FEATURES_CORE)} (M2_FEATURES_CORE, no VIX/OAS)")
print(f"Total events       : {len(df)}")
print(f"Initial train size : {MIN_TRAIN}")
print(f"Out-of-sample      : {len(df) - MIN_TRAIN}")
print(f"Features dropped   : VIX (5) + OAS (5) = 10 features removed")

# ---------------------------------------------------------------------------
# Run walk-forward with core features (threshold=0.5 as baseline)
# ---------------------------------------------------------------------------
preds = run_walk_forward(df, min_train_size=MIN_TRAIN, threshold=0.5,
                         features=M2_FEATURES_CORE)
save_predictions(preds, PRED_PATH)

y_true = preds["meta_label"].values
y_prob = preds["m2_prob"].values
returns = preds["meta_target_return"].values

accuracy = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
auc      = roc_auc_score(y_true, y_prob)

m1_mean   = float(np.mean(returns))
m1_std    = float(np.std(returns))
m1_sharpe = float(m1_mean / m1_std * np.sqrt(12)) if m1_std > 0 else 0.0
m1_wr     = float(np.mean(y_true))

print(f"\nClassification (threshold=0.5):")
print(f"  Accuracy  : {accuracy:.3f}")
print(f"  ROC-AUC   : {auc:.3f}")

# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------
sweep = sweep_thresholds(preds)

print(f"\n{'Threshold':>10} {'N':>5} {'Pct%':>6} {'WinRate':>8} {'AvgRet':>8} {'Sharpe':>8} {'vs M1':>7}")
print("-" * 60)
for _, row in sweep.iterrows():
    if row["n_approved"] < 2:
        continue
    flag = " <-- BEST" if row["sharpe"] == sweep["sharpe"].max() else ""
    print(f"  {row['threshold']:>6.2f}    {int(row['n_approved']):>4}  "
          f"{row['pct_traded']:>5.0%}  {row['win_rate']:>7.1%}  "
          f"{row['avg_return']*100:>+6.2f}%  {row['sharpe']:>7.3f}  "
          f"{row['vs_m1_sharpe']:>+6.3f}{flag}")

best_row = sweep.loc[sweep["sharpe"].idxmax()]
best_threshold = float(best_row["threshold"])
best_sharpe    = float(best_row["sharpe"])

print(f"\nBest threshold by Sharpe: {best_threshold:.2f}  (Sharpe = {best_sharpe:.3f})")

# Re-run with best threshold to get final approved/rejected split
preds_best = run_walk_forward(df, min_train_size=MIN_TRAIN,
                              threshold=best_threshold,
                              features=M2_FEATURES_CORE)
approved = preds_best[preds_best["m2_approve"] == 1]
rejected = preds_best[preds_best["m2_approve"] == 0]

m2_wr     = float(approved["meta_label"].mean()) if len(approved) else float("nan")
m2_ret    = float(approved["meta_target_return"].mean()) if len(approved) else float("nan")
m2_sh_val = approved["meta_target_return"]
m2_sharpe = float(m2_sh_val.mean() / m2_sh_val.std() * np.sqrt(12)) \
            if (len(m2_sh_val) > 1 and m2_sh_val.std() > 0) else float("nan")

rej_wr  = float(rejected["meta_label"].mean()) if len(rejected) else float("nan")
rej_ret = float(rejected["meta_target_return"].mean()) if len(rejected) else float("nan")

print(f"\n{'':=<60}")
print(f"{'ECONOMIC PERFORMANCE — BEST THRESHOLD':^60}")
print(f"{'':=<60}")
print(f"{'':30s}  {'N':>4}  {'Win%':>6}  {'AvgRet':>7}  {'Sharpe':>7}")
print(f"{'M1 baseline (all OOS)':30s}  {len(preds):>4}  {m1_wr:>6.1%}  {m1_mean*100:>+6.2f}%  {m1_sharpe:>7.2f}")
print(f"{'M2 v2 approved (trade)':30s}  {len(approved):>4}  {m2_wr:>6.1%}  {m2_ret*100:>+6.2f}%  {m2_sharpe:>7.2f}")
print(f"{'M2 v2 rejected (skip)':30s}  {len(rejected):>4}  {rej_wr:>6.1%}  {rej_ret*100:>+6.2f}%  {'n/a':>7}")

# ---------------------------------------------------------------------------
# Compare v1 vs v2
# ---------------------------------------------------------------------------
v1_sharpe_approved = None
if V1_METRICS.exists():
    with open(V1_METRICS) as f:
        v1 = json.load(f)
    v1_sharpe_approved = v1["m2_approved"]["annualised_sharpe"]
    v1_auc = v1["roc_auc"]

    print(f"\n{'':=<60}")
    print(f"{'V1 vs V2 COMPARISON':^60}")
    print(f"{'':=<60}")
    print(f"{'':30s}  {'ROC-AUC':>8}  {'Sharpe(appr)':>12}")
    print(f"{'v1 (22 features, t=0.50)':30s}  {v1_auc:>8.3f}  {v1_sharpe_approved:>12.3f}")
    print(f"{'v2 (12 features, best t)':30s}  {auc:>8.3f}  {m2_sharpe:>12.3f}")

# ---------------------------------------------------------------------------
# Save metrics
# ---------------------------------------------------------------------------
metrics = {
    "version": "v2",
    "features": "M2_FEATURES_CORE (12, no VIX/OAS)",
    "n_features": len(M2_FEATURES_CORE),
    "oos_events": len(preds),
    "min_train_size": MIN_TRAIN,
    "threshold_sweep": sweep.dropna().to_dict(orient="records"),
    "best_threshold": best_threshold,
    "accuracy_at_0_5": round(accuracy, 4),
    "roc_auc": round(auc, 4),
    "m1_baseline": {
        "n": len(preds),
        "win_rate": round(m1_wr, 4),
        "avg_return": round(m1_mean, 6),
        "annualised_sharpe": round(m1_sharpe, 3),
    },
    "m2_approved_best_threshold": {
        "n": len(approved),
        "threshold": best_threshold,
        "win_rate": round(m2_wr, 4) if not np.isnan(m2_wr) else None,
        "avg_return": round(m2_ret, 6) if not np.isnan(m2_ret) else None,
        "annualised_sharpe": round(m2_sharpe, 3) if not np.isnan(m2_sharpe) else None,
    },
    "m2_rejected_best_threshold": {
        "n": len(rejected),
        "win_rate": round(rej_wr, 4) if not np.isnan(rej_wr) else None,
        "avg_return": round(rej_ret, 6) if not np.isnan(rej_ret) else None,
    },
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved -> {METRICS_PATH.relative_to(ROOT)}")

# ---------------------------------------------------------------------------
# Chart 1: Threshold sweep — Sharpe vs threshold
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("M2 v2: Threshold Sweep (Core Features, No VIX/OAS)", fontsize=13, fontweight="bold")

valid = sweep.dropna()
axes[0].plot(valid["threshold"], valid["sharpe"], color="#2ecc71", linewidth=2, marker="o", markersize=5)
axes[0].axhline(m1_sharpe, color="#3498db", linestyle="--", linewidth=1.5, label=f"M1 baseline ({m1_sharpe:.2f})")
axes[0].axvline(best_threshold, color="#e74c3c", linestyle=":", linewidth=1.5, label=f"Best t={best_threshold:.2f}")
axes[0].set_xlabel("Threshold")
axes[0].set_ylabel("Annualised Sharpe (M2 approved trades)")
axes[0].set_title("Sharpe vs Threshold")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].bar(valid["threshold"], valid["n_approved"], width=0.03, color="#9b59b6", alpha=0.8)
axes[1].axhline(len(preds), color="#3498db", linestyle="--", linewidth=1.2, label=f"All OOS ({len(preds)})")
axes[1].set_xlabel("Threshold")
axes[1].set_ylabel("N Approved Trades")
axes[1].set_title("Trades Approved vs Threshold")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "1_threshold_sweep.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[Chart 1] Threshold sweep saved.")

# ---------------------------------------------------------------------------
# Chart 2: V1 vs V2 comparison
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("M2 v1 vs v2: Side-by-Side Comparison", fontsize=13, fontweight="bold")

versions    = ["M1 Baseline", "v1 (22 feat\nt=0.50)", "v2 (12 feat\nbest t)"]
sharpes     = [m1_sharpe,
               v1_sharpe_approved if v1_sharpe_approved else float("nan"),
               m2_sharpe]
win_rates   = [m1_wr,
               v1["m2_approved"]["win_rate"] if V1_METRICS.exists() else float("nan"),
               m2_wr]
colors      = ["#95a5a6", "#e74c3c", "#2ecc71"]

bars = axes[0].bar(versions, sharpes, color=colors, edgecolor="white", width=0.5)
for bar, val in zip(bars, sharpes):
    if not np.isnan(val):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")
axes[0].axhline(m1_sharpe, color="#3498db", linestyle="--", linewidth=1.2, label=f"M1 ({m1_sharpe:.2f})")
axes[0].set_ylabel("Annualised Sharpe")
axes[0].set_title("Sharpe Comparison")
axes[0].set_ylim(0, max(s for s in sharpes if not np.isnan(s)) * 1.3)
axes[0].legend()

bars2 = axes[1].bar(versions, [r*100 for r in win_rates], color=colors, edgecolor="white", width=0.5)
for bar, val in zip(bars2, win_rates):
    if not np.isnan(val):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val:.1%}", ha="center", fontsize=10, fontweight="bold")
axes[1].axhline(50, color="red", linestyle=":", linewidth=1.2, label="Random (50%)")
axes[1].set_ylabel("Win Rate (%)")
axes[1].set_title("Win Rate Comparison")
axes[1].set_ylim(0, 100)
axes[1].legend()

plt.tight_layout()
fig.savefig(OUT_DIR / "2_v1_vs_v2_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[Chart 2] v1 vs v2 comparison saved.")

# ---------------------------------------------------------------------------
# Chart 3: Cumulative return at best threshold
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 5))
fig.suptitle(f"M2 v2: Cumulative Return at Best Threshold ({best_threshold:.2f})",
             fontsize=13, fontweight="bold")

preds_sorted = preds_best.sort_values("date").reset_index(drop=True)
m1_cum = (1 + preds_sorted["meta_target_return"]).cumprod()
m2_ret_series = preds_sorted["meta_target_return"].where(preds_sorted["m2_approve"] == 1, 0.0)
m2_cum = (1 + m2_ret_series).cumprod()
dates  = pd.to_datetime(preds_sorted["date"])

ax.plot(dates, m1_cum, label=f"M1 Baseline (n={len(preds)})", color="#3498db", linewidth=2)
ax.plot(dates, m2_cum, label=f"M2 v2 Approved (n={len(approved)}, t={best_threshold:.2f})",
        color="#2ecc71", linewidth=2)
ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)

for _, row in preds_sorted[preds_sorted["m2_approve"] == 0].iterrows():
    ax.axvspan(row["date"], row["date"] + pd.Timedelta(days=20), alpha=0.07, color="#e74c3c")

ax.set_ylabel("Cumulative Growth ($1 invested)")
ax.set_xlabel("Date")
ax.set_title(f"OOS: {dates.iloc[0].date()} to {dates.iloc[-1].date()}  |  Red shading = M2 rejected")
ax.legend()

plt.tight_layout()
fig.savefig(OUT_DIR / "3_cumulative_return_best_threshold.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[Chart 3] Cumulative return saved.")

print(f"\n{'':=<60}")
print(f"All outputs in: {OUT_DIR.relative_to(ROOT)}/")
print(f"{'':=<60}")
