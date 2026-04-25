"""Run the M2 secondary meta-labeling model end-to-end.

Loads the secondary dataset, runs walk-forward prediction, prints a
statistical summary, saves predictions CSV, and writes 3 result charts
to reports/assets/m2_results/.

Run from repo root:

    python scripts/run_m2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from metalabel.secondary.model import M2_FEATURES, run_walk_forward, save_predictions

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT / "reports" / "results" / "secondary_dataset.csv"
PRED_PATH  = ROOT / "reports" / "results" / "m2_predictions.csv"
METRICS_PATH = ROOT / "reports" / "results" / "m2_metrics.json"
OUT_DIR    = ROOT / "reports" / "assets" / "m2_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TRAIN  = 60
THRESHOLD  = 0.5

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

print("=" * 60)
print("M2 WALK-FORWARD RESULTS")
print("=" * 60)
print(f"Total events       : {len(df)}")
print(f"Initial train size : {MIN_TRAIN}")
print(f"Out-of-sample      : {len(df) - MIN_TRAIN}")
print(f"Date range (OOS)   : {df['date'].iloc[MIN_TRAIN].date()} → {df['date'].iloc[-1].date()}")

# ---------------------------------------------------------------------------
# Run walk-forward
# ---------------------------------------------------------------------------
preds = run_walk_forward(df, min_train_size=MIN_TRAIN, threshold=THRESHOLD)
save_predictions(preds, PRED_PATH)
print(f"\nPredictions saved → {PRED_PATH.relative_to(ROOT)}")

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
y_true     = preds["meta_label"].values
y_pred     = preds["m2_approve"].values
y_prob     = preds["m2_prob"].values
returns    = preds["meta_target_return"].values

accuracy   = accuracy_score(y_true, y_pred)
auc        = roc_auc_score(y_true, y_prob)
approved   = preds[preds["m2_approve"] == 1]
rejected   = preds[preds["m2_approve"] == 0]

# M1 baseline: all OOS events
m1_win_rate   = float(np.mean(y_true))
m1_avg_return = float(np.mean(returns))
m1_sharpe     = float(np.mean(returns) / np.std(returns) * np.sqrt(12)) if np.std(returns) > 0 else 0.0

# M2 approved: only trade when M2 says yes
m2_win_rate   = float(approved["meta_label"].mean()) if len(approved) else float("nan")
m2_avg_return = float(approved["meta_target_return"].mean()) if len(approved) else float("nan")
m2_sharpe_val = approved["meta_target_return"]
m2_sharpe     = float(m2_sharpe_val.mean() / m2_sharpe_val.std() * np.sqrt(12)) if (len(m2_sharpe_val) > 1 and m2_sharpe_val.std() > 0) else float("nan")

# M2 rejected: trades M2 filtered out
rej_win_rate   = float(rejected["meta_label"].mean()) if len(rejected) else float("nan")
rej_avg_return = float(rejected["meta_target_return"].mean()) if len(rejected) else float("nan")

print(f"\n{'':=<60}")
print(f"{'CLASSIFICATION METRICS':^60}")
print(f"{'':=<60}")
print(f"  Accuracy           : {accuracy:.3f}")
print(f"  ROC-AUC            : {auc:.3f}")
print(f"\n{classification_report(y_true, y_pred, target_names=['M1 Loss', 'M1 Win'])}")

print(f"{'':=<60}")
print(f"{'ECONOMIC PERFORMANCE (OOS)':^60}")
print(f"{'':=<60}")
print(f"{'':30s}  {'N':>4}  {'Win%':>6}  {'AvgRet':>7}  {'Sharpe':>7}")
print(f"{'M1 baseline (all OOS)':30s}  {len(preds):>4}  {m1_win_rate:>6.1%}  {m1_avg_return*100:>+6.2f}%  {m1_sharpe:>7.2f}")
print(f"{'M2 approved (trade)':30s}  {len(approved):>4}  {m2_win_rate:>6.1%}  {m2_avg_return*100:>+6.2f}%  {m2_sharpe:>7.2f}")
print(f"{'M2 rejected (skip)':30s}  {len(rejected):>4}  {rej_win_rate:>6.1%}  {rej_avg_return*100:>+6.2f}%  {'n/a':>7}")

# Save metrics JSON
metrics = {
    "oos_events": len(preds),
    "min_train_size": MIN_TRAIN,
    "threshold": THRESHOLD,
    "accuracy": round(accuracy, 4),
    "roc_auc": round(auc, 4),
    "m1_baseline": {
        "n": len(preds),
        "win_rate": round(m1_win_rate, 4),
        "avg_return": round(m1_avg_return, 6),
        "annualised_sharpe": round(m1_sharpe, 3),
    },
    "m2_approved": {
        "n": len(approved),
        "win_rate": round(m2_win_rate, 4) if not np.isnan(m2_win_rate) else None,
        "avg_return": round(m2_avg_return, 6) if not np.isnan(m2_avg_return) else None,
        "annualised_sharpe": round(m2_sharpe, 3) if not np.isnan(m2_sharpe) else None,
    },
    "m2_rejected": {
        "n": len(rejected),
        "win_rate": round(rej_win_rate, 4) if not np.isnan(rej_win_rate) else None,
        "avg_return": round(rej_avg_return, 6) if not np.isnan(rej_avg_return) else None,
    },
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved → {METRICS_PATH.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Chart 1: Predicted probability distribution
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("M2 Walk-Forward: Predicted Probability Distribution", fontsize=13, fontweight="bold")

wins_prob  = preds[preds["meta_label"] == 1]["m2_prob"]
losses_prob = preds[preds["meta_label"] == 0]["m2_prob"]

bins = np.linspace(0, 1, 21)
axes[0].hist(wins_prob,   bins=bins, alpha=0.65, color="#2ecc71", label=f"M1 Win  (n={len(wins_prob)})")
axes[0].hist(losses_prob, bins=bins, alpha=0.65, color="#e74c3c", label=f"M1 Loss (n={len(losses_prob)})")
axes[0].axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.5, label=f"Threshold={THRESHOLD}")
axes[0].set_xlabel("M2 Predicted Probability (P(M1 wins))")
axes[0].set_ylabel("Count")
axes[0].set_title("Prob distribution by actual outcome")
axes[0].legend()

# Calibration-style: win rate by prob bucket
preds["prob_bucket"] = pd.cut(preds["m2_prob"], bins=np.linspace(0, 1, 6), include_lowest=True)
calib = preds.groupby("prob_bucket", observed=True)["meta_label"].agg(["mean", "count"])
axes[1].bar(range(len(calib)), calib["mean"], color="#3498db", edgecolor="white")
for i, (_, row) in enumerate(calib.iterrows()):
    axes[1].text(i, row["mean"] + 0.02, f"{row['mean']:.0%}\nn={int(row['count'])}", ha="center", fontsize=9)
axes[1].axhline(m1_win_rate, color="black", linestyle="--", linewidth=1.2, label=f"M1 base ({m1_win_rate:.1%})")
axes[1].set_xticks(range(len(calib)))
axes[1].set_xticklabels([str(b) for b in calib.index], fontsize=8)
axes[1].set_ylim(0, 1.15)
axes[1].set_ylabel("Actual M1 win rate")
axes[1].set_title("Actual win rate by predicted prob bucket")
axes[1].legend()

plt.tight_layout()
fig.savefig(OUT_DIR / "1_probability_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[Chart 1] Probability distribution saved.")


# ---------------------------------------------------------------------------
# Chart 2: M2 approve/reject performance comparison
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("M2 Walk-Forward: Approve vs Reject Economic Performance", fontsize=13, fontweight="bold")

# Win rate comparison
labels     = ["M1 Baseline\n(all OOS)", "M2 Approved\n(trade)", "M2 Rejected\n(skip)"]
win_rates  = [m1_win_rate, m2_win_rate, rej_win_rate]
avg_rets   = [m1_avg_return * 100, m2_avg_return * 100, rej_avg_return * 100]
colors_bar = ["#95a5a6", "#2ecc71", "#e74c3c"]
ns         = [len(preds), len(approved), len(rejected)]

bars = axes[0].bar(labels, win_rates, color=colors_bar, edgecolor="white", width=0.5)
for bar, wr, n in zip(bars, win_rates, ns):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{wr:.1%}\n(n={n})", ha="center", va="bottom", fontsize=10, fontweight="bold")
axes[0].axhline(0.5, color="red", linestyle=":", linewidth=1.2, label="Random (50%)")
axes[0].set_ylim(0, 1.1)
axes[0].set_ylabel("M1 Win Rate")
axes[0].set_title("Win Rate: M1 Baseline vs M2 Gate")
axes[0].legend()

bars2 = axes[1].bar(labels, avg_rets, color=colors_bar, edgecolor="white", width=0.5)
for bar, ret in zip(bars2, avg_rets):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.02 if ret >= 0 else -0.06),
                 f"{ret:+.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_ylabel("Avg Monthly Net Return (%)")
axes[1].set_title("Avg Return: M1 Baseline vs M2 Gate")

plt.tight_layout()
fig.savefig(OUT_DIR / "2_approve_reject_performance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[Chart 2] Approve/reject performance saved.")


# ---------------------------------------------------------------------------
# Chart 3: Cumulative return — M1 baseline vs M2-filtered
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 5))
fig.suptitle("M2 Walk-Forward: Cumulative Return (Out-of-Sample)", fontsize=13, fontweight="bold")

preds_sorted = preds.sort_values("date").reset_index(drop=True)

# M1 baseline: invest in every OOS event
m1_cum = (1 + preds_sorted["meta_target_return"]).cumprod()

# M2 approved: invest only when M2 says yes, else return 0 (sit in cash)
m2_returns = preds_sorted["meta_target_return"].where(preds_sorted["m2_approve"] == 1, 0.0)
m2_cum = (1 + m2_returns).cumprod()

dates = pd.to_datetime(preds_sorted["date"])

ax.plot(dates, m1_cum, label=f"M1 Baseline (n={len(preds)})", color="#3498db", linewidth=2)
ax.plot(dates, m2_cum, label=f"M2 Approved only (n={len(approved)})", color="#2ecc71", linewidth=2)
ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)

# shade rejected events
for _, row in preds_sorted[preds_sorted["m2_approve"] == 0].iterrows():
    ax.axvspan(row["date"], row["date"] + pd.Timedelta(days=20), alpha=0.08, color="#e74c3c")

ax.set_ylabel("Cumulative Growth ($1 invested)")
ax.set_xlabel("Date")
ax.set_title(f"OOS period: {dates.iloc[0].date()} → {dates.iloc[-1].date()}")
ax.legend()

plt.tight_layout()
fig.savefig(OUT_DIR / "3_cumulative_return.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[Chart 3] Cumulative return chart saved.")

# Confusion matrix printout
print(f"\nConfusion matrix (rows=actual, cols=predicted):")
cm = confusion_matrix(y_true, y_pred)
print(pd.DataFrame(cm, index=["Actual Loss", "Actual Win"], columns=["Pred Reject", "Pred Approve"]).to_string())

print(f"\n{'':=<60}")
print(f"All outputs in: {OUT_DIR.relative_to(ROOT)}/")
print(f"{'':=<60}")
