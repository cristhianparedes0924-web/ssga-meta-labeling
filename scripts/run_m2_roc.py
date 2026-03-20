"""M2 ROC Curve: Compare v1 (22 features) vs v2 (12 core features).

Generates a single chart showing:
  - ROC curve for M2 v1 (all 22 features including VIX/OAS)
  - ROC curve for M2 v2 (12 core features, no VIX/OAS)
  - Random classifier diagonal (AUC = 0.50)
  - AUC annotation for each curve

Run from repo root:
    python scripts/run_m2_roc.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parent.parent
V1_PREDS  = ROOT / "reports" / "results" / "m2_predictions.csv"
V2_PREDS  = ROOT / "reports" / "results" / "m2_v2_predictions.csv"
OUT_DIR   = ROOT / "reports" / "assets" / "m2_roc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load predictions
# ---------------------------------------------------------------------------
v1 = pd.read_csv(V1_PREDS)
v2 = pd.read_csv(V2_PREDS)

y_true_v1 = v1["meta_label"].values
y_prob_v1 = v1["m2_prob"].values

y_true_v2 = v2["meta_label"].values
y_prob_v2 = v2["m2_prob"].values

auc_v1 = roc_auc_score(y_true_v1, y_prob_v1)
auc_v2 = roc_auc_score(y_true_v2, y_prob_v2)

fpr_v1, tpr_v1, _ = roc_curve(y_true_v1, y_prob_v1)
fpr_v2, tpr_v2, _ = roc_curve(y_true_v2, y_prob_v2)

print(f"M2 v1 ROC-AUC: {auc_v1:.4f}  (22 features, incl. VIX/OAS)")
print(f"M2 v2 ROC-AUC: {auc_v2:.4f}  (12 core features, no VIX/OAS)")

# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor("#0F1C2C")
ax.set_facecolor("#19293E")

ax.plot(fpr_v1, tpr_v1,
        label=f"M2 v1 — 22 features incl. VIX/OAS  (AUC = {auc_v1:.3f})",
        color="#e74c3c", linewidth=2.2)

ax.plot(fpr_v2, tpr_v2,
        label=f"M2 v2 — 12 core features, no VIX/OAS  (AUC = {auc_v2:.3f})",
        color="#2ecc71", linewidth=2.2)

ax.plot([0, 1], [0, 1],
        label="Random classifier  (AUC = 0.500)",
        color="#95a5a6", linewidth=1.2, linestyle="--")

# Shade area under v2 curve slightly
ax.fill_between(fpr_v2, tpr_v2, alpha=0.06, color="#2ecc71")

# Annotations: mark the 0.51 operating point on v2
v2_thresholds = roc_curve(y_true_v2, y_prob_v2)[2]
fpr_v2_arr, tpr_v2_arr, thr_v2_arr = roc_curve(y_true_v2, y_prob_v2)
idx = np.argmin(np.abs(thr_v2_arr - 0.51))
ax.scatter(fpr_v2_arr[idx], tpr_v2_arr[idx],
           color="#f39c12", s=80, zorder=5,
           label=f"v2 operating point t=0.51  (FPR={fpr_v2_arr[idx]:.2f}, TPR={tpr_v2_arr[idx]:.2f})")

ax.set_xlabel("False Positive Rate  (1 - Specificity)", color="#C8D2DC", fontsize=11)
ax.set_ylabel("True Positive Rate  (Sensitivity / Recall)", color="#C8D2DC", fontsize=11)
ax.set_title("M2 ROC Curve: v1 vs v2\n"
             "Can the model distinguish winning M1 trades from losing ones?",
             color="white", fontsize=12, fontweight="bold", pad=14)

ax.tick_params(colors="#8C9BAA")
for spine in ax.spines.values():
    spine.set_edgecolor("#19293E")

legend = ax.legend(fontsize=9, loc="lower right",
                   facecolor="#19293E", edgecolor="#3C5070",
                   labelcolor="#C8D2DC")

ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
ax.grid(alpha=0.15, color="#C8D2DC")

# Text box: AUC interpretation
interp = (
    "AUC = 0.50: no better than random\n"
    "AUC = 0.55: slight edge\n"
    "AUC > 0.60: practically useful"
)
ax.text(0.40, 0.12, interp,
        transform=ax.transAxes,
        fontsize=8, color="#8C9BAA",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#19293E",
                  edgecolor="#3C5070", alpha=0.9))

plt.tight_layout()
out_path = OUT_DIR / "roc_v1_vs_v2.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\nROC chart saved -> {out_path.relative_to(ROOT)}")
