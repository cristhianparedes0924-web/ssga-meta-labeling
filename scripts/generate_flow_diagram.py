"""Generate a clean 1-page flow diagram showing M1 + M2 process."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports" / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor("#0F1C2C")
ax.set_facecolor("#0F1C2C")
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")

# ── helpers ──────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, text, bg, tc="white", fs=9, bold=False, radius=0.2):
    patch = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle=f"round,pad=0.05,rounding_size={radius}",
                           facecolor=bg, edgecolor="#2A4060", linewidth=1.2, zorder=3)
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal",
            zorder=4, multialignment="center")

def arrow(ax, x1, y1, x2, y2, color="#4A90D9", lw=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=14),
                zorder=5)

def label(ax, x, y, text, color="#8CA0B8", fs=7.5):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=color, zorder=4, style="italic")

# ── title ─────────────────────────────────────────────────────────────────────
ax.text(7, 9.55, "How M1 + M2 Turn $100 Into a Trade",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color="white", zorder=4)
ax.text(7, 9.15, "M1 decides WHAT to buy  |  M2 decides HOW MUCH to buy",
        ha="center", va="center", fontsize=9, color="#8CA0B8", zorder=4)

# ── Step 0: Starting capital ──────────────────────────────────────────────────
box(ax, 7, 8.45, 2.2, 0.6, "$100\nStarting Capital", "#1A3A5C", tc="#00C4A8", fs=10, bold=True)

arrow(ax, 7, 8.15, 7, 7.65)

# ── Step 1: 4 asset classes ───────────────────────────────────────────────────
ax.text(7, 7.5, "STEP 1 — M1 reads price data from 4 asset classes",
        ha="center", va="center", fontsize=8.5, color="#00C4A8",
        fontweight="bold", zorder=4)

assets = [
    (2.8,  6.85, "S&P 500\n(SPX)",        "#1E4D2B"),
    (5.1,  6.85, "Commodities\n(BCOM)",    "#1E3A4D"),
    (8.9,  6.85, "Corp Bonds",             "#2D2A1E"),
    (11.2, 6.85, "10Y Treasury",           "#2A1E2D"),
]
for x, y, txt, bg in assets:
    box(ax, x, y, 2.0, 0.65, txt, bg, fs=8.5)

# arrows from assets down to indicators
for x, _, _, _ in assets:
    arrow(ax, x, 6.52, x, 6.12, color="#3A6090")

# ── Step 2: 6 indicators ──────────────────────────────────────────────────────
ax.text(7, 5.98, "STEP 2 — M1 computes 6 macro indicators (z-scored)",
        ha="center", va="center", fontsize=8.5, color="#00C4A8",
        fontweight="bold", zorder=4)

inds = [
    (1.5,  5.35, "SPX Trend\nz-score"),
    (3.5,  5.35, "BCOM Trend\nz-score"),
    (5.5,  5.35, "Credit vs\nRates z"),
    (7.5,  5.35, "Risk\nBreadth z"),
    (9.5,  5.35, "BCOM\nAccel z"),
    (11.5, 5.35, "Yield\nMom z"),
]
for x, y, txt in inds:
    box(ax, x, y, 1.75, 0.65, txt, "#1A2E42", fs=7.5)

# convergence arrows to composite score
for x, _, _ in inds:
    arrow(ax, x, 5.02, 7, 4.42, color="#3A6090")

# ── Step 3: Composite score + signal ─────────────────────────────────────────
ax.text(7, 4.28, "STEP 3 — Weighted composite score  ->  BUY / SELL signal",
        ha="center", va="center", fontsize=8.5, color="#00C4A8",
        fontweight="bold", zorder=4)

box(ax, 5.2, 3.78, 2.6, 0.72,
    "Composite Score\n(Spearman IC weighted)", "#193050", fs=8)
arrow(ax, 6.52, 3.78, 7.48, 3.78, color="#4A90D9")
box(ax, 8.8, 3.78, 2.4, 0.72,
    "Score > 0.31  ->  BUY\nScore < -0.31 -> SELL", "#193050", fs=8)

arrow(ax, 7, 3.42, 7, 2.98)

# ── Step 4: M1 fixed allocation ───────────────────────────────────────────────
ax.text(7, 2.84, "STEP 4 — M1 maps signal to fixed asset weights",
        ha="center", va="center", fontsize=8.5, color="#00C4A8",
        fontweight="bold", zorder=4)

# BUY box
box(ax, 3.8, 2.28, 3.8, 0.75,
    "BUY  ->  SPX 40%  |  BCOM 15%  |  Corp Bonds 45%  |  Treasury 0%",
    "#1E3D1E", tc="#2ecc71", fs=8)
# SELL box
box(ax, 10.2, 2.28, 3.8, 0.75,
    "SELL  ->  SPX 5%  |  BCOM 0%  |  Corp Bonds 35%  |  Treasury 60%",
    "#3D1E1E", tc="#e74c3c", fs=8)

arrow(ax, 5.7, 2.28, 6.3, 1.75, color="#2ecc71")
arrow(ax, 8.3, 2.28, 7.7, 1.75, color="#e74c3c")

# ── Step 5: M2 sizing ─────────────────────────────────────────────────────────
ax.text(7, 1.62, "STEP 5 — M2 reads M1's state and scales the trade",
        ha="center", va="center", fontsize=8.5, color="#FFC432",
        fontweight="bold", zorder=4)

box(ax, 7, 1.15, 8.5, 0.72,
    "M2 logistic regression  ->  probability = 0.73     size = 0.73 / 0.625 = 1.17x",
    "#2A2010", tc="#FFC432", fs=8.5)

arrow(ax, 7, 0.79, 7, 0.44)

# ── Final result ──────────────────────────────────────────────────────────────
box(ax, 7, 0.26, 9.5, 0.42,
    "SPX: $46.80   |   BCOM: $17.55   |   Corp Bonds: $52.65   |   Treasury: $0",
    "#0A2010", tc="#2ecc71", fs=9, bold=True)

# ── Side note on M2 sizing logic ─────────────────────────────────────────────
ax.text(13.6, 1.62, "M2 sizing rule:", ha="right", va="center",
        fontsize=7.5, color="#8CA0B8", zorder=4)
ax.text(13.6, 1.38, "size = prob / avg_prob", ha="right", va="center",
        fontsize=7.5, color="#FFC432", zorder=4, style="italic")
ax.text(13.6, 1.14, "avg = 0.625 across all months", ha="right", va="center",
        fontsize=7, color="#8CA0B8", zorder=4)
ax.text(13.6, 0.88, "0.80 -> 1.28x  (trade more)", ha="right", va="center",
        fontsize=7, color="#2ecc71", zorder=4)
ax.text(13.6, 0.66, "0.55 -> 0.88x  (trade less)", ha="right", va="center",
        fontsize=7, color="#e74c3c", zorder=4)

# ── footer ────────────────────────────────────────────────────────────────────
ax.text(7, 0.04, "SSGA Meta-Labeling  |  Brandeis MSF 2026  |  CONFIDENTIAL",
        ha="center", va="center", fontsize=7, color="#3A5070", zorder=4)

plt.tight_layout(pad=0.3)
out = OUT_DIR / "m1_m2_flow_diagram.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out}")
