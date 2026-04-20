"""Generate M2 Refinement Journey PDF - 5 pages, one per refinement step.

Pages:
  1. Starting Point  - position sizing baseline
  2. Ridge Regression - worked, improves all metrics
  3. Position Size Clipping - did not improve
  4. Regime Conditioning - did not improve
  5. Summary - current best state

Run from repo root:
    python scripts/generate_refinement_journey.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "reports" / "results" / "M2_Refinement_Journey.pdf"

# ---------------------------------------------------------------------------
# Color palette  (warm beige / parchment)
# ---------------------------------------------------------------------------
BG    = "#F4EFE6"   # page background
CARD  = "#EAE3D5"   # card / panel background
DARK  = "#2C2416"   # primary text
MID   = "#5C4F3A"   # secondary text / labels
LINE  = "#C8B99A"   # grid / divider lines
GREEN = "#4A7A50"   # positive / worked
RED   = "#9A4A38"   # negative / did not work
BLUE  = "#3A5F7A"   # neutral highlight
GOLD  = "#8A6E30"   # accent / M1 baseline
TEAL  = "#3A7068"   # second highlight


def _setup_page(fig):
    fig.patch.set_facecolor(BG)


def _card(ax, facecolor=None):
    if facecolor is None:
        facecolor = CARD
    ax.set_facecolor(facecolor)
    for sp in ax.spines.values():
        sp.set_edgecolor(LINE)
    ax.tick_params(colors=MID, labelsize=8)


def _hdr(fig, title: str, subtitle: str, step_label: str):
    # Step label on its own line at the very top
    fig.text(0.50, 0.985, step_label, fontsize=8.5, color=MID,
             fontfamily="monospace", ha="center", va="top")
    # Title on next line, smaller to avoid clipping
    fig.text(0.50, 0.965, title, fontsize=13, fontweight="bold",
             color=DARK, ha="center", va="top")
    # Subtitle below title
    fig.text(0.50, 0.945, subtitle, fontsize=8.5, color=MID,
             ha="center", va="top", style="italic")


def _metric_box(ax, label, val, color=DARK, fontsize=13):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.62, val, fontsize=fontsize,
            fontweight="bold", color=color, ha="center", va="center")
    ax.text(0.5, 0.25, label, fontsize=8, color=MID, ha="center", va="center")


def _bar_group(ax, groups, values_list, colors, labels=None, ylabel=""):
    x = np.arange(len(groups))
    n = len(values_list)
    w = 0.65 / n
    for i, (vals, col) in enumerate(zip(values_list, colors)):
        offset = (i - (n - 1) / 2) * w
        ax.bar(x + offset, vals, w, color=col, alpha=0.88,
               edgecolor=BG, linewidth=0.5)
    if labels:
        ax.legend(labels, fontsize=7.5, facecolor=CARD, edgecolor=LINE,
                  labelcolor=MID)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, color=MID, fontsize=8)
    ax.set_ylabel(ylabel, color=MID, fontsize=8)
    ax.axhline(0, color=LINE, lw=0.8)
    _card(ax)
    ax.grid(axis="y", alpha=0.3, color=LINE)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
M1   = dict(ann=0.0953, sh=1.310, ir=0.000)
PS   = dict(ann=0.1055, sh=1.384, ir=0.213)   # baseline position sizing (expanding mean)
BS   = dict(ann=0.1055, sh=1.384, ir=0.213, auc=0.5252)  # same, for ridge comparison
RG   = dict(ann=0.1196, sh=1.491, ir=0.384, auc=0.5284)  # ridge CV final (expanding mean)
CLIP = dict(ann=0.1049, sh=1.373, ir=0.207)
REGIME = dict(ann=0.1033, sh=1.360, ir=0.191)

C_OPTS  = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
C_FREQS = [4,    0,    0,   2,   5,   34,  33]


# ---------------------------------------------------------------------------
# Page 1: Starting Point
# ---------------------------------------------------------------------------
def page1(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    _setup_page(fig)
    _hdr(fig, "M2 Position Sizing: The Starting Point",
         "Walk-forward logistic regression probability directly scales trade size",
         "STEP 0 / 3   -   BASELINE")

    gs = fig.add_gridspec(3, 4, left=0.06, right=0.97,
                          top=0.88, bottom=0.07,
                          hspace=0.55, wspace=0.45)

    # -- Formula card
    ax_f = fig.add_subplot(gs[0, :3])
    ax_f.set_facecolor(CARD); ax_f.axis("off")
    for sp in ax_f.spines.values(): sp.set_edgecolor(LINE)
    ax_f.text(0.02, 0.80, "Sizing Formula:", fontsize=9.5, color=MID,
              fontweight="bold", transform=ax_f.transAxes)
    ax_f.text(0.02, 0.45,
              "  position_size  =  m2_prob  /  mean(m2_prob)    [normalised, avg = 1.0x]",
              fontsize=11, color=DARK, fontfamily="monospace",
              transform=ax_f.transAxes)
    ax_f.text(0.02, 0.10,
              "  sized_return  =  size * M1_return  +  (1 - size) * carry_return",
              fontsize=11, color=DARK, fontfamily="monospace",
              transform=ax_f.transAxes)

    # -- Key idea card
    ax_k = fig.add_subplot(gs[0, 3])
    ax_k.set_facecolor("#DDD6C5"); ax_k.axis("off")
    for sp in ax_k.spines.values(): sp.set_edgecolor(LINE)
    ax_k.text(0.5, 0.70, "Key Idea", fontsize=9, fontweight="bold",
              color=DARK, ha="center", transform=ax_k.transAxes)
    ax_k.text(0.5, 0.35,
              "High confidence = bigger bet\nLow confidence = smaller bet\nMean size stays 1.0x",
              fontsize=8.5, color=MID, ha="center", va="center",
              transform=ax_k.transAxes, linespacing=1.6)

    # -- Metric boxes
    metrics = [
        ("Ann. Return", f"{PS['ann']:.1%}", GREEN),
        ("Sharpe",      f"{PS['sh']:.3f}",  BLUE),
        ("Info Ratio",  f"{PS['ir']:+.3f}", TEAL),
        ("OOS AUC",     "0.525",            GOLD),
    ]
    for col, (lbl, val, col_c) in enumerate(metrics):
        ax_m = fig.add_subplot(gs[1, col])
        ax_m.set_facecolor(CARD)
        for sp in ax_m.spines.values(): sp.set_edgecolor(LINE)
        _metric_box(ax_m, lbl, val, color=col_c, fontsize=14)

    # -- Fixed vs dynamic sizing: 10 example months
    ax_s = fig.add_subplot(gs[2, :2])
    _card(ax_s)
    _sc   = pd.read_csv(ROOT / "reports/results/m2_v2_predictions.csv",
                        parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    # 5 illustrative scenarios
    scenario_labels = ["Low\nConfidence", "Medium\nConfidence", "Full\nSize", "High\nConfidence", "Very High\nConfidence"]
    scenario_sizes  = [0.55, 0.78, 1.00, 1.32, 1.65]
    x = np.arange(5)
    w = 0.35
    ax_s.bar(x - w/2, np.ones(5), w, color=GOLD, alpha=0.85, edgecolor=BG)
    s_cols = [RED if s < 1.0 else (GOLD if s == 1.0 else GREEN) for s in scenario_sizes]
    ax_s.bar(x + w/2, scenario_sizes, w, color=s_cols, alpha=0.85, edgecolor=BG)
    for i, (sz, col) in enumerate(zip(scenario_sizes, s_cols)):
        ax_s.text(x[i] + w/2, sz + 0.03, f"{sz:.2f}x",
                  ha="center", fontsize=9, color=DARK, fontweight="bold")
    ax_s.axhline(1.0, color=DARK, lw=1.0, ls="--", alpha=0.4)
    ax_s.set_xticks(x)
    ax_s.set_xticklabels(scenario_labels, color=MID, fontsize=9)
    ax_s.set_ylabel("Position Size (x)", color=MID, fontsize=8)
    ax_s.set_ylim(0, 2.0)
    ax_s.set_title("M1 Fixed vs M2 Dynamic Sizing", color=DARK, fontsize=9, fontweight="bold")
    import matplotlib.patches as mpatches
    ax_s.legend(handles=[
        mpatches.Patch(color=GOLD,  label="M1 (always 1.0x)"),
        mpatches.Patch(color=GREEN, label="M2 sizes up"),
        mpatches.Patch(color=RED,   label="M2 sizes down"),
    ], fontsize=8, facecolor=CARD, edgecolor=LINE, labelcolor=MID)
    ax_s.grid(axis="y", alpha=0.3, color=LINE)

    # -- Vs M1 comparison
    ax_vs = fig.add_subplot(gs[2, 2:])
    _card(ax_vs)
    lbls = ["Ann Return", "Sharpe", "Info Ratio"]
    m1v  = [M1["ann"] * 100, M1["sh"], M1["ir"]]
    psv  = [PS["ann"] * 100, PS["sh"], PS["ir"]]
    _bar_group(ax_vs, lbls, [m1v, psv],
               [GOLD, GREEN], ["M1 Baseline", "M2 Position Sizing"], "Value")
    ax_vs.set_title("M2 Sizing vs M1 Baseline", color=DARK, fontsize=9,
                    fontweight="bold")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 2: Ridge Regression
# ---------------------------------------------------------------------------
def page2(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    _setup_page(fig)
    _hdr(fig, "Refinement 1: Ridge Regression (LogisticRegressionCV)",
         "Tune regularisation strength C via inner 5-fold CV at each walk-forward step",
         "STEP 1 / 3   -   WORKED")

    gs = fig.add_gridspec(3, 4, left=0.06, right=0.97,
                          top=0.88, bottom=0.07,
                          hspace=0.55, wspace=0.45)

    # -- Rationale card
    ax_r = fig.add_subplot(gs[0, :])
    ax_r.set_facecolor(CARD); ax_r.axis("off")
    for sp in ax_r.spines.values(): sp.set_edgecolor(LINE)
    lines = [
        ("Rationale:", MID, "bold", 9.5),
        ("With only 12 features and 60-100 training samples per step, a fixed C=0.5 may be too loose or too tight.", DARK, "normal", 9),
        ("LogisticRegressionCV searches Cs = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]", DARK, "normal", 9),
        ("and selects the best via ROC-AUC on inner 5-fold CV splits within each training window.", DARK, "normal", 9),
    ]
    y = 0.85
    for txt, col, wt, sz in lines:
        ax_r.text(0.02, y, txt, color=col, fontweight=wt, fontsize=sz,
                  transform=ax_r.transAxes)
        y -= 0.22

    # -- Metric comparison boxes
    m_labels = ["AUC", "Ann Return %", "Sharpe", "Info Ratio"]
    bs_vals  = [BS["auc"], BS["ann"] * 100, BS["sh"], BS["ir"]]
    rg_vals  = [RG["auc"], RG["ann"] * 100, RG["sh"], RG["ir"]]

    for i, (lbl, bv, rv) in enumerate(zip(m_labels, bs_vals, rg_vals)):
        ax_m = fig.add_subplot(gs[1, i])
        ax_m.set_facecolor(CARD)
        for sp in ax_m.spines.values(): sp.set_edgecolor(LINE)
        delta_col = GREEN if rv > bv else RED
        ax_m.text(0.5, 0.82, f"{rv:.4g}", fontsize=14, fontweight="bold",
                  color=delta_col, ha="center", transform=ax_m.transAxes)
        ax_m.text(0.5, 0.58, f"vs {bv:.4g} baseline", fontsize=7.5, color=MID,
                  ha="center", transform=ax_m.transAxes)
        diff = rv - bv
        sign = "+" if diff >= 0 else ""
        ax_m.text(0.5, 0.38, f"({sign}{diff:.4g})", fontsize=8,
                  color=delta_col, ha="center", transform=ax_m.transAxes,
                  fontweight="bold")
        ax_m.text(0.5, 0.14, lbl, fontsize=8, color=MID,
                  ha="center", transform=ax_m.transAxes)
        ax_m.axis("off")

    # -- C value distribution
    ax_c = fig.add_subplot(gs[2, :2])
    _card(ax_c)
    x_pos = np.arange(len(C_OPTS))
    bars = ax_c.bar(x_pos, C_FREQS, color=TEAL, alpha=0.85,
                    edgecolor=BG, linewidth=0.5)
    for bar, freq in zip(bars, C_FREQS):
        if freq > 0:
            ax_c.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + 0.4,
                      str(freq), ha="center", fontsize=8,
                      color=DARK, fontweight="bold")
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels([str(c) for c in C_OPTS], color=MID, fontsize=8)
    ax_c.set_title("C Values Chosen by CV (78 steps)", color=DARK,
                   fontsize=9, fontweight="bold")
    ax_c.set_xlabel("Regularisation C", color=MID, fontsize=8)
    ax_c.set_ylabel("Times selected", color=MID, fontsize=8)
    ax_c.grid(axis="y", alpha=0.3, color=LINE)

    # -- Bar comparison
    ax_b = fig.add_subplot(gs[2, 2:])
    _card(ax_b)
    lbls2 = ["Ann Ret %", "Sharpe", "Info Ratio"]
    m1v   = [M1["ann"] * 100, M1["sh"],  M1["ir"]]
    bsv   = [BS["ann"] * 100, BS["sh"],  BS["ir"]]
    rgv   = [RG["ann"] * 100, RG["sh"],  RG["ir"]]
    _bar_group(ax_b, lbls2, [m1v, bsv, rgv],
               [GOLD, BLUE, GREEN],
               ["M1 Baseline", "Baseline (C=0.5)", "Ridge CV"], "Value")
    ax_b.set_title("Performance Comparison", color=DARK, fontsize=9,
                   fontweight="bold")

    fig.text(0.5, 0.02,
             "VERDICT: Ridge CV improves all metrics.  "
             "AUC +0.0032  |  Ann Return +1.41pp  |  Sharpe +0.107  |  IR +0.171",
             ha="center", fontsize=9, color=GREEN, fontweight="bold",
             bbox=dict(facecolor=CARD, edgecolor=GREEN, linewidth=1.2,
                       boxstyle="round,pad=0.4"))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 3: Position Size Clipping
# ---------------------------------------------------------------------------
def page3(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    _setup_page(fig)
    _hdr(fig, "Refinement 2: Position Size Clipping",
         "Walk-forward p10/p90 bounds to cap extreme position sizes",
         "STEP 2 / 3   -   DID NOT IMPROVE")

    gs = fig.add_gridspec(3, 4, left=0.06, right=0.97,
                          top=0.88, bottom=0.07,
                          hspace=0.55, wspace=0.45)

    # -- Rationale
    ax_r = fig.add_subplot(gs[0, :])
    ax_r.set_facecolor(CARD); ax_r.axis("off")
    for sp in ax_r.spines.values(): sp.set_edgecolor(LINE)
    text_lines = [
        ("Rationale:", MID, "bold", 9.5),
        ("Position sizes range from 0.35x to 1.72x.  The largest sizes might just be noise - so we tested capping them.", DARK, "normal", 9),
        ("At each step we look at all sizes produced so far, find the typical range, and set a floor and ceiling from that history.", DARK, "normal", 9),
        ("The bounds update as more months accumulate, so we never use future data to set them.", DARK, "normal", 9),
    ]
    y = 0.85
    for txt, col, wt, sz in text_lines:
        ax_r.text(0.02, y, txt, color=col, fontweight=wt, fontsize=sz,
                  transform=ax_r.transAxes)
        y -= 0.22

    # -- Metric comparison boxes
    m_labels = ["Ann Return %", "Sharpe", "Info Ratio", "Clip Range"]
    ps_vals  = [PS["ann"] * 100,   PS["sh"],   PS["ir"],   "0.35x-1.72x"]
    cl_vals  = [CLIP["ann"] * 100, CLIP["sh"], CLIP["ir"], "0.68x-1.40x"]

    for i in range(4):
        ax_m = fig.add_subplot(gs[1, i])
        ax_m.set_facecolor(CARD)
        for sp in ax_m.spines.values(): sp.set_edgecolor(LINE)
        ax_m.axis("off")
        bv = ps_vals[i]; rv = cl_vals[i]
        if i < 3:
            delta = rv - bv
            delta_col = GREEN if delta > 0 else RED
            ax_m.text(0.5, 0.82, f"{rv:.4g}", fontsize=14, fontweight="bold",
                      color=delta_col, ha="center", transform=ax_m.transAxes)
            ax_m.text(0.5, 0.58, f"vs {bv:.4g} unclipped", fontsize=7.5,
                      color=MID, ha="center", transform=ax_m.transAxes)
            sign = "+" if delta >= 0 else ""
            ax_m.text(0.5, 0.38, f"({sign}{delta:.4g})", fontsize=8,
                      color=delta_col, ha="center", transform=ax_m.transAxes,
                      fontweight="bold")
        else:
            ax_m.text(0.5, 0.65, str(rv), fontsize=11, fontweight="bold",
                      color=BLUE, ha="center", transform=ax_m.transAxes)
            ax_m.text(0.5, 0.38, f"was: {bv}", fontsize=8, color=MID,
                      ha="center", transform=ax_m.transAxes)
        ax_m.text(0.5, 0.14, m_labels[i], fontsize=8, color=MID,
                  ha="center", transform=ax_m.transAxes)

    # -- Why it failed
    ax_why = fig.add_subplot(gs[2, :2])
    ax_why.set_facecolor(CARD); ax_why.axis("off")
    for sp in ax_why.spines.values(): sp.set_edgecolor(LINE)
    ax_why.text(0.05, 0.90, "Why Clipping Fails:", fontsize=9.5, color=MID,
                fontweight="bold", transform=ax_why.transAxes)
    reasons = [
        "- When we applied the cap: floor=0.68x, ceiling=1.40x",
        "- The months with the largest sizes were mostly M1 winners",
        "- Capping those months cut the gains that mattered most",
        "- The extreme sizes were carrying real signal, not noise",
    ]
    y = 0.72
    for r in reasons:
        ax_why.text(0.05, y, r, fontsize=8.5, color=DARK,
                    transform=ax_why.transAxes)
        y -= 0.18

    # -- Bar chart
    ax_b = fig.add_subplot(gs[2, 2:])
    _card(ax_b)
    lbls2 = ["Ann Ret %", "Sharpe", "Info Ratio"]
    m1v   = [M1["ann"] * 100,   M1["sh"],   M1["ir"]]
    psv   = [PS["ann"] * 100,   PS["sh"],   PS["ir"]]
    clv   = [CLIP["ann"] * 100, CLIP["sh"], CLIP["ir"]]
    _bar_group(ax_b, lbls2, [m1v, psv, clv],
               [GOLD, BLUE, RED],
               ["M1 Baseline", "No Clipping", "Clipped"], "Value")
    ax_b.set_title("Clipping vs No Clipping", color=DARK, fontsize=9,
                   fontweight="bold")

    fig.text(0.5, 0.02,
             "VERDICT: Clipping hurts performance.  Extreme position sizes carry signal.  Clipping abandoned.",
             ha="center", fontsize=9, color=RED, fontweight="bold",
             bbox=dict(facecolor=CARD, edgecolor=RED, linewidth=1.2,
                       boxstyle="round,pad=0.4"))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 4: Regime Conditioning
# ---------------------------------------------------------------------------
def page4(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    _setup_page(fig)
    _hdr(fig, "Refinement 3: Regime Conditioning",
         "Activate M2 only in high-stress regimes where it has more classification skill",
         "STEP 3 / 3   -   DID NOT IMPROVE")

    gs = fig.add_gridspec(3, 4, left=0.06, right=0.97,
                          top=0.88, bottom=0.07,
                          hspace=0.55, wspace=0.45)

    # -- Rationale
    ax_r = fig.add_subplot(gs[0, :])
    ax_r.set_facecolor(CARD); ax_r.axis("off")
    for sp in ax_r.spines.values(): sp.set_edgecolor(LINE)
    text_lines = [
        ("Hypothesis:", MID, "bold", 9.5),
        ("M2 AUC is higher in stress regimes: VIX High (z>0): 0.577, Very High VIX (z>1): 0.636, M1 Struggling: 0.583.", DARK, "normal", 9),
        ("Test: use M2 sizing only when VIX z-score > 0 (or > 1), otherwise revert to full M1 size = 1.0x.", DARK, "normal", 9),
        ("Problem: M2 still has mild skill in calm markets (AUC=0.498), and switching off M2 there loses the sizing benefit.", DARK, "normal", 9),
    ]
    y = 0.85
    for txt, col, wt, sz in text_lines:
        ax_r.text(0.02, y, txt, color=col, fontweight=wt, fontsize=sz,
                  transform=ax_r.transAxes)
        y -= 0.22

    # -- AUC by regime
    ax_auc = fig.add_subplot(gs[1, :2])
    _card(ax_auc)
    reg_labels = ["All OOS\n(0.525)", "Calm VIX\n(0.498)", "High VIX\n(0.577)", "Very High\n(0.636)", "M1 Struggling\n(0.583)"]
    aucs = [0.525, 0.498, 0.577, 0.636, 0.583]
    cols = [GOLD, MID, TEAL, GREEN, BLUE]
    bars = ax_auc.bar(range(len(aucs)), aucs, color=cols, alpha=0.85,
                      edgecolor=BG, linewidth=0.4)
    ax_auc.axhline(0.5, color=RED, lw=1.2, ls="--", label="Random (0.50)")
    for bar, v in zip(bars, aucs):
        ax_auc.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", fontsize=7.5,
                    color=DARK, fontweight="bold")
    ax_auc.set_xticks(range(len(aucs)))
    ax_auc.set_xticklabels(reg_labels, color=MID, fontsize=7)
    ax_auc.set_ylabel("OOS AUC", color=MID, fontsize=8)
    ax_auc.set_title("M2 AUC by Market Regime", color=DARK, fontsize=9,
                     fontweight="bold")
    ax_auc.set_ylim(0.45, 0.67)
    ax_auc.legend(fontsize=7.5, facecolor=CARD, edgecolor=LINE, labelcolor=MID)
    ax_auc.grid(axis="y", alpha=0.3, color=LINE)

    # -- Why it still fails
    ax_why = fig.add_subplot(gs[1, 2:])
    ax_why.set_facecolor(CARD); ax_why.axis("off")
    for sp in ax_why.spines.values(): sp.set_edgecolor(LINE)
    ax_why.text(0.05, 0.90, "Why Regime Conditioning Fails:", fontsize=9.5,
                color=MID, fontweight="bold", transform=ax_why.transAxes)
    reasons = [
        "- Calm markets have AUC ~ 0.50 (near random)",
        "- But M2 sizing still improves return in calm periods",
        "  by avoiding the worst-probability months",
        "- Switching off M2 there gives up those gains",
        "- Net effect: regime model is worse on Sharpe and Return",
    ]
    y = 0.72
    for r in reasons:
        ax_why.text(0.05, y, r, fontsize=8.5, color=DARK,
                    transform=ax_why.transAxes)
        y -= 0.15

    # -- Performance comparison
    ax_b = fig.add_subplot(gs[2, :])
    _card(ax_b)
    lbls3 = ["Ann Return %", "Sharpe", "Info Ratio"]
    m1v   = [M1["ann"] * 100,     M1["sh"],     M1["ir"]]
    psv   = [PS["ann"] * 100,     PS["sh"],     PS["ir"]]
    rgv   = [REGIME["ann"] * 100, REGIME["sh"], REGIME["ir"]]
    x = np.arange(3); w = 0.22
    ax_b.bar(x - w, m1v, w, color=GOLD, alpha=0.85, edgecolor=BG, label="M1 Baseline")
    ax_b.bar(x,     psv, w, color=BLUE, alpha=0.85, edgecolor=BG, label="Full M2 Sizing")
    ax_b.bar(x + w, rgv, w, color=RED,  alpha=0.85, edgecolor=BG, label="Regime-Conditioned")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(lbls3, color=MID, fontsize=9)
    ax_b.legend(fontsize=8.5, facecolor=CARD, edgecolor=LINE, labelcolor=MID)
    ax_b.axhline(0, color=LINE, lw=0.8)
    ax_b.set_title("Regime Conditioning vs Full M2 Sizing", color=DARK,
                   fontsize=9, fontweight="bold")
    ax_b.grid(axis="y", alpha=0.3, color=LINE)

    fig.text(0.5, 0.02,
             "VERDICT: Despite higher AUC in stress, regime conditioning reduces overall performance.  Approach abandoned.",
             ha="center", fontsize=9, color=RED, fontweight="bold",
             bbox=dict(facecolor=CARD, edgecolor=RED, linewidth=1.2,
                       boxstyle="round,pad=0.4"))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 5: Summary
# ---------------------------------------------------------------------------
def page5(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    _setup_page(fig)
    _hdr(fig, "M2 Refinement Summary: Where We Stand",
         "Final model = Ridge CV logistic regression + normalised position sizing",
         "SUMMARY   -   ALL 3 REFINEMENT STEPS")

    gs = fig.add_gridspec(3, 4, left=0.06, right=0.97,
                          top=0.88, bottom=0.07,
                          hspace=0.55, wspace=0.45)

    # -- Final metric boxes
    metrics = [
        ("Ann. Return",  f"{RG['ann']:.1%}",  GREEN),
        ("Sharpe Ratio", f"{RG['sh']:.3f}",   BLUE),
        ("Info Ratio",   f"{RG['ir']:+.3f}",  TEAL),
        ("OOS AUC",      f"{RG['auc']:.4f}",  GOLD),
    ]
    for i, (lbl, val, col) in enumerate(metrics):
        ax_m = fig.add_subplot(gs[0, i])
        ax_m.set_facecolor(CARD)
        for sp in ax_m.spines.values(): sp.set_edgecolor(LINE)
        _metric_box(ax_m, lbl, val, color=col, fontsize=14)

    # -- Journey table
    ax_jt = fig.add_subplot(gs[1, :])
    ax_jt.set_facecolor(CARD); ax_jt.axis("off")
    for sp in ax_jt.spines.values(): sp.set_edgecolor(LINE)

    ax_jt.text(0.5, 0.94, "Refinement Journey", fontsize=10,
               fontweight="bold", color=DARK, ha="center",
               transform=ax_jt.transAxes)

    headers = ["Refinement", "Hypothesis", "Verdict", "Ann Return", "Sharpe", "IR"]
    col_x   = [0.01, 0.18, 0.50, 0.68, 0.78, 0.88]
    for j, h in enumerate(headers):
        ax_jt.text(col_x[j], 0.80, h, fontsize=8.5, fontweight="bold",
                   color=MID, transform=ax_jt.transAxes)

    verdict_colors = {
        "Starting point": GOLD,
        "WORKED":         GREEN,
        "DID NOT WORK":   RED,
    }
    journey_rows = [
        ("Baseline",         "Prob-scaled position sizing",        "Starting point", f"{PS['ann']:.1%}", f"{PS['sh']:.3f}", f"{PS['ir']:+.3f}"),
        ("1. Ridge CV",      "Tune C via inner 5-fold CV",          "WORKED",          f"{RG['ann']:.1%}", f"{RG['sh']:.3f}", f"{RG['ir']:+.3f}"),
        ("2. Clip Sizes",    "Cap p10/p90 to remove noise",         "DID NOT WORK",    f"{CLIP['ann']:.1%}", f"{CLIP['sh']:.3f}", f"{CLIP['ir']:+.3f}"),
        ("3. Regime Cond.",  "Activate M2 only in VIX stress",      "DID NOT WORK",    f"{REGIME['ann']:.1%}", f"{REGIME['sh']:.3f}", f"{REGIME['ir']:+.3f}"),
    ]
    for k, row in enumerate(journey_rows):
        y_r = 0.62 - k * 0.155
        for j, val in enumerate(row):
            col_v = verdict_colors.get(val, DARK) if j == 2 else DARK
            fw = "bold" if j == 2 else "normal"
            ax_jt.text(col_x[j], y_r, val, fontsize=8, color=col_v,
                       fontweight=fw, transform=ax_jt.transAxes)

    # -- Start vs finish bar chart (full width)
    ax_b = fig.add_subplot(gs[2, :])
    _card(ax_b)
    lbls = ["Ann Ret %", "Sharpe", "Info Ratio"]
    m1v  = [M1["ann"] * 100, M1["sh"], M1["ir"]]
    psv  = [PS["ann"] * 100, PS["sh"], PS["ir"]]
    rgv  = [RG["ann"] * 100, RG["sh"], RG["ir"]]
    _bar_group(ax_b, lbls, [m1v, psv, rgv],
               [GOLD, BLUE, GREEN],
               ["M1 Baseline", "Initial M2 Sizing", "Final (Ridge CV)"], "Value")
    ax_b.set_title("Start vs Finish vs M1", color=DARK, fontsize=9,
                   fontweight="bold")

    fig.text(0.5, 0.02,
             "Final Model: Ridge CV logistic regression + normalised position sizing  "
             "|  Ann Return 11.96%  |  Sharpe 1.491  |  IR +0.384",
             ha="center", fontsize=9, color=DARK, fontweight="bold",
             bbox=dict(facecolor=CARD, edgecolor=GOLD, linewidth=1.5,
                       boxstyle="round,pad=0.4"))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
with PdfPages(str(OUT)) as pdf:
    print("Generating page 1: Starting Point...")
    page1(pdf)
    print("Generating page 2: Ridge Regression...")
    page2(pdf)
    print("Generating page 3: Position Size Clipping...")
    page3(pdf)
    print("Generating page 4: Regime Conditioning...")
    page4(pdf)
    print("Generating page 5: Summary...")
    page5(pdf)

print(f"\nPDF saved -> {OUT.relative_to(ROOT)}")
print(f"Pages: 5  |  Background: warm beige (#F4EFE6)")
