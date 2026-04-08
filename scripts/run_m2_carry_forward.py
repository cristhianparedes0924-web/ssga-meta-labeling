"""
Corrected M2 binary gate backtest using carry-forward allocation.

When M2 rejects a trade, the portfolio holds the PREVIOUS allocation
and earns the actual market return of that allocation — not 0%.

Compares:
  1. M1 baseline          : all 78 events at M1's weights
  2. M2 binary (0% wrong) : rejected months earn 0%  [old incorrect assumption]
  3. M2 binary (carry fwd): rejected months earn prev allocation's actual return
  4. M2 position sizing   : all 78 events, size = m2_prob / mean(m2_prob)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from metalabel.secondary.model import M2_FEATURES_CORE, apply_position_sizing, compute_carry_returns, run_walk_forward

ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "reports" / "results" / "secondary_dataset.csv"
OUT_DIR   = ROOT / "reports" / "assets" / "m2_carry_fwd"
METRICS   = ROOT / "reports" / "results" / "m2_carry_fwd_metrics.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load secondary dataset ────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["date", "realized_date"])
df = df.sort_values("date").reset_index(drop=True)

# ── Load per-asset monthly returns ────────────────────────────────────────────
def load_asset(name):
    path = ROOT / "data" / "clean" / f"{name}.csv"
    a = pd.read_csv(path, parse_dates=["Date"])
    a = a.set_index("Date")["Return"].rename(name)
    return a

spx  = load_asset("spx")
bcom = load_asset("bcom")
corp = load_asset("corp_bonds")
tsy  = load_asset("treasury_10y")
asset_rets = pd.concat([spx, bcom, corp, tsy], axis=1).sort_index()

# ── Map each event's realized_date to asset returns ───────────────────────────
# meta_target_return at row t = weights_t · asset_returns_at_realized_date_t
def get_asset_returns_at(realized_date):
    """Return asset return vector for the period realized at realized_date."""
    if realized_date in asset_rets.index:
        return asset_rets.loc[realized_date]
    # fallback: nearest date
    idx = asset_rets.index.get_indexer([realized_date], method="nearest")[0]
    return asset_rets.iloc[idx]

# ── Walk-forward M2 predictions ───────────────────────────────────────────────
preds = run_walk_forward(df, min_train_size=60, threshold=0.5,
                         features=M2_FEATURES_CORE)
preds = preds.sort_values("date").reset_index(drop=True)

# Merge weight columns and realized_date from the secondary dataset
w_merge = df[["date", "realized_date", "weight_spx", "weight_bcom",
              "weight_treasury_10y", "weight_corp_bonds"]]
preds = preds.merge(w_merge, on="date", how="left")

# Build asset return matrix aligned to the 78 OOS events (using their realized_date)
asset_ret_oos = pd.DataFrame(
    [get_asset_returns_at(rd) for rd in preds["realized_date"]],
    columns=["spx", "bcom", "corp_bonds", "treasury_10y"],
    index=preds.index
)

# ── Build return streams ──────────────────────────────────────────────────────
m1_returns = preds["meta_target_return"].values
threshold  = 0.51
approved   = (preds["m2_prob"] >= threshold).values

# Weight columns in preds (carried from secondary dataset)
w_cols = ["weight_spx", "weight_bcom", "weight_treasury_10y", "weight_corp_bonds"]
weights = preds[w_cols].values  # shape (78, 4)  — M1's weights for each event

# Asset return matrix aligned to preds (78 x 4)
A = asset_ret_oos[["spx", "bcom", "corp_bonds", "treasury_10y"]].values

# 1. M1 baseline
m1_stream = m1_returns.copy()

# 2. Old binary gate — rejected earns 0 (WRONG)
binary_wrong = np.where(approved, m1_returns, 0.0)

# 3. Correct binary gate — rejected earns prev allocation · current asset rets
carry_stream = np.zeros(len(preds))
for i in range(len(preds)):
    if approved[i]:
        carry_stream[i] = m1_returns[i]
    else:
        if i == 0:
            # No previous allocation — use M1 return as fallback
            carry_stream[i] = m1_returns[i]
        else:
            prev_w = weights[i - 1]   # previous event's M1 weights
            curr_a = A[i]             # current period's asset returns
            # map: spx, bcom, treasury_10y, corp_bonds
            carry_stream[i] = (prev_w[0] * curr_a[0] +   # spx
                               prev_w[1] * curr_a[1] +   # bcom
                               prev_w[2] * curr_a[3] +   # treasury_10y (col 3 in A)
                               prev_w[3] * curr_a[2])    # corp_bonds   (col 2 in A)

# 4. Position sizing (correct: unallocated fraction earns prev allocation return)
carry_rets  = compute_carry_returns(preds, asset_rets)
sized       = apply_position_sizing(preds, normalize=True, carry_returns=carry_rets)
norm_stream = sized["sized_return"].values

# ── Stats ─────────────────────────────────────────────────────────────────────
def stats(portfolio, benchmark):
    mu     = np.mean(portfolio)
    std    = np.std(portfolio)
    sharpe = mu / std * np.sqrt(12) if std > 0 else np.nan
    ann    = (1 + mu) ** 12 - 1
    active = portfolio - benchmark
    te     = np.std(active)
    ir     = np.mean(active) / te * np.sqrt(12) if te > 0 else 0.0
    return ann, sharpe, ir

m1_ann,  m1_sh,  m1_ir  = stats(m1_stream,    m1_stream)
bw_ann,  bw_sh,  bw_ir  = stats(binary_wrong, m1_stream)
cf_ann,  cf_sh,  cf_ir  = stats(carry_stream, m1_stream)
sz_ann,  sz_sh,  sz_ir  = stats(norm_stream,  m1_stream)

n_approved = approved.sum()
n_rejected = (~approved).sum()

# ── Print ─────────────────────────────────────────────────────────────────────
print(f"\n{'':=<76}")
print(f"{'M2 BINARY GATE: CARRY-FORWARD vs ZERO-RETURN COMPARISON':^76}")
print(f"{'':=<76}")
print(f"Threshold: {threshold}  |  Approved: {n_approved}  |  Held (carry fwd): {n_rejected}")
print(f"\n{'Setup':<38} {'Ann Return':>10}  {'Sharpe':>7}  {'Info Ratio':>11}")
print("-" * 76)
print(f"{'M1 baseline (no filter)':<38} {m1_ann:>9.2%}  {m1_sh:>7.3f}  {m1_ir:>+11.3f}")
print(f"{'M2 binary (wrong: rejected = 0%)':<38} {bw_ann:>9.2%}  {bw_sh:>7.3f}  {bw_ir:>+11.3f}")
print(f"{'M2 binary (correct: carry fwd)':<38} {cf_ann:>9.2%}  {cf_sh:>7.3f}  {cf_ir:>+11.3f}")
print(f"{'M2 position sizing (norm)':<38} {sz_ann:>9.2%}  {sz_sh:>7.3f}  {sz_ir:>+11.3f}")
print()

diff_ann = cf_ann - bw_ann
diff_ir  = cf_ir  - bw_ir
print(f"Carry-fwd vs zero-return:  Ann Return {diff_ann:+.2%}  |  IR {diff_ir:+.3f}")
print(f"{'':=<76}")

# ── Save metrics ──────────────────────────────────────────────────────────────
metrics = {
    "threshold": threshold,
    "n_approved": int(n_approved),
    "n_held_carry_fwd": int(n_rejected),
    "results": {
        "m1_baseline":         {"ann_return": round(m1_ann,4), "sharpe": round(m1_sh,3), "ir": round(m1_ir,3)},
        "m2_binary_wrong_0pct":{"ann_return": round(bw_ann,4), "sharpe": round(bw_sh,3), "ir": round(bw_ir,3)},
        "m2_binary_carry_fwd": {"ann_return": round(cf_ann,4), "sharpe": round(cf_sh,3), "ir": round(cf_ir,3)},
        "m2_sizing_norm":      {"ann_return": round(sz_ann,4), "sharpe": round(sz_sh,3), "ir": round(sz_ir,3)},
    }
}
with open(METRICS, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved -> {METRICS.relative_to(ROOT)}")

# ── Chart ─────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dates = pd.to_datetime(preds["date"])

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("M2 Binary Gate: Carry-Forward vs Zero-Return Assumption",
             fontsize=13, fontweight="bold")

# Left: cumulative returns
ax = axes[0]
streams = {
    "M1 Baseline":                    (m1_stream,    "#3498db", "-",  2.5),
    "M2 Binary — wrong (0% held)":    (binary_wrong, "#e74c3c", "--", 1.8),
    "M2 Binary — correct (carry fwd)":(carry_stream, "#f39c12", "-",  2.0),
    "M2 Position Sizing":             (norm_stream,  "#2ecc71", "-",  2.2),
}
for label, (s, col, ls, lw) in streams.items():
    ax.plot(dates, (1 + s).cumprod(), label=label, color=col, linestyle=ls, linewidth=lw)
ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8)
ax.set_title("Cumulative Return")
ax.set_ylabel("Growth of $1")
ax.legend(fontsize=8.5)
ax.grid(alpha=0.2)

# Right: bar comparison
ax2 = axes[1]
labels   = ["M1\nBaseline", "Binary\n(0% held)", "Binary\n(carry fwd)", "Position\nSizing"]
ann_rets = [m1_ann, bw_ann, cf_ann, sz_ann]
colors   = ["#3498db", "#e74c3c", "#f39c12", "#2ecc71"]
bars = ax2.bar(labels, [r * 100 for r in ann_rets], color=colors, width=0.5, edgecolor="white")
for bar, val in zip(bars, ann_rets):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.1,
             f"{val:.2%}", ha="center", fontsize=9, fontweight="bold")
ax2.set_title("Annualised Return Comparison")
ax2.set_ylabel("Ann. Return (%)")
ax2.grid(alpha=0.2)

plt.tight_layout()
out = OUT_DIR / "carry_fwd_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Chart saved -> {out.relative_to(ROOT)}")
