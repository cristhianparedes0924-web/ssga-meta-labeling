"""
EDA: Justification for VIX and Liquidity (OAS) as M2 meta-labeling features.

Produces 6 charts saved to reports/assets/secondary_eda/ and prints a
statistical summary. Run from repo root:

    python scripts/eda_m2_features.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "reports" / "results" / "secondary_dataset.csv"
OUT_DIR = ROOT / "reports" / "assets" / "secondary_eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "win":   "#2ecc71",
    "loss":  "#e74c3c",
    "low":   "#3498db",
    "high":  "#e67e22",
    "tight": "#2ecc71",
    "wide":  "#e74c3c",
    "neutral": "#95a5a6",
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

wins  = df[df["meta_label"] == 1]
losses = df[df["meta_label"] == 0]

print("=" * 60)
print("M2 FEATURE JUSTIFICATION — STATISTICAL SUMMARY")
print("=" * 60)
print(f"\nTotal M1 decision events : {len(df)}")
print(f"  BUY signals            : {(df['primary_signal']=='BUY').sum()}")
print(f"  SELL signals           : {(df['primary_signal']=='SELL').sum()}")
print(f"M1 overall win rate      : {df['meta_label'].mean():.1%}")
print(f"Date range               : {df['date'].min().date()} → {df['date'].max().date()}")


# ===========================================================================
# CHART 1: M1 Win Rate by VIX Regime and Signal Type
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Chart 1 — M1 Win Rate by VIX Regime", fontsize=14, fontweight="bold")

# Overall VIX regime win rates
for ax, signal in zip(axes, ["All signals", "BUY / SELL split"]):
    ax.set_title(signal)

# Left: overall
regime_stats = df.groupby("vix_high_regime")["meta_label"].agg(["mean", "count", "sum"])
regime_stats.index = ["Low VIX", "High VIX"]
colors_bar = [COLORS["low"], COLORS["high"]]
bars = axes[0].bar(regime_stats.index, regime_stats["mean"], color=colors_bar, width=0.5, edgecolor="white")
for bar, (_, row) in zip(bars, regime_stats.iterrows()):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{row['mean']:.1%}\n(n={int(row['count'])})", ha="center", va="bottom", fontsize=11, fontweight="bold")
axes[0].axhline(df["meta_label"].mean(), color="black", linestyle="--", linewidth=1.2, label=f"Overall: {df['meta_label'].mean():.1%}")
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel("M1 Win Rate")
axes[0].legend()
axes[0].set_title("All signals — Low vs High VIX")

# Right: split by signal type
for i, sig in enumerate(["BUY", "SELL"]):
    sub = df[df["primary_signal"] == sig]
    stats_sig = sub.groupby("vix_high_regime")["meta_label"].agg(["mean", "count"])
    stats_sig.index = ["Low VIX", "High VIX"]
    x = np.array([0, 1]) + i * 2.5
    b = axes[1].bar(x, stats_sig["mean"], color=colors_bar, width=0.7, edgecolor="white",
                    label=sig if i == 0 else None)
    for bar, (_, row) in zip(b, stats_sig.iterrows()):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{row['mean']:.1%}\n(n={int(row['count'])})", ha="center", va="bottom", fontsize=9)
axes[1].set_xticks([0, 1, 2.5, 3.5])
axes[1].set_xticklabels(["BUY\nLow VIX", "BUY\nHigh VIX", "SELL\nLow VIX", "SELL\nHigh VIX"])
axes[1].set_ylim(0, 1.15)
axes[1].set_ylabel("M1 Win Rate")
axes[1].axhline(df["meta_label"].mean(), color="black", linestyle="--", linewidth=1.2)
axes[1].set_title("BUY vs SELL by VIX regime")

low_n  = regime_stats.loc["Low VIX",  "count"]
high_n = regime_stats.loc["High VIX", "count"]
low_s  = regime_stats.loc["Low VIX",  "sum"]
high_s = regime_stats.loc["High VIX", "sum"]
chi2, p_vix = stats.chi2_contingency([[low_s, low_n - low_s], [high_s, high_n - high_s]])[:2]
fig.text(0.5, 0.01, f"Chi-square test (Low vs High VIX): p = {p_vix:.4f}", ha="center", fontsize=10,
         color="black" if p_vix > 0.05 else "red")

plt.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(OUT_DIR / "1_win_rate_by_vix_regime.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[Chart 1] VIX regime win rates:")
print(regime_stats[["mean", "count"]].rename(columns={"mean": "win_rate", "count": "n_events"}).to_string())
print(f"  Chi-square p-value: {p_vix:.4f} {'*** SIGNIFICANT' if p_vix < 0.05 else '(not significant at 5%)'}")


# ===========================================================================
# CHART 2: M1 Win Rate by OAS Regime and Signal Type
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Chart 2 — M1 Win Rate by OAS (Liquidity) Regime", fontsize=14, fontweight="bold")

oas_stats = df.groupby("oas_wide_regime")["meta_label"].agg(["mean", "count", "sum"])
oas_stats.index = ["Tight OAS", "Wide OAS"]
colors_oas = [COLORS["tight"], COLORS["wide"]]
bars = axes[0].bar(oas_stats.index, oas_stats["mean"], color=colors_oas, width=0.5, edgecolor="white")
for bar, (_, row) in zip(bars, oas_stats.iterrows()):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{row['mean']:.1%}\n(n={int(row['count'])})", ha="center", va="bottom", fontsize=11, fontweight="bold")
axes[0].axhline(df["meta_label"].mean(), color="black", linestyle="--", linewidth=1.2, label=f"Overall: {df['meta_label'].mean():.1%}")
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel("M1 Win Rate")
axes[0].legend()
axes[0].set_title("All signals — Tight vs Wide OAS")

for i, sig in enumerate(["BUY", "SELL"]):
    sub = df[df["primary_signal"] == sig]
    stats_sig = sub.groupby("oas_wide_regime")["meta_label"].agg(["mean", "count"])
    stats_sig.index = ["Tight OAS", "Wide OAS"]
    x = np.array([0, 1]) + i * 2.5
    b = axes[1].bar(x, stats_sig["mean"], color=colors_oas, width=0.7, edgecolor="white")
    for bar, (_, row) in zip(b, stats_sig.iterrows()):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{row['mean']:.1%}\n(n={int(row['count'])})", ha="center", va="bottom", fontsize=9)
axes[1].set_xticks([0, 1, 2.5, 3.5])
axes[1].set_xticklabels(["BUY\nTight OAS", "BUY\nWide OAS", "SELL\nTight OAS", "SELL\nWide OAS"])
axes[1].set_ylim(0, 1.15)
axes[1].set_ylabel("M1 Win Rate")
axes[1].axhline(df["meta_label"].mean(), color="black", linestyle="--", linewidth=1.2)
axes[1].set_title("BUY vs SELL by OAS regime")

tight_n = oas_stats.loc["Tight OAS", "count"]
wide_n  = oas_stats.loc["Wide OAS",  "count"]
tight_s = oas_stats.loc["Tight OAS", "sum"]
wide_s  = oas_stats.loc["Wide OAS",  "sum"]
chi2_oas, p_oas = stats.chi2_contingency([[tight_s, tight_n - tight_s], [wide_s, wide_n - wide_s]])[:2]
fig.text(0.5, 0.01, f"Chi-square test (Tight vs Wide OAS): p = {p_oas:.4f}", ha="center", fontsize=10,
         color="black" if p_oas > 0.05 else "red")

plt.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(OUT_DIR / "2_win_rate_by_oas_regime.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[Chart 2] OAS regime win rates:")
print(oas_stats[["mean", "count"]].rename(columns={"mean": "win_rate", "count": "n_events"}).to_string())
print(f"  Chi-square p-value: {p_oas:.4f} {'*** SIGNIFICANT' if p_oas < 0.05 else '(not significant at 5%)'}")


# ===========================================================================
# CHART 3: Joint VIX × OAS Regime — Win Rate Heatmap
# ===========================================================================
df["regime_label"] = df.apply(
    lambda r: ("Low VIX" if r["vix_high_regime"] == 0 else "High VIX") + " / " +
              ("Tight OAS" if r["oas_wide_regime"] == 0 else "Wide OAS"),
    axis=1,
)
joint = df.groupby(["vix_high_regime", "oas_wide_regime"])["meta_label"].agg(["mean", "count", "std"])
pivot_mean  = joint["mean"].unstack()
pivot_count = joint["count"].unstack()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Chart 3 — Joint VIX × OAS Regime: M1 Win Rate & Avg Return", fontsize=14, fontweight="bold")

# Win rate heatmap
im = axes[0].imshow(pivot_mean.values, cmap="RdYlGn", vmin=0.4, vmax=1.0)
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["Tight OAS\n(easy credit)", "Wide OAS\n(credit stress)"])
axes[0].set_yticks([0, 1])
axes[0].set_yticklabels(["Low VIX\n(calm market)", "High VIX\n(fearful market)"])
for i in range(2):
    for j in range(2):
        val = pivot_mean.values[i, j]
        n   = int(pivot_count.values[i, j])
        axes[0].text(j, i, f"{val:.1%}\nn={n}", ha="center", va="center",
                     fontsize=13, fontweight="bold",
                     color="white" if val < 0.55 or val > 0.85 else "black")
plt.colorbar(im, ax=axes[0], label="M1 Win Rate")
axes[0].set_title("M1 Win Rate by Joint Regime")

# Average return heatmap
pivot_ret = df.groupby(["vix_high_regime", "oas_wide_regime"])["meta_target_return"].mean().unstack()
im2 = axes[1].imshow(pivot_ret.values * 100, cmap="RdYlGn", vmin=-0.5, vmax=1.5)
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["Tight OAS\n(easy credit)", "Wide OAS\n(credit stress)"])
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(["Low VIX\n(calm market)", "High VIX\n(fearful market)"])
for i in range(2):
    for j in range(2):
        val = pivot_ret.values[i, j] * 100
        axes[1].text(j, i, f"{val:+.2f}%", ha="center", va="center",
                     fontsize=14, fontweight="bold",
                     color="white" if val < 0.1 else "black")
plt.colorbar(im2, ax=axes[1], label="Avg Monthly Return (%)")
axes[1].set_title("Avg M1 Monthly Return by Joint Regime")

plt.tight_layout()
fig.savefig(OUT_DIR / "3_joint_regime_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[Chart 3] Joint VIX × OAS regime win rates:")
print(pivot_mean.rename(index={0: "Low VIX", 1: "High VIX"},
                        columns={0: "Tight OAS", 1: "Wide OAS"}).round(3).to_string())
print(f"\n  Avg returns (%):")
print((pivot_ret * 100).rename(index={0: "Low VIX", 1: "High VIX"},
                                columns={0: "Tight OAS", 1: "Wide OAS"}).round(3).to_string())


# ===========================================================================
# CHART 4: Feature Correlation with meta_label
# ===========================================================================
feature_cols = [
    "composite_score",
    "spx_trend_z", "bcom_trend_z", "credit_vs_rates_z",
    "risk_breadth_z", "bcom_accel_z", "yield_mom_z",
    "trailing_hit_rate_12", "trailing_avg_net_return_12",
    "vix_level_z", "vix_change_z", "vix_trend", "vix_high_regime", "vix_rising",
    "oas_level_z", "oas_change_z", "oas_trend", "oas_wide_regime", "oas_widening",
]
feature_cols = [c for c in feature_cols if c in df.columns]

corrs = []
pvals = []
for col in feature_cols:
    valid = df[[col, "meta_label"]].dropna()
    r, p = stats.pointbiserialr(valid["meta_label"], valid[col])
    corrs.append(r)
    pvals.append(p)

corr_df = pd.DataFrame({"feature": feature_cols, "correlation": corrs, "pvalue": pvals})
corr_df = corr_df.sort_values("correlation", key=abs, ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
fig.suptitle("Chart 4 — Feature Correlation with M1 Win/Loss (meta_label)", fontsize=14, fontweight="bold")

bar_colors = []
for _, row in corr_df.iterrows():
    if "vix" in row["feature"] or "oas" in row["feature"]:
        bar_colors.append(COLORS["high"] if row["correlation"] < 0 else COLORS["low"])
    else:
        bar_colors.append("#7f8c8d")

bars = ax.barh(corr_df["feature"], corr_df["correlation"], color=bar_colors, edgecolor="white")
for bar, (_, row) in zip(bars, corr_df.iterrows()):
    sig = "**" if row["pvalue"] < 0.01 else ("*" if row["pvalue"] < 0.05 else "")
    ax.text(row["correlation"] + (0.005 if row["correlation"] >= 0 else -0.005),
            bar.get_y() + bar.get_height() / 2,
            f"  {row['correlation']:+.3f}{sig}", va="center", fontsize=8,
            ha="left" if row["correlation"] >= 0 else "right")

ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Point-Biserial Correlation with meta_label")
ax.set_xlim(-0.5, 0.55)
legend_patches = [
    mpatches.Patch(color=COLORS["high"], label="VIX / OAS feature (negative = bad for M1)"),
    mpatches.Patch(color=COLORS["low"],  label="VIX / OAS feature (positive = good for M1)"),
    mpatches.Patch(color="#7f8c8d",      label="Primary model feature"),
]
ax.legend(handles=legend_patches, loc="lower right", fontsize=8)
ax.set_title("* p<0.05   ** p<0.01", fontsize=9)

plt.tight_layout()
fig.savefig(OUT_DIR / "4_feature_correlations.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[Chart 4] Top feature correlations with meta_label:")
top = corr_df.sort_values("correlation", key=abs, ascending=False).head(8)
for _, row in top.iterrows():
    sig = "**" if row["pvalue"] < 0.01 else ("*" if row["pvalue"] < 0.05 else "   ")
    print(f"  {sig} {row['feature']:35s}  r={row['correlation']:+.3f}  p={row['pvalue']:.4f}")


# ===========================================================================
# CHART 5: Return Distributions — Wins vs Losses, coloured by regime
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Chart 5 — M1 Return Distribution by Regime", fontsize=14, fontweight="bold")

for ax, (regime_col, low_label, high_label, title) in zip(axes, [
    ("vix_high_regime",  "Low VIX",   "High VIX",  "VIX Regime"),
    ("oas_wide_regime",  "Tight OAS", "Wide OAS",   "OAS Regime"),
]):
    low_returns  = df[df[regime_col] == 0]["meta_target_return"] * 100
    high_returns = df[df[regime_col] == 1]["meta_target_return"] * 100

    bins = np.linspace(df["meta_target_return"].min() * 100 - 0.5,
                       df["meta_target_return"].max() * 100 + 0.5, 25)
    ax.hist(low_returns,  bins=bins, alpha=0.6, color=COLORS["low"],   label=low_label,  edgecolor="white")
    ax.hist(high_returns, bins=bins, alpha=0.6, color=COLORS["high"],  label=high_label, edgecolor="white")
    ax.axvline(low_returns.mean(),  color=COLORS["low"],  linestyle="--", linewidth=2,
               label=f"{low_label} mean: {low_returns.mean():+.2f}%")
    ax.axvline(high_returns.mean(), color=COLORS["high"], linestyle="--", linewidth=2,
               label=f"{high_label} mean: {high_returns.mean():+.2f}%")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("M1 Monthly Net Return (%)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{title}")
    ax.legend(fontsize=8)

    t_stat, p_t = stats.ttest_ind(low_returns.dropna(), high_returns.dropna())
    ax.set_title(f"{title}  (t-test p={p_t:.4f}{'*' if p_t < 0.05 else ''})")

plt.tight_layout()
fig.savefig(OUT_DIR / "5_return_distributions_by_regime.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[Chart 5] Return distributions:")
for regime_col, low_label, high_label in [
    ("vix_high_regime", "Low VIX",   "High VIX"),
    ("oas_wide_regime", "Tight OAS", "Wide OAS"),
]:
    low_r  = df[df[regime_col] == 0]["meta_target_return"] * 100
    high_r = df[df[regime_col] == 1]["meta_target_return"] * 100
    _, p = stats.ttest_ind(low_r.dropna(), high_r.dropna())
    print(f"  {low_label}: mean={low_r.mean():+.2f}%  {high_label}: mean={high_r.mean():+.2f}%  t-test p={p:.4f}")


# ===========================================================================
# CHART 6: Simulated M2 Gate — What happens if you filter by regime?
# ===========================================================================
df_sorted = df.sort_values("date").copy()
df_sorted["month_num"] = range(len(df_sorted))

def cumulative_equity(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()

# Strategy variants
all_returns       = df_sorted["meta_target_return"]
low_vix_only      = df_sorted[df_sorted["vix_high_regime"] == 0]["meta_target_return"]
tight_oas_only    = df_sorted[df_sorted["oas_wide_regime"] == 0]["meta_target_return"]
best_regime       = df_sorted[(df_sorted["vix_high_regime"] == 0) & (df_sorted["oas_wide_regime"] == 0)]["meta_target_return"]
worst_regime      = df_sorted[(df_sorted["vix_high_regime"] == 1) & (df_sorted["oas_wide_regime"] == 1)]["meta_target_return"]

def sharpe(r: pd.Series) -> float:
    if r.std() == 0 or len(r) < 5:
        return float("nan")
    return (r.mean() / r.std()) * np.sqrt(12)

def win_rate(r: pd.Series) -> float:
    return (r > 0).mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Chart 6 — Simulated M2 Regime Gate: If You Only Traded in Favorable Regimes",
             fontsize=13, fontweight="bold")

# Left: performance table
regimes = {
    "All M1 events\n(no filter)":         all_returns,
    "Low VIX only":                        low_vix_only,
    "Tight OAS only":                      tight_oas_only,
    "Low VIX + Tight OAS\n(best regime)":  best_regime,
    "High VIX + Wide OAS\n(worst regime)": worst_regime,
}

table_data = []
for label, r in regimes.items():
    table_data.append({
        "Filter": label,
        "N events": len(r),
        "Win Rate": f"{win_rate(r):.1%}",
        "Avg Return": f"{r.mean()*100:+.2f}%",
        "Sharpe": f"{sharpe(r):.2f}",
        "Max Loss": f"{r.min()*100:.2f}%",
    })
table_df = pd.DataFrame(table_data)

axes[0].axis("off")
tbl = axes[0].table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.1, 2.0)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    elif row == 4:
        cell.set_facecolor("#d5f5e3")
    elif row == 5:
        cell.set_facecolor("#fadbd8")
axes[0].set_title("Performance Under Different Regime Filters", fontsize=10, fontweight="bold")

# Right: win rate bar chart per regime
labels = ["All events", "Low VIX", "Tight OAS", "Low VIX\n+Tight OAS", "High VIX\n+Wide OAS"]
win_rates = [win_rate(r) for r in regimes.values()]
n_events  = [len(r) for r in regimes.values()]
bar_cols  = ["#95a5a6", COLORS["low"], COLORS["tight"], "#1abc9c", COLORS["high"]]
bars = axes[1].bar(labels, win_rates, color=bar_cols, edgecolor="white")
for bar, wr, n in zip(bars, win_rates, n_events):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{wr:.1%}\n(n={n})", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[1].axhline(0.5, color="red", linestyle=":", linewidth=1.2, label="Random (50%)")
axes[1].axhline(win_rate(all_returns), color="black", linestyle="--", linewidth=1.2,
                label=f"M1 baseline ({win_rate(all_returns):.1%})")
axes[1].set_ylim(0, 1.1)
axes[1].set_ylabel("M1 Win Rate")
axes[1].legend(fontsize=8)
axes[1].set_title("Win Rate Under Regime Filters")

plt.tight_layout()
fig.savefig(OUT_DIR / "6_regime_gate_simulation.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\n[Chart 6] Regime gate simulation:")
for label, r in regimes.items():
    clean_label = label.replace("\n", " ")
    print(f"  {clean_label:40s}  n={len(r):3d}  win={win_rate(r):.1%}  avg={r.mean()*100:+.2f}%  Sharpe={sharpe(r):.2f}")


# ===========================================================================
# Final summary
# ===========================================================================
print("\n" + "=" * 60)
print("FILES SAVED:")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f}")
print("=" * 60)
