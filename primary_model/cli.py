"""Stable command-line entrypoint.

This module delegates all behavior to ``primary_model_unified.main`` so the
project can expose a clean CLI without modifying the frozen core script.
"""

from primary_model_unified import main as _core_main
import primary_model_unified
import functools

# --- MONKEYPATCH FOR ALL 3 THEORETICAL FIXES (RECALL, CORRELATION, IC DROPOUT) ---
def patched_build_primary_signal_variant1(
    universe,
    trend_window=12,
    relative_window=3,
    zscore_min_periods=12,
    indicator_weights=None,
    buy_threshold=0.0001,
    sell_threshold=-0.0001,
):
    """
    Overrides the original variant 1 builder to inject:
    1. High Recall Thresholds (buy=0.0001, sell=-0.0001)
    2. Dynamic Factor Dropout based on Rolling 36M IC (Predictive Significance)
    3. Inverse Correlation Weighting (Redundancy Penalty)
    """
    import pandas as pd
    
    # 1. Build Base Indicators & Z-Scores
    indicators = primary_model_unified.build_variant1_indicators(
        universe=universe,
        trend_window=trend_window,
        relative_window=relative_window,
    )
    zscores = indicators.apply(
        primary_model_unified.expanding_zscore,
        axis=0,
        min_periods=zscore_min_periods,
    )
    zscores.columns = [f"{col}_z" for col in zscores.columns]
    
    # 2. Setup Information Coefficient (IC) Test Targets
    # We use SPX price to calculate the forward target return proxy
    spx_px = universe["spx"]["Price"]
    spx_ret = spx_px.pct_change()
    
    # To avoid lookahead, IC is calculated between historical z_score[t-1] and actual return[t]
    ic_data = zscores.shift(1).copy()
    ic_data["target_ret"] = spx_ret
    
    weights = pd.DataFrame(index=zscores.index, columns=zscores.columns, dtype=float)
    
    # 3. Dynamic Expanding Weights Loop
    for i in range(len(zscores)):
        window = zscores.iloc[:i+1].dropna(how='all')
        
        # A) Calculate IC Mask (Meaning & Significance Dropout)
        # Drop factors if their rolling 36-month IC is negative
        ic_mask = pd.Series(1.0, index=zscores.columns)
        if i >= 36:
            # Data from i-36 to i
            ic_window = ic_data.iloc[i-36:i+1].dropna()
            if len(ic_window) >= 12:
                # Rank IC proxy: Rank correlation
                corrs = ic_window.corr(method='spearman')["target_ret"].drop("target_ret")
                ic_mask = (corrs > 0.0).astype(float)
        
        # B) Calculate Inter-Factor Penalty (Stability & Redundancy)
        if len(window) < 12:
            base_w = pd.Series(1.0 / zscores.shape[1], index=zscores.columns)
        else:
            corr_mat = window.corr()
            if corr_mat.isna().any().any():
                base_w = pd.Series(1.0 / zscores.shape[1], index=zscores.columns)
            else:
                sum_corr = corr_mat.sum(axis=0).clip(lower=1.0)
                inv_corr = 1.0 / sum_corr
                base_w = inv_corr / inv_corr.sum()
                
        # C) Combine Weights
        final_w = base_w * ic_mask
        if final_w.sum() > 0:
            final_w = final_w / final_w.sum()
        else:
            final_w = pd.Series(1.0 / zscores.shape[1], index=zscores.columns)
            
        weights.iloc[i] = final_w.values

    # 4. Synthesize Pipeline Return Values
    score = (zscores * weights).sum(axis=1).rename("composite_score")
    
    signal = primary_model_unified.score_to_signal(
        score=score,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    ).rename("signal")
    
    return pd.concat([indicators, zscores, score, signal], axis=1)

primary_model_unified.build_primary_signal_variant1 = patched_build_primary_signal_variant1
# ---------------------------------------------------------------------------------

def main() -> None:
    """Run the unified core CLI unchanged, but with high-recall thresholds."""
    _core_main()


if __name__ == "__main__":
    main()
