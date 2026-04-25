import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from sklearn.ensemble import GradientBoostingRegressor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION
# ==========================================
# UPDATE THESE FILENAMES TO MATCH YOUR LOCAL FILES
FILES = {
    'spx': "SNP500 (1).xlsx - Worksheet.csv",
    'rates': "US 10Y treasury bonds (1).xlsx - Worksheet.csv",
    'commodities': "BCOM INDEX (4).xlsx - Worksheet.csv",
    'credit': "US_IG_corporate_bonds.xlsx - Worksheet.csv"
}

# The Core 5 Features for the ML Model
ML_FEATURES = ['rates_phase', 'credit_phase', 'spx_phase', 'commodities_phase', 'spx_roc_3m']

# Risk Constraints (Normal Regime)
MIN_WEIGHT = 0.00
MAX_WEIGHT = 1.00

# ==========================================
# 2. DATA LOADING & PRE-PROCESSING
# ==========================================
def load_and_clean_data(file_map):
    print("Loading data...")
    prices = pd.DataFrame()
    for name, filepath in file_map.items():
        try:
            # Dynamic header detection to handle metadata rows
            with open(filepath, 'r') as f: lines = f.readlines()
            header_row = -1
            for i, line in enumerate(lines[:20]):
                if 'Date' in line and ('PX_LAST' in line or 'Close' in line):
                    header_row = i
                    break
            
            if header_row == -1:
                print(f"Warning: Could not find header for {name}")
                continue
            
            df = pd.read_csv(filepath, skiprows=header_row)
            df.columns = [c.strip() for c in df.columns]
            
            # Identify Date and Price columns
            date_col = next((c for c in df.columns if 'Date' in c), None)
            close_col = next((c for c in df.columns if 'PX_LAST' in c or 'Close' in c), None)
            
            if date_col and close_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
                # Rename to asset name and force numeric
                series = df[[close_col]].sort_index().rename(columns={close_col: name})
                series[name] = pd.to_numeric(series[name], errors='coerce')
                
                # Merge into master dataframe
                if prices.empty:
                    prices = series
                else:
                    prices = prices.join(series, how='outer')
        except Exception as e:
            print(f"Error reading {name}: {e}")
            
    # Forward fill gaps (standard practice) and drop initial NaNs
    return prices.fillna(method='ffill').dropna()

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
def get_hilbert_phase(series):
    # Detrend first to isolate the cycle
    detrended = series - series.rolling(12).mean()
    # Apply Hilbert Transform to get the phase angle
    return np.angle(hilbert(detrended.fillna(0)))

def generate_signals(prices):
    print("Generating Indicators (Phases, Momentum, Mania Scores)...")
    X = pd.DataFrame(index=prices.index)
    
    # 1. Cycle Phases (The "Weather")
    X['rates_phase'] = get_hilbert_phase(prices['rates'])
    X['credit_phase'] = get_hilbert_phase(prices['credit'])
    X['spx_phase'] = get_hilbert_phase(prices['spx'])
    X['commodities_phase'] = get_hilbert_phase(prices['commodities'])
    
    # 2. Momentum (The "Trend")
    X['spx_roc_3m'] = prices['spx'].pct_change(3)
    
    # 3. Mania Detection (The "Safety Valve")
    # Z-Score of current price vs 2-year moving average
    roll_mean = prices['spx'].rolling(24).mean()
    roll_std = prices['spx'].rolling(24).std()
    X['spx_mania_score'] = (prices['spx'] - roll_mean) / roll_std
    
    return X.dropna()

# ==========================================
# 4. STRATEGY ENGINE (WALK-FORWARD)
# ==========================================
def run_strategy(prices, X):
    # Calculate Returns for Targets
    returns = prices.pct_change()
    
    # Target: Outperform the Equal-Weight Benchmark next month
    benchmark = returns.mean(axis=1)
    targets = pd.DataFrame(index=returns.index)
    for col in prices.columns:
        targets[col] = returns[col].shift(-1) - benchmark.shift(-1)
    targets = targets.dropna()
    
    # Align Data
    idx = X.index.intersection(targets.index)
    X, targets, returns = X.loc[idx], targets.loc[idx], returns.loc[idx]
    
    # Storage for results
    weights = pd.DataFrame(index=idx, columns=prices.columns)
    start_window = 24 # Need 2 years of data before starting
    
    print(f"Running Walk-Forward Backtest on {len(idx)} months...")
    
    for i in range(len(idx)):
        # Current Context
        current_date = idx[i]
        X_curr = X.iloc[i:i+1]
        
        # --- 1. BASE ALLOCATION (Machine Learning) ---
        # Default to Equal Weight if not enough history yet
        allocation = pd.Series(0.25, index=prices.columns)
        
        if i >= start_window:
            # Train on expanding window (Past -> Present)
            X_train = X[ML_FEATURES].iloc[:i]
            y_train = targets.iloc[:i]
            
            preds = {}
            for asset in prices.columns:
                # Lightweight Gradient Boosting
                model = GradientBoostingRegressor(n_estimators=30, max_depth=2, learning_rate=0.1, random_state=42)
                model.fit(X_train, y_train[asset])
                preds[asset] = model.predict(X_curr[ML_FEATURES])[0]
            
            # Convert Predictions to "Tilt"
            score = pd.Series(preds)
            if score.std() > 1e-6:
                z_score = (score - score.mean()) / score.std()
            else:
                z_score = pd.Series(0, index=score.index)
                
            # Base Allocation = 25% + Tilt
            allocation = allocation + (z_score * 0.15)
            
        # --- 2. MANIA OVERRIDE (The "Hunter" Logic) ---
        mania_score = X_curr['spx_mania_score'].values[0]
        roc3 = X_curr['spx_roc_3m'].values[0]
        
        # RULE A: Raging Bull (1998-1999)
        # Price is > 1.5 Std Dev above trend AND Momentum is UP
        if mania_score > 1.5 and roc3 > 0:
            allocation['spx'] = 1.00 
            allocation['rates'] = 0.0
            allocation['commodities'] = 0.0
            allocation['credit'] = 0.0
            
        # RULE B: The Crash (2000, 2008)
        # Price is > 1.5 Std Dev above trend BUT Momentum broke DOWN
        elif mania_score > 1.5 and roc3 < 0:
            allocation['spx'] = 0.00
            allocation['rates'] = 0.50 # Fly to safety
            allocation['commodities'] = 0.0
            allocation['credit'] = 0.50
            
        # --- 3. FINALIZE WEIGHTS ---
        allocation = allocation.clip(MIN_WEIGHT, MAX_WEIGHT)
        allocation = allocation / allocation.sum() # Ensure sum = 100%
        
        weights.iloc[i] = allocation
        
    return weights, returns, benchmark

# ==========================================
# 5. PERFORMANCE REPORTING
# ==========================================
def report_performance(weights, returns, benchmark):
    # Calculate Strategy Returns
    # Shift weights by 1 because decisions made at T apply to returns at T+1
    strat_ret = (weights.shift(1) * returns).sum(axis=1).dropna()
    bench_ret = benchmark.loc[strat_ret.index]
    spx_ret = returns['spx'].loc[strat_ret.index]
    
    # Cumulative Curves
    strat_cum = (1 + strat_ret).cumprod()
    bench_cum = (1 + bench_ret).cumprod()
    spx_cum = (1 + spx_ret).cumprod()
    
    # Metrics
    def get_stats(r):
        ann_ret = r.mean() * 12
        ann_vol = r.std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol
        dd = (1+r).cumprod() / (1+r).cumprod().cummax() - 1
        max_dd = dd.min()
        return ann_ret, ann_vol, sharpe, max_dd

    s_ret, s_vol, s_sharpe, s_dd = get_stats(strat_ret)
    b_ret, b_vol, b_sharpe, b_dd = get_stats(bench_ret)
    spx_ret_ann, spx_vol, spx_sharpe, spx_dd = get_stats(spx_ret)
    
    print("\n" + "="*50)
    print("MANIA HUNTER STRATEGY: FINAL REPORT")
    print("="*50)
    print(f"{'Metric':<20} | {'Strategy':<12} | {'S&P 500':<12} | {'Benchmark':<12}")
    print("-" * 62)
    print(f"{'Total Return':<20} | {(strat_cum.iloc[-1]-1)*100:.0f}%{'':<5} | {(spx_cum.iloc[-1]-1)*100:.0f}%{'':<5} | {(bench_cum.iloc[-1]-1)*100:.0f}%")
    print(f"{'CAGR (Ann. Return)':<20} | {s_ret*100:.2f}%{'':<6} | {spx_ret_ann*100:.2f}%{'':<6} | {b_ret*100:.2f}%")
    print(f"{'Volatility':<20} | {s_vol*100:.2f}%{'':<6} | {spx_vol*100:.2f}%{'':<6} | {b_vol*100:.2f}%")
    print(f"{'Sharpe Ratio':<20} | {s_sharpe:.2f}{'':<8} | {spx_sharpe:.2f}{'':<8} | {b_sharpe:.2f}")
    print(f"{'Max Drawdown':<20} | {s_dd*100:.2f}%{'':<6} | {spx_dd*100:.2f}%{'':<6} | {b_dd*100:.2f}%")
    print("-" * 62)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(strat_cum, label='Mania Hunter Strategy', linewidth=2, color='#1f77b4')
    plt.plot(spx_cum, label='S&P 500 Only', linewidth=1.5, linestyle='--', color='#d62728')
    plt.plot(bench_cum, label='Passive Benchmark', linewidth=1.5, linestyle=':', color='gray')
    
    plt.title('Strategy Performance vs Benchmarks')
    plt.ylabel('Growth of $1 (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('mania_hunter_performance.png')
    print("\n[+] Performance graph saved as 'mania_hunter_performance.png'")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load
    prices = load_and_clean_data(FILES)
    
    # 2. Features
    X = generate_signals(prices)
    
    # 3. Run Strategy
    weights, returns, benchmark = run_strategy(prices, X)
    
    # 4. Report
    report_performance(weights, returns, benchmark)
