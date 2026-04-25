Python 3.14.3 (v3.14.3:323c59a5e34, Feb  3 2026, 11:41:37) [Clang 16.0.0 (clang-1600.0.26.6)] on darwin
Enter "help" below or click "Help" above for more information.
>>> import numpy as np
... import pandas as pd
... import warnings
... import matplotlib.pyplot as plt
... from scipy.signal import hilbert
... from sklearn.ensemble import GradientBoostingRegressor
... 
... # Suppress warnings
... warnings.filterwarnings('ignore')
... 
... # ==========================================
... # 1. CONFIGURATION
... # ==========================================
... # Update these filenames to match your local files
... FILES = {
...     'spx': "SNP500 (2).xlsx - Worksheet.csv",
...     'rates': "US 10Y treasury bonds (2).xlsx - Worksheet.csv",
...     'commodities': "BCOM INDEX (3).xlsx - Worksheet.csv",
...     'credit': "US IG corporate bonds (1).xlsx - Worksheet.csv"
... }
... 
... # The 5 Core Features (No Mania Override)
... ML_FEATURES = ['rates_phase', 'credit_phase', 'spx_phase', 'commodities_phase', 'spx_roc_3m']
... 
... # Risk Constraints
... MIN_WEIGHT = 0.00
... MAX_WEIGHT = 1.00
... 
... # ==========================================
... # 2. DATA LOADING & PROCESSING
... # ==========================================
... def load_data(file_map):
...     print("Loading data...")
...     prices = pd.DataFrame()
...     for name, filepath in file_map.items():
...         try:
            with open(filepath, 'r') as f: lines = f.readlines()
            header_row = -1
            for i, line in enumerate(lines[:20]):
                if 'Date' in line and ('PX_LAST' in line or 'Close' in line):
                    header_row = i
                    break
            
            if header_row == -1: 
                print(f"Skipping {name}: Header not found.")
                continue
            
            df = pd.read_csv(filepath, skiprows=header_row)
            df.columns = [c.strip() for c in df.columns]
            
            date_col = next((c for c in df.columns if 'Date' in c), None)
            close_col = next((c for c in df.columns if 'PX_LAST' in c or 'Close' in c), None)
            
            if date_col and close_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
                series = df[[close_col]].sort_index().rename(columns={close_col: name})
                series[name] = pd.to_numeric(series[name], errors='coerce')
                
                if prices.empty:
                    prices = series
                else:
                    prices = prices.join(series, how='outer')
        except Exception as e:
            print(f"Error reading {name}: {e}")
            
    # CRITICAL: Sort Index to ensure correct time sequence
    prices.sort_index(inplace=True)
    return prices.fillna(method='ffill').dropna()

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
def get_phase(series):
    # Detrend using 12m rolling mean to isolate cycle
    detrended = series - series.rolling(12).mean()
    # Hilbert Transform
    return np.angle(hilbert(detrended.fillna(0)))

def generate_features(prices):
    print("Generating Features...")
    X = pd.DataFrame(index=prices.index)
    
    # 1. Cycle Phases
    X['rates_phase'] = get_phase(prices['rates'])
    X['credit_phase'] = get_phase(prices['credit'])
    X['spx_phase'] = get_phase(prices['spx'])
    X['commodities_phase'] = get_phase(prices['commodities'])
    
    # 2. Momentum
    X['spx_roc_3m'] = prices['spx'].pct_change(3)
    
    return X.dropna()

# ==========================================
# 4. STRATEGY EXECUTION
# ==========================================
def run_strategy():
    # 1. Load
    prices = load_data(FILES)
    if prices.empty:
        print("Error: No data loaded.")
        return
        
    returns = prices.pct_change()
    
    # 2. Features
    X = generate_features(prices)
    
    # 3. Targets (Active Return vs Equal Weight Benchmark)
    benchmark = returns.mean(axis=1)
    targets = pd.DataFrame(index=returns.index)
    for col in prices.columns:
        # We predict Next Month's active return
        targets[col] = returns[col].shift(-1) - benchmark.shift(-1)
    targets = targets.dropna()
    
    # 4. Align Data
    idx = X.index.intersection(targets.index)
    X, targets, returns = X.loc[idx], targets.loc[idx], returns.loc[idx]
    benchmark = benchmark.loc[idx]
    
    # 5. Walk-Forward Backtest
    print(f"Running Backtest on {len(idx)} months...")
    weights = pd.DataFrame(index=idx, columns=prices.columns)
    start_window = 24 # 2 years warm-up
    
    for i in range(len(idx)):
        X_curr = X.iloc[i:i+1]
        
        # Base Allocation (Equal Weight)
        allocation = pd.Series(0.25, index=prices.columns)
        
        # ML Logic (Only if enough history)
        if i >= start_window:
            X_train = X[ML_FEATURES].iloc[:i]
            y_train = targets.iloc[:i]
            
            preds = {}
            for asset in prices.columns:
                # Lightweight Gradient Boosting (Trend Aware)
                model = GradientBoostingRegressor(n_estimators=30, max_depth=2, learning_rate=0.1, random_state=42)
                model.fit(X_train, y_train[asset])
                preds[asset] = model.predict(X_curr[ML_FEATURES])[0]
            
            # Convert Predictions to "Tilt" (Z-Score)
            score = pd.Series(preds)
            if score.std() > 1e-6:
                z_score = (score - score.mean()) / score.std()
                # Apply Tilt (0.25 +/- Tilt)
                allocation = allocation + (z_score * 0.15)
        
        # --- NO MANIA OVERRIDE ---
        # The ML model handles everything dynamically
        
        # Normalize to 100%
        allocation = allocation.clip(MIN_WEIGHT, MAX_WEIGHT)
        allocation = allocation / allocation.sum()
        weights.iloc[i] = allocation

    # 6. Calculate Performance
    # Weights at T are applied to Returns at T+1
    strat_ret = (weights.shift(1) * returns).sum(axis=1).dropna()
    spx_ret = returns['spx'].loc[strat_ret.index]
    bench_ret = benchmark.loc[strat_ret.index]
    
    # Cumulative Curves
    strat_cum = (1 + strat_ret).cumprod()
    spx_cum = (1 + spx_ret).cumprod()
    bench_cum = (1 + bench_ret).cumprod()
    
    # Metrics
    total_ret = strat_cum.iloc[-1] - 1
    cagr = strat_cum.iloc[-1]**(12/len(strat_cum)) - 1
    sharpe = (strat_ret.mean() * 12) / (strat_ret.std() * np.sqrt(12))
    dd = (strat_cum / strat_cum.cummax() - 1).min()
    
    print("\n" + "="*50)
    print("ORIGINAL 5 STRATEGY PERFORMANCE (1997-2025)")
    print("="*50)
    print(f"{'Metric':<20} | {'Strategy':<10} | {'S&P 500':<10}")
    print("-" * 50)
    print(f"{'Total Return':<20} | {total_ret:.0%}      | {spx_cum.iloc[-1]-1:.0%}")
    print(f"{'CAGR':<20} | {cagr:.2%}     | {(spx_cum.iloc[-1])**(12/len(spx_cum))-1:.2%}")
    print(f"{'Sharpe Ratio':<20} | {sharpe:.2f}       | {(spx_ret.mean()*12)/(spx_ret.std()*np.sqrt(12)):.2f}")
    print(f"{'Max Drawdown':<20} | {dd:.2%}    | {(spx_cum/spx_cum.cummax()-1).min():.2%}")
    print("-" * 50)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(strat_cum, label='Original 5 Strategy', linewidth=2, color='#2ca02c')
    plt.plot(spx_cum, label='S&P 500', linewidth=1, linestyle='--', color='#d62728')
    plt.plot(bench_cum, label='Benchmark', linewidth=1, linestyle=':', color='gray')
    plt.title('Original 5 Strategy vs Benchmarks')
    plt.ylabel('Growth of $1 (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('strategy_performance.png')
    print("\n[+] Chart saved as 'strategy_performance.png'")
    
    # Save Weights
    weights.to_csv('monthly_allocations.csv')
    print("[+] Weights saved as 'monthly_allocations.csv'")

if __name__ == "__main__":
