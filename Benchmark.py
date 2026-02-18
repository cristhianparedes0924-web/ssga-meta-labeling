import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PART 1: M1 PRIMARY SIGNAL MODEL
# =============================================================================
# I have used a bunch of functions so that each part can be independently changed later as per guidance.


# 1) Loading data:

def load_data(bcom_path, spx_path, treasury_path, ig_path):
    """Load and merge all 4 data sources"""
    
    def load_single_file(filepath, index_name):
        df = pd.read_excel(filepath)
        data_df = df.iloc[6:].copy()
        data_df.columns = ['Date', 'Price', 'Return']
        data_df['Date'] = pd.to_datetime(data_df['Date'], errors='coerce')
        data_df['Price'] = pd.to_numeric(data_df['Price'], errors='coerce')
        data_df['Return'] = pd.to_numeric(data_df['Return'], errors='coerce')
        data_df = data_df.dropna(subset=['Date'])
        data_df = data_df.sort_values('Date').reset_index(drop=True)
        data_df = data_df.rename(columns={
            'Price': f'{index_name}_Price', 
            'Return': f'{index_name}_Return'
        })
        return data_df
    
    # Load each file
    bcom = load_single_file(bcom_path, 'BCOM')
    spx = load_single_file(spx_path, 'SPX')
    treasury = load_single_file(treasury_path, 'Treasury10Y')
    ig_bonds = load_single_file(ig_path, 'IG_Corp')
    
    # Merge all on Date
    merged = bcom.merge(spx, on='Date', how='outer')
    merged = merged.merge(treasury, on='Date', how='outer')
    merged = merged.merge(ig_bonds, on='Date', how='outer')
    merged = merged.sort_values('Date').reset_index(drop=True)
    
    return merged


# 2) Creating the Indicators (5 used here, we can refine them further later on as per Xuesong's advice)
# There are guides in the quant book that he sent for choosing variables, I did try to follow them.

def create_indicators(df):
    """
    Create 5 indicators from the price/return data:
    
    1. MOMENTUM: 12-month S&P 500 price return
       - Logic: Stocks trending up tend to continue (momentum effect)
       - Bullish when positive, bearish when negative
    
    2. CROSS-ASSET: 6M Commodities return minus 6M Stock return  
       - Logic: Capital Rotating to commodities. Commodities are seen as a better hedge against inflation, lower faith in growth
       - Bullish when negative (stocks winning), bearish when positive
    
    3. YIELD CHANGE: 6-month change in 10Y Treasury yield
       - Logic: Rising yields = tighter financial conditions. Generally indicates Quantitative Tightening.
       - Bullish when falling, bearish when rising
    
    4. CREDIT: 3-month average IG Corporate bond returns
       - Logic: Strong credit market = healthy risk appetite. Could also be like a proxy for confidence in US corporates.
       - Bullish when positive, bearish when negative
    
    5. VOLATILITY: Ratio of 6M volatility to 12M volatility
       - Logic: Higher Volatility means more uncertainty for investors.
       - Bullish when <1 (vol contracting), bearish when >1
    """

    df = df.copy()
    
    # --- INDICATOR 1: PRICE MOMENTUM ---
    df['Ind1_Momentum_12M'] = df['SPX_Price'].pct_change(12) * 100
    
    # --- INDICATOR 2: CROSS-ASSET SIGNAL ---
    df['BCOM_Mom_6M'] = df['BCOM_Price'].pct_change(6) * 100
    df['SPX_Mom_6M'] = df['SPX_Price'].pct_change(6) * 100
    df['Ind2_CrossAsset'] = df['BCOM_Mom_6M'] - df['SPX_Mom_6M']
    
    # --- INDICATOR 3: YIELD CHANGE ---
    df['Ind3_YieldChange_6M'] = df['Treasury10Y_Price'].diff(6)
    
    # --- INDICATOR 4: CREDIT SIGNAL ---
    df['Ind4_Credit_3M'] = df['IG_Corp_Return'].rolling(3).mean()
    
    # --- INDICATOR 5: VOLATILITY REGIME ---
    df['Vol_6M'] = df['SPX_Return'].rolling(6).std()
    df['Vol_12M'] = df['SPX_Return'].rolling(12).std()
    df['Ind5_VolRegime'] = df['Vol_6M'] / df['Vol_12M']
    
    return df


# 3) Standardize using the Normal Z-Distribution
# Also we will flip signs after calculating to maintain consistency that positive z-score means BULLISH.

def zscore_expanding(series, min_periods=12):
    mean = series.expanding(min_periods=min_periods).mean()
    std = series.expanding(min_periods=min_periods).std()
    return (series - mean) / std


def standardize_indicators(df):
    
    df = df.copy()
    
    # Z-score each indicator with appropriate sign:
    # Positive z-score = BULLISH signal
    # Negative z-score = BEARISH signal
    
    # Momentum: positive is bullish (keep sign)
    df['Z1_Momentum'] = zscore_expanding(df['Ind1_Momentum_12M'])
    
    # Cross-Asset: commodities outperforming is BEARISH (flip sign)
    df['Z2_CrossAsset'] = -zscore_expanding(df['Ind2_CrossAsset'])
    
    # Yield Change: rising yields are BEARISH (flip sign)
    df['Z3_Yield'] = -zscore_expanding(df['Ind3_YieldChange_6M'])
    
    # Credit: strong credit is BULLISH (keep sign)
    df['Z4_Credit'] = zscore_expanding(df['Ind4_Credit_3M'])
    
    # Volatility: high vol regime is BEARISH (flip sign)
    df['Z5_Vol'] = -zscore_expanding(df['Ind5_VolRegime'])
    
    # Winsorize at +/- 3 standard deviations. This is also in accordance to the quant book as they also use 3 for clipping. 
    z_cols = ['Z1_Momentum', 'Z2_CrossAsset', 'Z3_Yield', 'Z4_Credit', 'Z5_Vol']
    for col in z_cols:
        df[col] = df[col].clip(-3, 3)
    
    return df


# 4) Create a composite score using the weighted average. The weights are just equally weighted for now.

def calculate_composite_score(df, weights=None):
    """
    Combine z-scores into a single composite score.
    
    Default weights:
    - Momentum: 20%
    - Cross-Asset: 20% 
    - Yield: 20%
    - Credit: 20%
    - Volatility: 20% 
    """
    
    if weights is None:
        weights = {
            'Z1_Momentum': 0.20,
            'Z2_CrossAsset': 0.20,
            'Z3_Yield': 0.20,
            'Z4_Credit': 0.20,
            'Z5_Vol': 0.20
        }
    
    df = df.copy()
    
    df['Composite_Score'] = (
        df['Z1_Momentum'] * weights['Z1_Momentum'] +
        df['Z2_CrossAsset'] * weights['Z2_CrossAsset'] +
        df['Z3_Yield'] * weights['Z3_Yield'] +
        df['Z4_Credit'] * weights['Z4_Credit'] +
        df['Z5_Vol'] * weights['Z5_Vol']
    )
    
    return df


def generate_signals(df, buy_threshold=0.5, sell_threshold=-0.5):
    """
    Generate BUY / SELL / HOLD signals based on composite score.
    
    - BUY:  Composite Score > 0.5
    - SELL: Composite Score < -0.5
    - HOLD: Between -0.5 and 0.5
    """
    
    df = df.copy()
    
    def get_signal(score):
        if pd.isna(score):
            return None
        elif score > buy_threshold:
            return 'BUY'
        elif score < sell_threshold:
            return 'SELL'
        else:
            return 'HOLD'
    
    df['Signal'] = df['Composite_Score'].apply(get_signal)
    
    return df


# 5) Creating a function to run the main model

def run_signal_model(bcom_path, spx_path, treasury_path, ig_path,
                     weights=None, buy_threshold=0.5, sell_threshold=-0.5):
    
    # Load data
    print("Loading data...")
    df = load_data(bcom_path, spx_path, treasury_path, ig_path)
    
    # Create indicators
    print("Creating indicators...")
    df = create_indicators(df)
    
    # Standardize
    print("Standardizing indicators...")
    df = standardize_indicators(df)
    
    # Calculate composite score
    print("Calculating composite score...")
    df = calculate_composite_score(df, weights)
    
    # Generate signals
    print("Generating signals...")
    df = generate_signals(df, buy_threshold, sell_threshold)
    
    print("Done!")
    
    return df


def get_latest_signal(df):
    """Get the most recent signal and its components"""
    
    latest = df.dropna(subset=['Signal']).iloc[-1]
    
    print("\n" + "="*60)
    print(f"LATEST SIGNAL: {latest['Signal']}")
    print("="*60)
    print(f"Date: {latest['Date'].strftime('%Y-%m-%d')}")
    print(f"Composite Score: {latest['Composite_Score']:.3f}")
    print("\nIndicator Breakdown:")
    print(f"  1. Momentum (12M):     {latest['Z1_Momentum']:+.2f}")
    print(f"  2. Cross-Asset:        {latest['Z2_CrossAsset']:+.2f}")
    print(f"  3. Yield Change:       {latest['Z3_Yield']:+.2f}")
    print(f"  4. Credit:             {latest['Z4_Credit']:+.2f}")
    print(f"  5. Volatility Regime:  {latest['Z5_Vol']:+.2f}")
    print("="*60)
    
    return latest


# =============================================================================
# PART 2: BACKTEST MODULE
# =============================================================================
# This module takes the output of run_signal_model() and:
#   1. Simulates a trading strategy based on BUY/SELL/HOLD signals
#   2. Calculates strategy performance metrics (Sharpe, drawdown, etc.)
#   3. Calculates classification metrics (precision, recall, F1)
#   4. Generates meta-labels for future M2 development
#   5. Compares against MULTIPLE benchmarks (Buy-and-Hold, 60/40, SMA, Random)
#   6. Splits data into 60/40 train/test (per Joubert 2022 paper)
# =============================================================================


# --------------------------------------------------------------------------
# 1) POSITION MAPPING
# --------------------------------------------------------------------------

def map_signals_to_positions(df):
    """
    Convert BUY / SELL / HOLD signals into numerical positions.
    
    Position logic:
      - BUY  →  +1  (long)
      - SELL →  -1  (short) or 0 (flat) depending on strategy constraint
      - HOLD →  carry forward the previous position
    
    For a LONG-ONLY version, set SELL → 0 instead of -1.
    Change `long_only` below to toggle this.
    """
    
    long_only = True  # Set to False if you want to allow shorting
    
    df = df.copy()
    
    # Map signals to raw position
    signal_map = {
        'BUY': 1,
        'SELL': 0 if long_only else -1,
        'HOLD': np.nan  # will forward-fill
    }
    
    df['Position'] = df['Signal'].map(signal_map)
    
    # Forward-fill HOLD (carry previous position)
    df['Position'] = df['Position'].ffill()
    
    # Fill any remaining NaN at the start with 0 (no position until first signal)
    df['Position'] = df['Position'].fillna(0)
    
    return df


# --------------------------------------------------------------------------
# 2) RETURN CALCULATION
# --------------------------------------------------------------------------

def calculate_strategy_returns(df):
    """
    Calculate strategy returns based on positions.
    
    The position at time t is determined by the signal at time t,
    and the return earned is at time t+1 (you act on signal, earn next period return).
    
    This avoids look-ahead bias: signal is generated at end of month t,
    position is taken, return is earned in month t+1.
    """
    
    df = df.copy()
    
    # Forward return: the return you earn NEXT period after taking the position
    df['Forward_Return'] = df['SPX_Return'].shift(-1)
    
    # Strategy return = position * forward return
    df['Strategy_Return'] = df['Position'] * df['Forward_Return']
    
    # Buy-and-hold benchmark return (always long)
    df['BH_Return'] = df['Forward_Return']
    
    # Cumulative returns (growth of $1)
    df['Strategy_Cumulative'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()
    df['BH_Cumulative'] = (1 + df['BH_Return'].fillna(0)).cumprod()
    
    return df


# --------------------------------------------------------------------------
# 3) BENCHMARK SUITE  [NEW — 3 ADDITIONAL BENCHMARKS]
# --------------------------------------------------------------------------

def calculate_benchmark_6040(df):
    """
    BENCHMARK 1: 60/40 Portfolio (60% S&P 500, 40% 10Y Treasury Bonds)
    
    WHY: The classic institutional balanced benchmark. Our M1 model is 
    a timing model that shifts between 100% equity and 0% equity. If 
    M1 can't beat a passive 60/40, the timing is not adding value vs 
    the simplest diversification approach.
    
    HOW: Each month, the portfolio return is:
        r_6040 = 0.60 * r_SPX + 0.40 * r_Treasury
    Uses the same forward-return convention (shift -1) as the strategy
    to maintain consistent timing and avoid look-ahead bias.
    
    NOTE: This is a static (non-rebalanced) 60/40 for simplicity. 
    Monthly rebalancing effects are negligible at this frequency.
    """
    
    df = df.copy()
    
    # Forward returns for both assets (same timing convention)
    fwd_spx = df['SPX_Return'].shift(-1)
    fwd_treasury = df['Treasury10Y_Return'].shift(-1)
    
    # 60/40 blended return
    df['Bench_6040_Return'] = 0.60 * fwd_spx + 0.40 * fwd_treasury
    
    # Cumulative growth of $1
    df['Bench_6040_Cumulative'] = (1 + df['Bench_6040_Return'].fillna(0)).cumprod()
    
    return df


def calculate_benchmark_sma(df, short_window=10, long_window=12):
    """
    BENCHMARK 2: Simple Moving Average (SMA) Crossover on S&P 500
    
    WHY: Our M1 model's strongest signal is Indicator 1 (12M Momentum). 
    This benchmark tests whether the additional complexity of the other 
    4 indicators (cross-asset, yield, credit, vol regime) actually adds 
    value beyond a simple trend-following rule using price alone.
    
    If M1 can't beat SMA, the extra indicators are adding noise, not signal.
    
    HOW: 
      - Compute 10-month and 12-month SMA of SPX price
      - If SMA_10 > SMA_12 → long (momentum is positive, trend is up)
      - If SMA_10 ≤ SMA_12 → flat (trend broken, step aside)
      - Position applied to next month's return (same shift convention)
    
    WINDOW CHOICE (10M vs 12M):
      - 10/12 is a standard medium-term trend pair (Faber 2007)
      - Matches our indicator lookback range (6M-12M)
      - Short enough to catch regime shifts, long enough to avoid whipsaws
    """
    
    df = df.copy()
    
    # Compute SMAs on SPX price
    df['SMA_Short'] = df['SPX_Price'].rolling(short_window).mean()
    df['SMA_Long'] = df['SPX_Price'].rolling(long_window).mean()
    
    # Signal: long when short SMA > long SMA (uptrend), flat otherwise
    df['SMA_Position'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1, 0)
    
    # Apply to forward returns (same timing as strategy)
    fwd_spx = df['SPX_Return'].shift(-1)
    df['Bench_SMA_Return'] = df['SMA_Position'] * fwd_spx
    
    # Cumulative growth
    df['Bench_SMA_Cumulative'] = (1 + df['Bench_SMA_Return'].fillna(0)).cumprod()
    
    return df


def calculate_benchmark_random(df, n_simulations=1000, seed=42):
    """
    BENCHMARK 3: Random Signal Baseline (Monte Carlo)
    
    WHY: Statistical validity. If M1's performance can't be distinguished 
    from random signal generation with the same distribution of BUY/SELL/HOLD, 
    then M1 has no real predictive edge — any apparent performance is luck.
    
    This is particularly important for the meta-labeling framework because 
    M2's job is to filter M1's signals. If M1 isn't better than random to 
    begin with, M2 has nothing useful to learn from.
    
    HOW:
      1. Count the empirical distribution of M1's signals:
         e.g., 45% BUY, 20% SELL, 35% HOLD
      2. Generate 1,000 random signal sequences with the same distribution
      3. For each simulation:
         - Map random signals → positions (same logic as M1)
         - Calculate returns using same forward-return convention
         - Compute Sharpe ratio
      4. Store the MEDIAN simulation's returns as the benchmark
      5. Store the full Sharpe distribution for p-value calculation
    
    INTERPRETATION:
      - If M1 Sharpe > 95th percentile of random Sharpes → statistically significant (p < 0.05)
      - If M1 Sharpe is near the median → M1 has no edge
      - The median random series serves as a "what luck looks like" benchmark line
    
    n_simulations=1000 gives stable percentile estimates.
    seed=42 for reproducibility.
    """
    
    df = df.copy()
    rng = np.random.RandomState(seed)
    
    # Get the empirical signal distribution from M1
    valid_signals = df.dropna(subset=['Signal'])
    signal_counts = valid_signals['Signal'].value_counts(normalize=True)
    signal_labels = signal_counts.index.tolist()
    signal_probs = signal_counts.values.tolist()
    
    n_periods = len(valid_signals)
    fwd_returns = df['SPX_Return'].shift(-1).values  # full series
    
    # Storage for simulation results
    sim_sharpes = []
    sim_cumulative_returns = []  # store each sim's cumulative for percentile bands
    
    for i in range(n_simulations):
        # Generate random signals with same distribution
        random_signals = rng.choice(signal_labels, size=n_periods, p=signal_probs)
        
        # Map to positions (same logic as map_signals_to_positions)
        positions = np.zeros(n_periods)
        for t in range(n_periods):
            if random_signals[t] == 'BUY':
                positions[t] = 1
            elif random_signals[t] == 'SELL':
                positions[t] = 0  # long-only
            else:  # HOLD
                positions[t] = positions[t-1] if t > 0 else 0
        
        # Get forward returns aligned to valid signal dates
        valid_indices = valid_signals.index.tolist()
        sim_returns = np.array([
            positions[j] * fwd_returns[idx] if not np.isnan(fwd_returns[idx]) else 0
            for j, idx in enumerate(valid_indices)
        ])
        
        # Sharpe ratio (annualized, monthly data)
        if sim_returns.std() > 0:
            sharpe = (sim_returns.mean() * 12) / (sim_returns.std() * np.sqrt(12))
        else:
            sharpe = 0
        sim_sharpes.append(sharpe)
        
        # Cumulative returns for this simulation
        sim_cum = np.cumprod(1 + sim_returns)
        sim_cumulative_returns.append(sim_cum)
    
    sim_sharpes = np.array(sim_sharpes)
    
    # Find the median simulation (by Sharpe) to use as the benchmark line
    median_idx = np.argsort(sim_sharpes)[n_simulations // 2]
    median_cum = sim_cumulative_returns[median_idx]
    
    # Map median cumulative back to the DataFrame
    valid_indices = valid_signals.index.tolist()
    df['Bench_Random_Return'] = np.nan
    df['Bench_Random_Cumulative'] = np.nan
    
    # Reconstruct median simulation's period returns from cumulative
    median_returns = np.diff(median_cum, prepend=1.0) / np.concatenate([[1.0], median_cum[:-1]])
    
    for j, idx in enumerate(valid_indices):
        if j < len(median_returns):
            df.loc[idx, 'Bench_Random_Return'] = median_returns[j]
            df.loc[idx, 'Bench_Random_Cumulative'] = median_cum[j]
    
    # Forward fill cumulative for pre-signal periods
    df['Bench_Random_Cumulative'] = df['Bench_Random_Cumulative'].ffill().fillna(1.0)
    
    # Store simulation statistics for later analysis
    df.attrs['random_sim_sharpes'] = sim_sharpes
    df.attrs['random_sim_median_sharpe'] = np.median(sim_sharpes)
    df.attrs['random_sim_95th_sharpe'] = np.percentile(sim_sharpes, 95)
    df.attrs['random_sim_5th_sharpe'] = np.percentile(sim_sharpes, 5)
    
    return df


def calculate_all_benchmarks(df):
    """
    Master function: compute all 3 additional benchmarks.
    Buy-and-Hold is already computed in calculate_strategy_returns().
    
    Benchmarks added:
      1. 60/40 Portfolio — passive diversification baseline
      2. SMA Crossover — simple trend-following baseline
      3. Random Signals — statistical significance baseline
    """
    
    print("  Computing benchmarks...")
    
    # Benchmark 1: 60/40 Portfolio
    print("    → 60/40 Portfolio (60% SPX / 40% Treasury)")
    df = calculate_benchmark_6040(df)
    
    # Benchmark 2: SMA Crossover
    print("    → SMA Crossover (10M / 12M)")
    df = calculate_benchmark_sma(df, short_window=10, long_window=12)
    
    # Benchmark 3: Random Signal Baseline
    print("    → Random Signal Baseline (1,000 simulations)")
    df = calculate_benchmark_random(df, n_simulations=1000, seed=42)
    
    print("  Benchmarks complete.")
    
    return df


# --------------------------------------------------------------------------
# 4) PERFORMANCE METRICS
# --------------------------------------------------------------------------

def calculate_max_drawdown(cumulative_returns):
    """Calculate maximum drawdown from a cumulative return series."""
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    return drawdown.min()


def calculate_performance_metrics(df, periods_per_year=12):
    """
    Calculate comprehensive strategy performance metrics.
    
    Assumes monthly data (periods_per_year=12).
    
    Metrics calculated:
      - Annualized Return
      - Annualized Volatility
      - Sharpe Ratio (assuming risk-free rate = 0 for simplicity)
      - Maximum Drawdown
      - Calmar Ratio (return / |max drawdown|)
      - Win Rate (% of periods with positive return)
      - Number of Trades (signal changes)
      - Best / Worst Period
    """
    
    # Filter to periods where we have valid returns
    valid = df.dropna(subset=['Strategy_Return', 'Forward_Return']).copy()
    strat_rets = valid['Strategy_Return']
    bh_rets = valid['BH_Return']
    
    def compute_metrics(returns, label):
        n_periods = len(returns)
        
        # Annualized return
        cumulative = (1 + returns).prod()
        n_years = n_periods / periods_per_year
        ann_return = cumulative ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Annualized volatility
        ann_vol = returns.std() * np.sqrt(periods_per_year)
        
        # Sharpe ratio (excess return over risk-free; using rf=0)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Max drawdown
        cum_series = (1 + returns).cumprod()
        max_dd = calculate_max_drawdown(cum_series)
        
        # Calmar ratio
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        return {
            'Label': label,
            'Total Periods': n_periods,
            'Annualized Return': f"{ann_return:.4f}  ({ann_return*100:.2f}%)",
            'Annualized Volatility': f"{ann_vol:.4f}  ({ann_vol*100:.2f}%)",
            'Sharpe Ratio': f"{sharpe:.3f}",
            'Maximum Drawdown': f"{max_dd:.4f}  ({max_dd*100:.2f}%)",
            'Calmar Ratio': f"{calmar:.3f}",
            'Win Rate': f"{win_rate:.3f}  ({win_rate*100:.1f}%)",
            'Best Period': f"{returns.max():.4f}  ({returns.max()*100:.2f}%)",
            'Worst Period': f"{returns.min():.4f}  ({returns.min()*100:.2f}%)",
            'Cumulative Return': f"{(cumulative - 1):.4f}  ({(cumulative-1)*100:.2f}%)",
        }
    
    strat_metrics = compute_metrics(strat_rets, 'Strategy (M1)')
    bh_metrics = compute_metrics(bh_rets, 'Buy & Hold')
    
    # ── NEW: Compute metrics for additional benchmarks ──
    bench_metrics = {}
    
    # 60/40 benchmark
    bench_6040_rets = valid['Bench_6040_Return'].dropna()
    if len(bench_6040_rets) > 0:
        bench_metrics['60/40 Portfolio'] = compute_metrics(bench_6040_rets, '60/40 Portfolio')
    
    # SMA benchmark
    bench_sma_rets = valid['Bench_SMA_Return'].dropna()
    if len(bench_sma_rets) > 0:
        bench_metrics['SMA Crossover'] = compute_metrics(bench_sma_rets, 'SMA Crossover')
    
    # Random benchmark (median simulation)
    bench_rand_rets = valid['Bench_Random_Return'].dropna()
    if len(bench_rand_rets) > 0:
        bench_metrics['Random Baseline'] = compute_metrics(bench_rand_rets, 'Random Baseline (Median)')
    
    # Additional strategy-specific metrics
    position_changes = (valid['Position'].diff() != 0).sum()
    strat_metrics['Number of Trades (Position Changes)'] = int(position_changes)
    
    time_in_market = (valid['Position'] != 0).sum() / len(valid)
    strat_metrics['Time in Market'] = f"{time_in_market:.3f}  ({time_in_market*100:.1f}%)"
    
    n_long = (valid['Position'] == 1).sum()
    n_flat = (valid['Position'] == 0).sum()
    n_short = (valid['Position'] == -1).sum()
    strat_metrics['Periods Long / Flat / Short'] = f"{n_long} / {n_flat} / {n_short}"
    
    return strat_metrics, bh_metrics, bench_metrics


def print_performance(strat_metrics, bh_metrics, bench_metrics=None):
    """Pretty-print performance comparison across all benchmarks."""
    
    # ── UPDATED: Now prints all benchmarks in a comparison table ──
    
    all_benchmarks = {'Strategy (M1)': strat_metrics, 'Buy & Hold': bh_metrics}
    if bench_metrics:
        all_benchmarks.update(bench_metrics)
    
    # Header
    labels = list(all_benchmarks.keys())
    col_width = 22
    header = f"\n{'Metric':<35}"
    for label in labels:
        # Truncate long labels for formatting
        short_label = label[:col_width-2]
        header += f" {short_label:<{col_width}}"
    print(header)
    print("-" * (35 + col_width * len(labels)))
    
    common_keys = [
        'Total Periods', 'Annualized Return', 'Annualized Volatility',
        'Sharpe Ratio', 'Maximum Drawdown', 'Calmar Ratio',
        'Win Rate', 'Best Period', 'Worst Period', 'Cumulative Return'
    ]
    
    for key in common_keys:
        row = f"  {key:<33}"
        for label in labels:
            val = all_benchmarks[label].get(key, 'N/A')
            val_str = str(val)[:col_width-2]
            row += f" {val_str:<{col_width}}"
        print(row)
    
    # Strategy-specific metrics
    print()
    print("Strategy-Specific Metrics:")
    print("-" * 50)
    strat_only_keys = [
        'Number of Trades (Position Changes)',
        'Time in Market',
        'Periods Long / Flat / Short'
    ]
    for key in strat_only_keys:
        print(f"  {key:<38} {strat_metrics.get(key, 'N/A')}")


# --------------------------------------------------------------------------
# 5) RANDOM BASELINE SIGNIFICANCE TEST  [NEW]
# --------------------------------------------------------------------------

def print_random_significance(df, strategy_sharpe_str):
    """
    Print statistical significance of M1 vs random signals.
    
    Compares M1's Sharpe against the distribution of 1,000 random Sharpes.
    Reports percentile rank and whether p < 0.05.
    """
    
    if 'random_sim_sharpes' not in df.attrs:
        print("  (Random simulation data not available)")
        return
    
    sim_sharpes = df.attrs['random_sim_sharpes']
    
    # Parse strategy Sharpe from the formatted string
    strategy_sharpe = float(strategy_sharpe_str.split()[0])
    
    # Calculate percentile rank of M1's Sharpe
    percentile = (sim_sharpes < strategy_sharpe).sum() / len(sim_sharpes) * 100
    p_value = 1 - percentile / 100
    
    print("\n" + "=" * 70)
    print("  STATISTICAL SIGNIFICANCE: M1 vs RANDOM SIGNALS")
    print("=" * 70)
    print(f"  M1 Strategy Sharpe:        {strategy_sharpe:.3f}")
    print(f"  Random Median Sharpe:      {np.median(sim_sharpes):.3f}")
    print(f"  Random 5th Percentile:     {np.percentile(sim_sharpes, 5):.3f}")
    print(f"  Random 95th Percentile:    {np.percentile(sim_sharpes, 95):.3f}")
    print(f"  M1 Percentile Rank:        {percentile:.1f}th percentile")
    print(f"  p-value (one-tailed):      {p_value:.3f}")
    
    if p_value < 0.05:
        print(f"  Result:                    ✅ SIGNIFICANT (p < 0.05)")
        print(f"                             M1 outperforms {percentile:.0f}% of random strategies")
    elif p_value < 0.10:
        print(f"  Result:                    ⚠️  MARGINALLY SIGNIFICANT (p < 0.10)")
    else:
        print(f"  Result:                    ❌ NOT SIGNIFICANT (p = {p_value:.3f})")
        print(f"                             M1 performance could be due to chance")
    print("=" * 70)


# --------------------------------------------------------------------------
# 6) CLASSIFICATION METRICS (Precision, Recall, F1)
# --------------------------------------------------------------------------

def calculate_signal_accuracy(df):
    """
    Evaluate signal quality using classification-style metrics.
    
    For each BUY/SELL signal, check if the subsequent return was 
    in the predicted direction:
      - BUY  correct if Forward_Return > 0
      - SELL correct if Forward_Return < 0
    
    Metrics (from the Joubert paper's framework):
      - Precision: of all BUY signals, how many led to positive returns?
      - Recall: of all positive return periods, how many did we BUY?
      - F1 Score: harmonic mean of precision and recall
    """
    
    valid = df.dropna(subset=['Signal', 'Forward_Return']).copy()
    
    # --- BUY signal analysis ---
    buy_signals = valid[valid['Signal'] == 'BUY']
    n_buy = len(buy_signals)
    n_buy_correct = (buy_signals['Forward_Return'] > 0).sum() if n_buy > 0 else 0
    buy_precision = n_buy_correct / n_buy if n_buy > 0 else 0
    
    # --- SELL signal analysis ---
    sell_signals = valid[valid['Signal'] == 'SELL']
    n_sell = len(sell_signals)
    n_sell_correct = (sell_signals['Forward_Return'] < 0).sum() if n_sell > 0 else 0
    sell_precision = n_sell_correct / n_sell if n_sell > 0 else 0
    
    # --- Overall accuracy ---
    valid['Signal_Correct'] = False
    valid.loc[(valid['Signal'] == 'BUY') & (valid['Forward_Return'] > 0), 'Signal_Correct'] = True
    valid.loc[(valid['Signal'] == 'SELL') & (valid['Forward_Return'] < 0), 'Signal_Correct'] = True
    
    active_signals = valid[valid['Signal'].isin(['BUY', 'SELL'])]
    overall_accuracy = active_signals['Signal_Correct'].mean() if len(active_signals) > 0 else 0
    
    # --- Recall (for BUY signals) ---
    positive_periods = valid[valid['Forward_Return'] > 0]
    n_positive = len(positive_periods)
    n_caught = len(positive_periods[positive_periods['Signal'] == 'BUY'])
    recall = n_caught / n_positive if n_positive > 0 else 0
    
    # --- F1 Score ---
    f1 = 2 * (buy_precision * recall) / (buy_precision + recall) if (buy_precision + recall) > 0 else 0
    
    results = {
        'BUY Signals': n_buy,
        'BUY Correct (precision)': f"{buy_precision:.3f}  ({n_buy_correct}/{n_buy})",
        'SELL Signals': n_sell,
        'SELL Correct (precision)': f"{sell_precision:.3f}  ({n_sell_correct}/{n_sell})",
        'HOLD Signals': (valid['Signal'] == 'HOLD').sum(),
        'Overall Directional Accuracy': f"{overall_accuracy:.3f}  ({overall_accuracy*100:.1f}%)",
        'Recall (BUY)': f"{recall:.3f}  ({n_caught}/{n_positive})",
        'F1 Score (BUY)': f"{f1:.3f}",
    }
    
    print("\n" + "=" * 70)
    print("SIGNAL CLASSIFICATION METRICS")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k:<35} {v}")
    print("=" * 70)
    
    return results


# --------------------------------------------------------------------------
# 7) META-LABEL GENERATION (for future M2)
# --------------------------------------------------------------------------

def generate_meta_labels(df):
    """
    Generate meta-labels: binary {0, 1} indicating whether the primary 
    model's signal was correct.
    
    From the Joubert paper (Equation 8):
      meta_label = 1 if the primary signal matched the direction of the 
                       subsequent return (i.e., the trade was profitable)
      meta_label = 0 otherwise
    
    Only BUY and SELL signals get meta-labels.
    HOLD signals are excluded (no position taken = no prediction to evaluate).
    
    These meta-labels become the target variable for M2.
    """
    
    df = df.copy()
    df['Meta_Label'] = np.nan
    
    # BUY signal: correct if forward return > 0
    df.loc[(df['Signal'] == 'BUY') & (df['Forward_Return'] > 0), 'Meta_Label'] = 1
    df.loc[(df['Signal'] == 'BUY') & (df['Forward_Return'] <= 0), 'Meta_Label'] = 0
    
    # SELL signal: correct if forward return < 0
    df.loc[(df['Signal'] == 'SELL') & (df['Forward_Return'] < 0), 'Meta_Label'] = 1
    df.loc[(df['Signal'] == 'SELL') & (df['Forward_Return'] >= 0), 'Meta_Label'] = 0
    
    # Summary
    meta_valid = df.dropna(subset=['Meta_Label'])
    n_total = len(meta_valid)
    n_correct = (meta_valid['Meta_Label'] == 1).sum()
    n_incorrect = (meta_valid['Meta_Label'] == 0).sum()
    
    print(f"\nMeta-Labels Generated:")
    print(f"  Total active signals:  {n_total}")
    print(f"  Correct (label=1):     {n_correct}  ({n_correct/n_total*100:.1f}%)")
    print(f"  Incorrect (label=0):   {n_incorrect}  ({n_incorrect/n_total*100:.1f}%)")
    
    return df


# --------------------------------------------------------------------------
# 8) ROLLING PERFORMANCE (for evaluation statistics / M2 features later)
# --------------------------------------------------------------------------

def calculate_rolling_metrics(df, window=12):
    """
    Calculate rolling performance metrics that can serve as 
    evaluation statistics (features for M2 later).
    
    From the paper: "evaluation statistics based on the primary model 
    should be included to help indicate when recent performance is poor."
    
    These include:
      - Rolling Sharpe ratio
      - Rolling hit rate (% correct signals)
    """
    
    df = df.copy()
    
    # Rolling Sharpe (annualized, assuming monthly)
    rolling_mean = df['Strategy_Return'].rolling(window).mean() * 12
    rolling_std = df['Strategy_Return'].rolling(window).std() * np.sqrt(12)
    df['Rolling_Sharpe'] = rolling_mean / rolling_std
    
    # Rolling hit rate (of active signals)
    df['Signal_Hit'] = np.nan
    df.loc[(df['Signal'] == 'BUY') & (df['Forward_Return'] > 0), 'Signal_Hit'] = 1
    df.loc[(df['Signal'] == 'BUY') & (df['Forward_Return'] <= 0), 'Signal_Hit'] = 0
    df.loc[(df['Signal'] == 'SELL') & (df['Forward_Return'] < 0), 'Signal_Hit'] = 1
    df.loc[(df['Signal'] == 'SELL') & (df['Forward_Return'] >= 0), 'Signal_Hit'] = 0
    df['Rolling_HitRate'] = df['Signal_Hit'].rolling(window, min_periods=1).mean()
    
    return df


# --------------------------------------------------------------------------
# 9) TRAIN / TEST SPLIT
# --------------------------------------------------------------------------

def train_test_split(df, train_ratio=0.60):
    """
    Sequential (time-based) train/test split.
    
    Following the Joubert paper: 60% train / 40% test, NO shuffling.
    The most recent 40% is the test set.
    
    This is critical for time series:
      - No shuffling (preserves temporal order)
      - No future data leaks into the training period
      - Test set represents true out-of-sample performance
    
    Note: For M1 (rule-based), the "train" set is the period where you  
    could observe and refine the model. The "test" set is where you 
    evaluate whether the rules generalize. When M2 is added later,
    M2 will literally be trained on the train set and scored on test.
    """
    
    # Only split on rows with valid signals
    valid_df = df.dropna(subset=['Signal']).copy()
    n = len(valid_df)
    split_idx = int(n * train_ratio)
    
    split_date = valid_df.iloc[split_idx]['Date']
    
    # Tag every row in the original df
    df = df.copy()
    df['Set'] = 'pre-signal'  # periods before first signal (warmup for indicators)
    df.loc[df['Date'].isin(valid_df.iloc[:split_idx]['Date']), 'Set'] = 'train'
    df.loc[df['Date'].isin(valid_df.iloc[split_idx:]['Date']), 'Set'] = 'test'
    
    train_df = df[df['Set'] == 'train'].copy()
    test_df = df[df['Set'] == 'test'].copy()
    
    print(f"\n  Train/Test Split (sequential, no shuffling):")
    print(f"    Train:  {len(train_df)} periods  |  {train_df['Date'].min().strftime('%Y-%m')} to {train_df['Date'].max().strftime('%Y-%m')}")
    print(f"    Test:   {len(test_df)} periods  |  {test_df['Date'].min().strftime('%Y-%m')} to {test_df['Date'].max().strftime('%Y-%m')}")
    print(f"    Split at: {split_date.strftime('%Y-%m-%d')}")
    
    return df, train_df, test_df


# --------------------------------------------------------------------------
# 10) MAIN BACKTEST RUNNER
# --------------------------------------------------------------------------

def run_backtest(df, periods_per_year=12, train_ratio=0.60):
    """
    Full backtest pipeline:
      1. Map signals to positions
      2. Calculate strategy returns
      3. Calculate all benchmarks (60/40, SMA, Random)     [NEW]
      4. Generate meta-labels
      5. Calculate rolling metrics
      6. Split into train/test (60/40 sequential)
      7. Print performance metrics for BOTH sets (all benchmarks)
      8. Print classification metrics for BOTH sets
      9. Print random signal significance test               [NEW]
      10. Return enriched full DataFrame + train/test DataFrames
    """
    
    print("\n" + "#" * 70)
    print("#  RUNNING BACKTEST ON M1 PRIMARY MODEL")
    print("#" * 70)
    
    # Step 1: Positions
    df = map_signals_to_positions(df)
    
    # Step 2: Returns
    df = calculate_strategy_returns(df)
    
    # Step 3: Benchmarks  [NEW]
    df = calculate_all_benchmarks(df)
    
    # Step 4: Meta-labels (on full dataset before splitting)
    df = generate_meta_labels(df)
    
    # Step 5: Rolling metrics
    df = calculate_rolling_metrics(df)
    
    # Step 6: Train/Test split
    df, train_df, test_df = train_test_split(df, train_ratio)
    
    # Step 7: Performance metrics — BOTH sets (now with all benchmarks)
    print("\n" + "=" * 70)
    print("  IN-SAMPLE (TRAIN SET) PERFORMANCE")
    print("=" * 70)
    train_strat, train_bh, train_bench = calculate_performance_metrics(train_df, periods_per_year)
    print_performance(train_strat, train_bh, train_bench)
    
    print("\n" + "=" * 70)
    print("  OUT-OF-SAMPLE (TEST SET) PERFORMANCE  ← Primary evaluation")
    print("=" * 70)
    test_strat, test_bh, test_bench = calculate_performance_metrics(test_df, periods_per_year)
    print_performance(test_strat, test_bh, test_bench)
    
    # Step 8: Classification metrics — BOTH sets
    print("\n--- In-Sample Signal Quality ---")
    calculate_signal_accuracy(train_df)
    
    print("\n--- Out-of-Sample Signal Quality ---  ← Primary evaluation")
    calculate_signal_accuracy(test_df)
    
    # Step 9: Random signal significance test  [NEW]
    print_random_significance(df, test_strat['Sharpe Ratio'])
    
    # Step 10: Meta-label summary by set
    for label, subset in [('Train', train_df), ('Test', test_df)]:
        meta_valid = subset.dropna(subset=['Meta_Label'])
        n = len(meta_valid)
        if n > 0:
            n1 = (meta_valid['Meta_Label'] == 1).sum()
            print(f"\n  Meta-Label Distribution ({label}): "
                  f"{n1}/{n} correct ({n1/n*100:.1f}%) | "
                  f"{n-n1}/{n} incorrect ({(n-n1)/n*100:.1f}%)")
    
    print("\n" + "#" * 70)
    print("#  BACKTEST COMPLETE")
    print("#" * 70)
    
    return df, train_df, test_df


# =============================================================================
# PART 3: RUN EVERYTHING
# =============================================================================

if __name__ == "__main__":
    
    # Change this to the location of your data files
    DATA_DIR = '/Users/samweg/Desktop/Brandeis/SSGA/Data'
    
    BCOM_PATH = f'{DATA_DIR}/BCOM_INDEX.xlsx'
    SPX_PATH = f'{DATA_DIR}/SNP500.xlsx'
    TREASURY_PATH = f'{DATA_DIR}/US_10Y_treasury_bonds.xlsx'
    IG_PATH = f'{DATA_DIR}/US_IG_corporate_bonds.xlsx'
    
    # --- Step 1: Run the M1 Signal Model ---
    results = run_signal_model(
        bcom_path=BCOM_PATH,
        spx_path=SPX_PATH,
        treasury_path=TREASURY_PATH,
        ig_path=IG_PATH,
        buy_threshold=0.5,
        sell_threshold=-0.5
    )
    
    # --- Step 2: Show the latest signal ---
    latest = get_latest_signal(results)
    
    # --- Step 3: Run the backtest ---
    full_df, train_df, test_df = run_backtest(results)
    
    # --- Step 4: Save results ---
    output_cols = [
        'Date', 'SPX_Price', 'SPX_Return',
        'Composite_Score', 'Signal', 'Position',
        'Forward_Return', 'Strategy_Return', 'BH_Return',
        'Strategy_Cumulative', 'BH_Cumulative',
        # NEW: Benchmark columns
        'Bench_6040_Return', 'Bench_6040_Cumulative',
        'Bench_SMA_Return', 'Bench_SMA_Cumulative',
        'Bench_Random_Return', 'Bench_Random_Cumulative',
        # Existing
        'Meta_Label', 'Rolling_Sharpe', 'Rolling_HitRate',
        'Z1_Momentum', 'Z2_CrossAsset', 'Z3_Yield', 'Z4_Credit', 'Z5_Vol',
        'Set'
    ]
    full_df[output_cols].to_csv(f'{DATA_DIR}/backtest_results.csv', index=False)
    print(f"\nResults saved to {DATA_DIR}/backtest_results.csv")