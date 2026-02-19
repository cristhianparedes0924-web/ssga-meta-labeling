import pandas as pd
import numpy as np
import warnings
from scipy.signal import hilbert
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings('ignore')


# =============================================================================
# PART 1: M1 PRIMARY SIGNAL MODEL — TREND AWARE STRATEGY
# =============================================================================
# Replaces the original z-score composite model with the Trend Aware strategy:
#   - 5 core features: 4 Hilbert Phase cycles + 3M SPX momentum
#   - Gradient Boosting ML for dynamic multi-asset weight allocation
#   - Signals derived from equity weight relative to baseline (25%)
#
# The Hilbert Phase tells you WHERE you are in a cycle (like a clock),
# not just how high the price is. This gives earlier regime detection
# than standard moving average or momentum indicators.
# =============================================================================


# --------------------------------------------------------------------------
# 1) DATA LOADING
# --------------------------------------------------------------------------

def load_data(bcom_path, spx_path, treasury_path, ig_path):
    """
    Load and merge all 4 data sources from Excel files.
    Returns a DataFrame with Date index and price columns for each asset.
    Compatible with the original project's Excel format (header at row 6).
    """

    def load_single_file(filepath, col_name):
        df = pd.read_excel(filepath)
        data_df = df.iloc[6:].copy()
        data_df.columns = ['Date', 'Price', 'Return']
        data_df['Date'] = pd.to_datetime(data_df['Date'], errors='coerce')
        data_df['Price'] = pd.to_numeric(data_df['Price'], errors='coerce')
        data_df['Return'] = pd.to_numeric(data_df['Return'], errors='coerce')
        data_df = data_df.dropna(subset=['Date'])
        data_df = data_df.sort_values('Date').reset_index(drop=True)
        data_df = data_df.rename(columns={
            'Price': f'{col_name}_Price',
            'Return': f'{col_name}_Return'
        })
        return data_df

    bcom = load_single_file(bcom_path, 'commodities')
    spx = load_single_file(spx_path, 'spx')
    treasury = load_single_file(treasury_path, 'rates')
    ig_bonds = load_single_file(ig_path, 'credit')

    # Merge all on Date
    merged = spx.merge(bcom, on='Date', how='outer')
    merged = merged.merge(treasury, on='Date', how='outer')
    merged = merged.merge(ig_bonds, on='Date', how='outer')
    merged = merged.sort_values('Date').reset_index(drop=True)

    # Forward-fill prices (handle minor gaps), then drop rows still missing
    price_cols = [c for c in merged.columns if '_Price' in c]
    merged[price_cols] = merged[price_cols].ffill()
    merged = merged.dropna(subset=price_cols)

    return merged


# --------------------------------------------------------------------------
# 2) HILBERT PHASE FEATURE ENGINEERING
# --------------------------------------------------------------------------

def get_phase(series):
    """
    Compute the Hilbert Phase of a price series.

    Steps:
      1. Detrend: subtract 12-month rolling mean to isolate the cyclical component
      2. Hilbert Transform: create a 90-degree shifted "shadow wave"
      3. Phase Angle: arctan2 of imaginary vs real part

    Output interpretation (like a clock):
      -π  (6:00 AM) → Bottom / Recovery  (Strong Buy)
       0  (12:00 PM) → Expansion          (Hold / Ramp Up)
      +π  (6:00 PM) → Peak / Exhaustion   (Sell / Reduce)

    The edge: Phase turns BEFORE price does, giving earlier signals
    than standard moving averages.
    """
    detrended = series - series.rolling(12).mean()
    analytic = hilbert(detrended.fillna(0))
    return np.angle(analytic)


def create_features(df):
    """
    Create the 5 core features for the Trend Aware model:

    1. rates_phase    — Bond Cycle: the "Gravity" signal
       Rising rates tightening cycle → reduces fair value of risk assets

    2. credit_phase   — Credit Health: the "Pulse"
       Widening spreads = earliest recession warning → defensive shift

    3. spx_phase      — Market Cycle: the "Top Caller"
       Detects when stock market wave is mathematically exhausted

    4. commodities_phase — Inflation Cycle: the "Pivot" signal
       Identifies inflationary regimes → capital rotates to commodities

    5. spx_roc_3m     — Price Momentum: the "Gas Pedal"
       Trend-following filter; don't fight the tape even if fundamentals look bad
    """
    df = df.copy()

    # Hilbert Phases (4 cycles)
    df['rates_phase'] = get_phase(df['rates_Price'])
    df['credit_phase'] = get_phase(df['credit_Price'])
    df['spx_phase'] = get_phase(df['spx_Price'])
    df['commodities_phase'] = get_phase(df['commodities_Price'])

    # Momentum
    df['spx_roc_3m'] = df['spx_Price'].pct_change(3)

    return df


# --------------------------------------------------------------------------
# 3) ML WEIGHT ALLOCATION (Walk-Forward Gradient Boosting)
# --------------------------------------------------------------------------

ML_FEATURES = ['rates_phase', 'credit_phase', 'spx_phase', 'commodities_phase', 'spx_roc_3m']
ASSET_NAMES = ['spx', 'rates', 'commodities', 'credit']
MIN_WEIGHT = 0.00
MAX_WEIGHT = 1.00
WARM_UP = 24  # 2-year warm-up before ML kicks in


def run_ml_allocation(df):
    """
    Walk-forward Gradient Boosting allocation.

    For each month t:
      1. Train GBR on all data up to t (expanding window, min 24 months)
      2. Predict next-month active return (vs equal-weight benchmark) for each asset
      3. Convert predictions to z-score tilts around equal-weight baseline (25%)
      4. Clip to [0%, 100%] and normalize to sum to 100%

    The model learns which phase configurations historically preceded
    outperformance of each asset class relative to the benchmark.

    Returns the df with weight columns and strategy returns.
    """
    df = df.copy()

    # Build returns for each asset
    return_cols = {}
    for asset in ASSET_NAMES:
        ret_col = f'{asset}_Return'
        if ret_col not in df.columns or df[ret_col].isna().all():
            # Compute from price if return column is missing/empty
            df[ret_col] = df[f'{asset}_Price'].pct_change()
        return_cols[asset] = ret_col

    # Equal-weight benchmark return each month
    ret_matrix = df[[return_cols[a] for a in ASSET_NAMES]].copy()
    ret_matrix.columns = ASSET_NAMES
    df['EW_Benchmark_Return'] = ret_matrix.mean(axis=1)

    # Targets: next-month active return vs equal-weight
    targets = pd.DataFrame(index=df.index)
    for asset in ASSET_NAMES:
        targets[asset] = ret_matrix[asset].shift(-1) - df['EW_Benchmark_Return'].shift(-1)
    # Don't drop NaN targets here; handle in the loop

    # Feature matrix (already computed)
    feature_df = df[ML_FEATURES].copy()

    # Walk-forward allocation
    n = len(df)
    weight_data = np.full((n, len(ASSET_NAMES)), 0.25)  # default equal-weight

    # Find first row where all features are valid
    valid_mask = feature_df.notna().all(axis=1) & targets.notna().all(axis=1)
    valid_indices = df.index[valid_mask].tolist()

    print(f"  Walk-forward ML allocation over {len(valid_indices)} valid months...")

    for i in range(n):
        if not valid_mask.iloc[i]:
            continue

        # Count how many valid training rows exist before this point
        train_mask = valid_mask.iloc[:i]
        n_train = train_mask.sum()

        allocation = pd.Series(0.25, index=ASSET_NAMES)

        if n_train >= WARM_UP:
            # Train on all valid data up to (not including) current month
            train_idx = df.index[:i][train_mask.iloc[:i]]
            X_train = feature_df.loc[train_idx]
            y_train = targets.loc[train_idx]

            X_curr = feature_df.iloc[i:i+1]

            preds = {}
            for asset in ASSET_NAMES:
                model = GradientBoostingRegressor(
                    n_estimators=30, max_depth=2,
                    learning_rate=0.1, random_state=42
                )
                model.fit(X_train, y_train[asset])
                preds[asset] = model.predict(X_curr)[0]

            # Convert predictions to tilt via z-score
            score = pd.Series(preds)
            if score.std() > 1e-6:
                z = (score - score.mean()) / score.std()
                allocation = allocation + (z * 0.15)

        # Clip and normalize
        allocation = allocation.clip(MIN_WEIGHT, MAX_WEIGHT)
        allocation = allocation / allocation.sum()
        weight_data[i] = allocation.values

    # Store weights
    for j, asset in enumerate(ASSET_NAMES):
        df[f'W_{asset}'] = weight_data[:, j]

    return df


# --------------------------------------------------------------------------
# 4) SIGNAL DERIVATION FROM WEIGHTS
# --------------------------------------------------------------------------
# The backtest framework expects BUY/SELL/HOLD signals.
# We derive these from the equity (SPX) weight relative to the 25% baseline.

def generate_signals(df, overweight_threshold=0.30, underweight_threshold=0.20):
    """
    Derive BUY / SELL / HOLD from the ML-assigned equity weight.

    - BUY:  SPX weight > 30%  (model is overweight equities → bullish)
    - SELL: SPX weight < 20%  (model is underweight equities → bearish)
    - HOLD: Between 20% and 30% (near baseline → neutral)

    These thresholds correspond to a ±5% tilt from the 25% equal-weight
    baseline, which aligns with the z-score * 0.15 tilt magnitude.
    """
    df = df.copy()

    def get_signal(w):
        if pd.isna(w):
            return None
        elif w > overweight_threshold:
            return 'BUY'
        elif w < underweight_threshold:
            return 'SELL'
        else:
            return 'HOLD'

    df['Signal'] = df['W_spx'].apply(get_signal)

    # Also store the composite score as the equity weight deviation from baseline
    # (positive = bullish, negative = bearish) for compatibility with downstream code
    df['Composite_Score'] = (df['W_spx'] - 0.25) / 0.25  # normalized deviation

    return df


# --------------------------------------------------------------------------
# 5) MAIN MODEL RUNNER
# --------------------------------------------------------------------------

def run_signal_model(bcom_path, spx_path, treasury_path, ig_path,
                     weights=None, buy_threshold=0.30, sell_threshold=0.20):
    """
    Run the full Trend Aware M1 model pipeline:
      1. Load data from Excel files
      2. Engineer Hilbert Phase features + momentum
      3. Walk-forward GBR allocation
      4. Derive BUY/SELL/HOLD signals from equity weight

    Parameters weights, buy_threshold, sell_threshold are kept for API
    compatibility with the original model runner, but semantics differ:
      - buy_threshold  → overweight threshold for SPX weight (default 0.30)
      - sell_threshold → underweight threshold for SPX weight (default 0.20)
      - weights        → not used (ML determines allocation)
    """
    print("=" * 70)
    print("  TREND AWARE M1 — Dynamic Regime Adaptation Strategy")
    print("=" * 70)

    # Step 1: Load data
    print("\n[1/4] Loading data...")
    df = load_data(bcom_path, spx_path, treasury_path, ig_path)
    print(f"       {len(df)} months loaded: {df['Date'].min().strftime('%Y-%m')} → {df['Date'].max().strftime('%Y-%m')}")

    # Step 2: Feature engineering
    print("[2/4] Computing Hilbert Phase features...")
    df = create_features(df)

    # Step 3: ML allocation
    print("[3/4] Running walk-forward ML allocation...")
    df = run_ml_allocation(df)

    # Step 4: Derive signals
    print("[4/4] Generating signals from equity weight...")
    df = generate_signals(df, overweight_threshold=buy_threshold,
                          underweight_threshold=sell_threshold)

    # Summary
    valid = df.dropna(subset=['Signal'])
    sig_counts = valid['Signal'].value_counts()
    print(f"\n  Signal distribution:")
    for s in ['BUY', 'HOLD', 'SELL']:
        cnt = sig_counts.get(s, 0)
        print(f"    {s:6s}: {cnt:3d}  ({cnt/len(valid)*100:.1f}%)")

    print("\nDone!")
    return df


def get_latest_signal(df):
    """Get the most recent signal and its component features."""
    latest = df.dropna(subset=['Signal']).iloc[-1]

    print("\n" + "=" * 60)
    print(f"LATEST SIGNAL: {latest['Signal']}")
    print("=" * 60)
    print(f"Date: {latest['Date'].strftime('%Y-%m-%d')}")
    print(f"Composite Score (equity weight deviation): {latest['Composite_Score']:.3f}")
    print(f"\nAsset Weights:")
    for asset in ASSET_NAMES:
        w = latest[f'W_{asset}']
        bar = '█' * int(w * 40)
        print(f"  {asset:>14s}: {w:5.1%}  {bar}")
    print(f"\nHilbert Phase Features:")
    print(f"  Rates Phase (Gravity):       {latest['rates_phase']:+.3f}")
    print(f"  Credit Phase (Pulse):        {latest['credit_phase']:+.3f}")
    print(f"  SPX Phase (Top Caller):      {latest['spx_phase']:+.3f}")
    print(f"  Commodities Phase (Pivot):   {latest['commodities_phase']:+.3f}")
    print(f"  SPX 3M Momentum (Gas Pedal): {latest['spx_roc_3m']:+.4f}")
    print("=" * 60)

    return latest


# =============================================================================
# PART 2: BACKTEST MODULE
# =============================================================================
# Adapted from the original backtest module. Key difference:
#   - Strategy returns are now MULTI-ASSET weighted returns (not just SPX * position)
#   - Signals (BUY/SELL/HOLD) are derived from equity weight for meta-labeling
#   - All benchmarks, metrics, and meta-labels preserved
# =============================================================================


# --------------------------------------------------------------------------
# 1) POSITION MAPPING
# --------------------------------------------------------------------------

def map_signals_to_positions(df):
    """
    Convert BUY / SELL / HOLD signals into numerical positions.

    For the Trend Aware model, the actual allocation is multi-asset
    (stored in W_spx, W_rates, etc.), but we still map signals to
    a single equity position for classification metrics and meta-labels.

    Position logic:
      BUY  →  +1  (overweight equities)
      SELL →   0  (underweight equities, long-only)
      HOLD →  carry forward
    """
    long_only = True

    df = df.copy()

    signal_map = {
        'BUY': 1,
        'SELL': 0 if long_only else -1,
        'HOLD': np.nan
    }

    df['Position'] = df['Signal'].map(signal_map)
    df['Position'] = df['Position'].ffill().fillna(0)

    return df


# --------------------------------------------------------------------------
# 2) RETURN CALCULATION — MULTI-ASSET WEIGHTED
# --------------------------------------------------------------------------

def calculate_strategy_returns(df):
    """
    Calculate strategy returns based on the ML-determined asset weights.

    Unlike the original model (SPX * position), the Trend Aware strategy
    earns a BLENDED return across all 4 asset classes each month:

        r_strategy(t) = Σ W_asset(t) * r_asset(t+1)

    The weights at time t are applied to returns at t+1 (no look-ahead).

    We also compute a buy-and-hold SPX benchmark for comparison.
    """
    df = df.copy()

    # Forward returns for each asset (earned next period)
    for asset in ASSET_NAMES:
        df[f'{asset}_Fwd_Return'] = df[f'{asset}_Return'].shift(-1)

    # Strategy return: weighted sum of forward returns
    df['Strategy_Return'] = 0.0
    for asset in ASSET_NAMES:
        df['Strategy_Return'] += df[f'W_{asset}'].shift(0) * df[f'{asset}_Fwd_Return']

    # Forward SPX return (for signal evaluation and buy-and-hold)
    df['Forward_Return'] = df['spx_Return'].shift(-1)

    # Buy-and-hold SPX benchmark
    df['BH_Return'] = df['Forward_Return']

    # Also store SPX price/return under legacy column names for benchmark functions
    df['SPX_Price'] = df['spx_Price']
    df['SPX_Return'] = df['spx_Return']
    df['Treasury10Y_Return'] = df['rates_Return']

    # Cumulative returns (growth of $1)
    df['Strategy_Cumulative'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()
    df['BH_Cumulative'] = (1 + df['BH_Return'].fillna(0)).cumprod()

    return df


# --------------------------------------------------------------------------
# 3) BENCHMARK SUITE
# --------------------------------------------------------------------------

def calculate_benchmark_6040(df):
    """
    BENCHMARK 1: 60/40 Portfolio (60% S&P 500, 40% 10Y Treasury)

    The classic institutional balanced benchmark. If the Trend Aware
    model can't beat passive 60/40, the ML allocation isn't adding value.
    """
    df = df.copy()

    fwd_spx = df['SPX_Return'].shift(-1)
    fwd_treasury = df['Treasury10Y_Return'].shift(-1)

    df['Bench_6040_Return'] = 0.60 * fwd_spx + 0.40 * fwd_treasury
    df['Bench_6040_Cumulative'] = (1 + df['Bench_6040_Return'].fillna(0)).cumprod()

    return df


def calculate_benchmark_ew(df):
    """
    BENCHMARK 2: Equal-Weight Portfolio (25% each asset)

    Since the Trend Aware model tilts around an equal-weight baseline,
    this benchmark directly measures whether the ML tilts add value.
    """
    df = df.copy()

    fwd_returns = 0.0
    for asset in ASSET_NAMES:
        fwd_returns += 0.25 * df[f'{asset}_Return'].shift(-1)

    df['Bench_EW_Return'] = fwd_returns
    df['Bench_EW_Cumulative'] = (1 + df['Bench_EW_Return'].fillna(0)).cumprod()

    return df


def calculate_benchmark_sma(df, short_window=10, long_window=12):
    """
    BENCHMARK 3: SMA Crossover on S&P 500

    Tests whether the Hilbert Phase + ML complexity adds value
    beyond a simple trend-following rule using price alone.
    """
    df = df.copy()

    df['SMA_Short'] = df['SPX_Price'].rolling(short_window).mean()
    df['SMA_Long'] = df['SPX_Price'].rolling(long_window).mean()
    df['SMA_Position'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1, 0)

    fwd_spx = df['SPX_Return'].shift(-1)
    df['Bench_SMA_Return'] = df['SMA_Position'] * fwd_spx
    df['Bench_SMA_Cumulative'] = (1 + df['Bench_SMA_Return'].fillna(0)).cumprod()

    return df


def calculate_benchmark_random(df, n_simulations=1000, seed=42):
    """
    BENCHMARK 4: Random Signal Baseline (Monte Carlo)

    Generates 1,000 random signal sequences with the same BUY/SELL/HOLD
    distribution as M1 to test statistical significance.
    """
    df = df.copy()
    rng = np.random.RandomState(seed)

    valid_signals = df.dropna(subset=['Signal'])
    signal_counts = valid_signals['Signal'].value_counts(normalize=True)
    signal_labels = signal_counts.index.tolist()
    signal_probs = signal_counts.values.tolist()

    n_periods = len(valid_signals)
    fwd_returns = df['Forward_Return'].values

    sim_sharpes = []
    sim_cumulative_returns = []

    for _ in range(n_simulations):
        random_signals = rng.choice(signal_labels, size=n_periods, p=signal_probs)

        positions = np.zeros(n_periods)
        for t in range(n_periods):
            if random_signals[t] == 'BUY':
                positions[t] = 1
            elif random_signals[t] == 'SELL':
                positions[t] = 0
            else:
                positions[t] = positions[t-1] if t > 0 else 0

        valid_indices = valid_signals.index.tolist()
        sim_returns = np.array([
            positions[j] * fwd_returns[idx] if not np.isnan(fwd_returns[idx]) else 0
            for j, idx in enumerate(valid_indices)
        ])

        if sim_returns.std() > 0:
            sharpe = (sim_returns.mean() * 12) / (sim_returns.std() * np.sqrt(12))
        else:
            sharpe = 0
        sim_sharpes.append(sharpe)
        sim_cumulative_returns.append(np.cumprod(1 + sim_returns))

    sim_sharpes = np.array(sim_sharpes)

    median_idx = np.argsort(sim_sharpes)[n_simulations // 2]
    median_cum = sim_cumulative_returns[median_idx]

    valid_indices = valid_signals.index.tolist()
    df['Bench_Random_Return'] = np.nan
    df['Bench_Random_Cumulative'] = np.nan

    median_returns = np.diff(median_cum, prepend=1.0) / np.concatenate([[1.0], median_cum[:-1]])

    for j, idx in enumerate(valid_indices):
        if j < len(median_returns):
            df.loc[idx, 'Bench_Random_Return'] = median_returns[j]
            df.loc[idx, 'Bench_Random_Cumulative'] = median_cum[j]

    df['Bench_Random_Cumulative'] = df['Bench_Random_Cumulative'].ffill().fillna(1.0)

    df.attrs['random_sim_sharpes'] = sim_sharpes
    df.attrs['random_sim_median_sharpe'] = np.median(sim_sharpes)
    df.attrs['random_sim_95th_sharpe'] = np.percentile(sim_sharpes, 95)
    df.attrs['random_sim_5th_sharpe'] = np.percentile(sim_sharpes, 5)

    return df


def calculate_all_benchmarks(df):
    """Compute all benchmarks."""
    print("  Computing benchmarks...")

    print("    → 60/40 Portfolio")
    df = calculate_benchmark_6040(df)

    print("    → Equal-Weight (25% each)")
    df = calculate_benchmark_ew(df)

    print("    → SMA Crossover (10M / 12M)")
    df = calculate_benchmark_sma(df)

    print("    → Random Signal Baseline (1,000 simulations)")
    df = calculate_benchmark_random(df)

    print("  Benchmarks complete.")
    return df


# --------------------------------------------------------------------------
# 4) PERFORMANCE METRICS
# --------------------------------------------------------------------------

def calculate_max_drawdown(cumulative_returns):
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    return drawdown.min()


def calculate_performance_metrics(df, periods_per_year=12):
    """Calculate comprehensive performance metrics for strategy and all benchmarks."""

    valid = df.dropna(subset=['Strategy_Return', 'Forward_Return']).copy()
    strat_rets = valid['Strategy_Return']
    bh_rets = valid['BH_Return']

    def compute_metrics(returns, label):
        n_periods = len(returns)
        cumulative = (1 + returns).prod()
        n_years = n_periods / periods_per_year
        ann_return = cumulative ** (1 / n_years) - 1 if n_years > 0 else 0
        ann_vol = returns.std() * np.sqrt(periods_per_year)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        cum_series = (1 + returns).cumprod()
        max_dd = calculate_max_drawdown(cum_series)
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
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

    strat_metrics = compute_metrics(strat_rets, 'Trend Aware (M1)')
    bh_metrics = compute_metrics(bh_rets, 'Buy & Hold SPX')

    bench_metrics = {}

    # 60/40
    b6040 = valid['Bench_6040_Return'].dropna()
    if len(b6040) > 0:
        bench_metrics['60/40 Portfolio'] = compute_metrics(b6040, '60/40 Portfolio')

    # Equal-Weight
    bew = valid['Bench_EW_Return'].dropna()
    if len(bew) > 0:
        bench_metrics['Equal-Weight'] = compute_metrics(bew, 'Equal-Weight (25%)')

    # SMA
    bsma = valid['Bench_SMA_Return'].dropna()
    if len(bsma) > 0:
        bench_metrics['SMA Crossover'] = compute_metrics(bsma, 'SMA Crossover')

    # Random
    brand = valid['Bench_Random_Return'].dropna()
    if len(brand) > 0:
        bench_metrics['Random Baseline'] = compute_metrics(brand, 'Random Baseline')

    # Strategy-specific metrics
    position_changes = (valid['Position'].diff() != 0).sum()
    strat_metrics['Number of Trades (Position Changes)'] = int(position_changes)

    time_in_market = (valid['Position'] != 0).sum() / len(valid)
    strat_metrics['Time in Market'] = f"{time_in_market:.3f}  ({time_in_market*100:.1f}%)"

    n_long = (valid['Position'] == 1).sum()
    n_flat = (valid['Position'] == 0).sum()
    n_short = (valid['Position'] == -1).sum()
    strat_metrics['Periods Long / Flat / Short'] = f"{n_long} / {n_flat} / {n_short}"

    # Average weights (Trend Aware specific)
    for asset in ASSET_NAMES:
        wcol = f'W_{asset}'
        if wcol in valid.columns:
            avg_w = valid[wcol].mean()
            strat_metrics[f'Avg Weight {asset}'] = f"{avg_w:.3f}  ({avg_w*100:.1f}%)"

    return strat_metrics, bh_metrics, bench_metrics


def print_performance(strat_metrics, bh_metrics, bench_metrics=None):
    """Pretty-print performance comparison across all benchmarks."""

    all_benchmarks = {'Trend Aware (M1)': strat_metrics, 'Buy & Hold SPX': bh_metrics}
    if bench_metrics:
        all_benchmarks.update(bench_metrics)

    labels = list(all_benchmarks.keys())
    col_width = 22
    header = f"\n{'Metric':<35}"
    for label in labels:
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

    print()
    print("Strategy-Specific Metrics:")
    print("-" * 60)
    strat_only_keys = [
        'Number of Trades (Position Changes)',
        'Time in Market',
        'Periods Long / Flat / Short',
    ]
    # Add weight metrics
    for asset in ASSET_NAMES:
        strat_only_keys.append(f'Avg Weight {asset}')

    for key in strat_only_keys:
        val = strat_metrics.get(key, 'N/A')
        if val != 'N/A':
            print(f"  {key:<38} {val}")


# --------------------------------------------------------------------------
# 5) RANDOM BASELINE SIGNIFICANCE TEST
# --------------------------------------------------------------------------

def print_random_significance(df, strategy_sharpe_str):
    """Print statistical significance of M1 vs random signals."""

    if 'random_sim_sharpes' not in df.attrs:
        print("  (Random simulation data not available)")
        return

    sim_sharpes = df.attrs['random_sim_sharpes']
    strategy_sharpe = float(strategy_sharpe_str.split()[0])

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
        print(f"  Result:                    SIGNIFICANT (p < 0.05)")
        print(f"                             M1 outperforms {percentile:.0f}% of random strategies")
    elif p_value < 0.10:
        print(f"  Result:                    MARGINALLY SIGNIFICANT (p < 0.10)")
    else:
        print(f"  Result:                    NOT SIGNIFICANT (p = {p_value:.3f})")
        print(f"                             M1 performance could be due to chance")
    print("=" * 70)


# --------------------------------------------------------------------------
# 6) CLASSIFICATION METRICS
# --------------------------------------------------------------------------

def calculate_signal_accuracy(df):
    """Evaluate signal quality using classification metrics."""

    valid = df.dropna(subset=['Signal', 'Forward_Return']).copy()

    buy_signals = valid[valid['Signal'] == 'BUY']
    n_buy = len(buy_signals)
    n_buy_correct = (buy_signals['Forward_Return'] > 0).sum() if n_buy > 0 else 0
    buy_precision = n_buy_correct / n_buy if n_buy > 0 else 0

    sell_signals = valid[valid['Signal'] == 'SELL']
    n_sell = len(sell_signals)
    n_sell_correct = (sell_signals['Forward_Return'] < 0).sum() if n_sell > 0 else 0
    sell_precision = n_sell_correct / n_sell if n_sell > 0 else 0

    valid['Signal_Correct'] = False
    valid.loc[(valid['Signal'] == 'BUY') & (valid['Forward_Return'] > 0), 'Signal_Correct'] = True
    valid.loc[(valid['Signal'] == 'SELL') & (valid['Forward_Return'] < 0), 'Signal_Correct'] = True

    active_signals = valid[valid['Signal'].isin(['BUY', 'SELL'])]
    overall_accuracy = active_signals['Signal_Correct'].mean() if len(active_signals) > 0 else 0

    positive_periods = valid[valid['Forward_Return'] > 0]
    n_positive = len(positive_periods)
    n_caught = len(positive_periods[positive_periods['Signal'] == 'BUY'])
    recall = n_caught / n_positive if n_positive > 0 else 0

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
    Generate meta-labels: binary {0, 1} indicating whether M1's
    directional signal was correct.

    BUY correct  if forward SPX return > 0  → meta_label = 1
    BUY wrong    if forward SPX return ≤ 0  → meta_label = 0
    SELL correct if forward SPX return < 0  → meta_label = 1
    SELL wrong   if forward SPX return ≥ 0  → meta_label = 0
    HOLD → excluded (no directional prediction)
    """
    df = df.copy()
    df['Meta_Label'] = np.nan

    df.loc[(df['Signal'] == 'BUY') & (df['Forward_Return'] > 0), 'Meta_Label'] = 1
    df.loc[(df['Signal'] == 'BUY') & (df['Forward_Return'] <= 0), 'Meta_Label'] = 0
    df.loc[(df['Signal'] == 'SELL') & (df['Forward_Return'] < 0), 'Meta_Label'] = 1
    df.loc[(df['Signal'] == 'SELL') & (df['Forward_Return'] >= 0), 'Meta_Label'] = 0

    meta_valid = df.dropna(subset=['Meta_Label'])
    n_total = len(meta_valid)
    if n_total > 0:
        n_correct = (meta_valid['Meta_Label'] == 1).sum()
        n_incorrect = (meta_valid['Meta_Label'] == 0).sum()
        print(f"\nMeta-Labels Generated:")
        print(f"  Total active signals:  {n_total}")
        print(f"  Correct (label=1):     {n_correct}  ({n_correct/n_total*100:.1f}%)")
        print(f"  Incorrect (label=0):   {n_incorrect}  ({n_incorrect/n_total*100:.1f}%)")

    return df


# --------------------------------------------------------------------------
# 8) ROLLING PERFORMANCE
# --------------------------------------------------------------------------

def calculate_rolling_metrics(df, window=12):
    """Calculate rolling Sharpe and hit rate for evaluation / M2 features."""
    df = df.copy()

    rolling_mean = df['Strategy_Return'].rolling(window).mean() * 12
    rolling_std = df['Strategy_Return'].rolling(window).std() * np.sqrt(12)
    df['Rolling_Sharpe'] = rolling_mean / rolling_std

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
    """Sequential 60/40 train/test split. No shuffling."""

    valid_df = df.dropna(subset=['Signal']).copy()
    n = len(valid_df)
    split_idx = int(n * train_ratio)
    split_date = valid_df.iloc[split_idx]['Date']

    df = df.copy()
    df['Set'] = 'pre-signal'
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
    Full backtest pipeline for the Trend Aware M1 model:
      1. Map signals to positions
      2. Calculate multi-asset weighted strategy returns
      3. Calculate all benchmarks (60/40, Equal-Weight, SMA, Random)
      4. Generate meta-labels
      5. Calculate rolling metrics
      6. Split into train/test (60/40 sequential)
      7. Print performance metrics for BOTH sets
      8. Print classification metrics for BOTH sets
      9. Print random signal significance test
      10. Return enriched DataFrames
    """

    print("\n" + "#" * 70)
    print("#  RUNNING BACKTEST ON TREND AWARE M1 MODEL")
    print("#" * 70)

    # Step 1: Positions
    df = map_signals_to_positions(df)

    # Step 2: Returns (multi-asset weighted)
    df = calculate_strategy_returns(df)

    # Step 3: Benchmarks
    df = calculate_all_benchmarks(df)

    # Step 4: Meta-labels
    df = generate_meta_labels(df)

    # Step 5: Rolling metrics
    df = calculate_rolling_metrics(df)

    # Step 6: Train/Test split
    df, train_df, test_df = train_test_split(df, train_ratio)

    # Step 7: Performance — BOTH sets
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

    # Step 9: Random signal significance
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

    # ── Update this to your local data directory ──
    DATA_DIR = '/Users/samweg/Desktop/Brandeis/SSGA/Data'

    BCOM_PATH = f'{DATA_DIR}/BCOM_INDEX.xlsx'
    SPX_PATH = f'{DATA_DIR}/SNP500.xlsx'
    TREASURY_PATH = f'{DATA_DIR}/US_10Y_treasury_bonds.xlsx'
    IG_PATH = f'{DATA_DIR}/US_IG_corporate_bonds.xlsx'

    # --- Step 1: Run the Trend Aware M1 Signal Model ---
    results = run_signal_model(
        bcom_path=BCOM_PATH,
        spx_path=SPX_PATH,
        treasury_path=TREASURY_PATH,
        ig_path=IG_PATH,
        buy_threshold=0.30,   # SPX weight > 30% → BUY
        sell_threshold=0.20   # SPX weight < 20% → SELL
    )

    # --- Step 2: Show the latest signal ---
    latest = get_latest_signal(results)

    # --- Step 3: Run the backtest ---
    full_df, train_df, test_df = run_backtest(results)

    # --- Step 4: Save results ---
    output_cols = [
        'Date',
        # Asset prices and returns
        'spx_Price', 'spx_Return',
        'rates_Price', 'rates_Return',
        'commodities_Price', 'commodities_Return',
        'credit_Price', 'credit_Return',
        # Hilbert Phase features
        'rates_phase', 'credit_phase', 'spx_phase', 'commodities_phase',
        'spx_roc_3m',
        # ML Weights
        'W_spx', 'W_rates', 'W_commodities', 'W_credit',
        # Signal and position
        'Composite_Score', 'Signal', 'Position',
        # Returns
        'Forward_Return', 'Strategy_Return', 'BH_Return',
        'Strategy_Cumulative', 'BH_Cumulative',
        # Benchmarks
        'Bench_6040_Return', 'Bench_6040_Cumulative',
        'Bench_EW_Return', 'Bench_EW_Cumulative',
        'Bench_SMA_Return', 'Bench_SMA_Cumulative',
        'Bench_Random_Return', 'Bench_Random_Cumulative',
        # Meta-labels and evaluation
        'Meta_Label', 'Rolling_Sharpe', 'Rolling_HitRate',
        'Set'
    ]

    # Only include columns that exist
    existing_cols = [c for c in output_cols if c in full_df.columns]
    full_df[existing_cols].to_csv(f'{DATA_DIR}/backtest_results.csv', index=False)
    print(f"\nResults saved to {DATA_DIR}/backtest_results.csv")