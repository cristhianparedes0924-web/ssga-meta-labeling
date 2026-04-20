import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# PART 1: M1 PRIMARY SIGNAL MODEL — 5-INDICATOR COMPOSITE STRATEGY
# =============================================================================
# Primary indicators (ALL long-only, cash or invested):
#   1. AR(3)  — Autoregressive Price Dynamics
#   2. MACD Velocity — Time-Series Momentum via EMA acceleration
#   3. Credit Yield Spread — Cross-Asset macro leading indicator
#   4. Commodity Momentum — Inflation Proxy / Regime Detection
#   5. Realized Volatility Regime — Volatility Z-Score filter
#
# Final signal: MAJORITY VOTE across the 5 indicators (3+ longs → BUY)
# =============================================================================


# --------------------------------------------------------------------------
# 1) DATA LOADING
# --------------------------------------------------------------------------

def load_data(bcom_path, spx_path, treasury_path, ig_path):
    def load_single_file(filepath, col_name):
        df = pd.read_excel(filepath)
        data_df = df.iloc[6:].copy()
        data_df.columns = ['Date', 'Price', 'Return']
        data_df['Date'] = pd.to_datetime(data_df['Date'], errors='coerce')
        data_df['Price'] = pd.to_numeric(data_df['Price'], errors='coerce')
        data_df['Return'] = pd.to_numeric(data_df['Return'], errors='coerce')
        data_df = data_df.dropna(subset=['Date'])
        data_df = data_df.sort_values('Date').reset_index(drop=True)
        # Returns are stored as percentages (e.g., 3.26 = 3.26%) → convert to decimals
        data_df['Return'] = data_df['Return'] / 100.0
        data_df = data_df.rename(columns={
            'Price': f'{col_name}_Price',
            'Return': f'{col_name}_Return'
        })
        return data_df

    bcom     = load_single_file(bcom_path,     'commodities')
    spx      = load_single_file(spx_path,      'spx')
    treasury = load_single_file(treasury_path, 'rates')
    ig_bonds = load_single_file(ig_path,       'credit')

    merged = spx.merge(bcom, on='Date', how='outer')
    merged = merged.merge(treasury, on='Date', how='outer')
    merged = merged.merge(ig_bonds, on='Date', how='outer')
    merged = merged.sort_values('Date').reset_index(drop=True)

    price_cols = [c for c in merged.columns if '_Price' in c]
    merged[price_cols] = merged[price_cols].ffill()
    merged = merged.dropna(subset=price_cols)

    return merged


# --------------------------------------------------------------------------
# 2) INDICATOR 1 — AR(3) Autoregressive Price Dynamics
# --------------------------------------------------------------------------
# Fits an AR(3) model at each time t using an expanding window of SPX returns.
# Coefficients c0, c1, c2 are OLS-estimated on data up to t-1.
# Forecasted return = c0*r(t-1) + c1*r(t-2) + c2*r(t-3) + const
# Signal: LONG if forecasted return > 0, else FLAT.

def compute_ar3_signal(df, min_window=12):
    """
    Expanding-window OLS AR(3) on SPX monthly returns.
    Returns a binary series: 1 = long, 0 = flat.
    """
    ret = df['spx_Return'].values
    n   = len(ret)
    signal = np.zeros(n, dtype=float)
    forecast = np.full(n, np.nan)

    for t in range(min_window, n):
        # Build lagged feature matrix on data available up to t-1
        # We need at least 4 obs to form 3 lags + 1 target
        hist = ret[:t]
        if len(hist) < 4:
            continue

        # Construct lags: y(t-1), y(t-2), y(t-3)
        y  = hist[3:]          # targets
        X  = np.column_stack([
            np.ones(len(y)),
            hist[2:-1],        # lag 1
            hist[1:-2],        # lag 2
            hist[0:-3],        # lag 3
        ])

        # OLS: beta = (X'X)^-1 X'y
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            continue

        # Forecast for time t using ret[t-1], ret[t-2], ret[t-3]
        x_new = np.array([1.0, hist[-1], hist[-2], hist[-3]])
        f = x_new @ beta
        forecast[t] = f
        signal[t] = 1 if f > 0 else 0

    return signal, forecast


# --------------------------------------------------------------------------
# 3) INDICATOR 2 — MACD Velocity
# --------------------------------------------------------------------------
# Fast EMA (3m) − Slow EMA (10m) = MACD line.
# Velocity = diff of MACD line.
# Signal: LONG when velocity > 0 (and positive MACD), else FLAT.

def compute_macd_signal(df):
    """
    MACD (3, 10) velocity filter on SPX price.
    Long when MACD velocity is positive (trend accelerating upward).
    """
    price = df['spx_Price']

    ema_fast = price.ewm(span=3,  adjust=False).mean()
    ema_slow = price.ewm(span=10, adjust=False).mean()
    macd     = ema_fast - ema_slow
    velocity = macd.diff()

    # Long when velocity > 0 (acceleration positive)
    signal = (velocity > 0).astype(float)
    # Require also MACD > 0 to confirm trend direction
    signal = ((velocity > 0) & (macd > 0)).astype(float)

    return signal.values, macd.values, velocity.values


# --------------------------------------------------------------------------
# 4) INDICATOR 3 — Cross-Asset Credit Yield Spread
# --------------------------------------------------------------------------
# Spread = Treasury 10Y yield proxy − IG bond yield proxy
# We approximate yields from price series using pct_change (or use Return cols).
# ROC of spread over 3 months, inverted: narrowing spread → LONG.
# Also compare spread level to its 12-month rolling mean.

def compute_credit_spread_signal(df, lookback_roc=3, lookback_mean=12):
    """
    Computes an approximate credit spread as:
        spread ≈ rates_Return (proxy for Treasury yield changes) 
                  − credit_Return (proxy for IG bond yield changes)
    
    A NARROWING spread (negative 3m ROC of spread) → LONG.
    Spread below trailing 12m mean → LONG.
    Either condition satisfied → LONG.
    """
    # Use price-derived yields: negative return ≈ yield rising
    # Spread = Treasury yield - IG yield  (widening = Treasury yield rising faster OR IG falling more)
    # We use the PRICE levels; higher price = lower yield.
    # Spread proxy: (1/credit_price - 1/treasury_price) or simply price ratio.
    # Simplest robust approach: spread = -log(credit_Price) + log(rates_Price)
    #   (rates_Price is bond price ↑ = yield ↓, credit_Price same)
    # Actually the most straightforward: use the return difference as a yield-change proxy.
    # spread_level = cumsum of (rates_Return - credit_Return) as a synthetic spread index.

    spread_index = (df['rates_Return'] - df['credit_Return']).cumsum()

    # 3-month ROC of spread (positive = widening, negative = narrowing)
    spread_roc = spread_index.diff(lookback_roc)

    # Inverted ROC: positive when spread is NARROWING
    inv_roc = -spread_roc

    # Trailing 12m mean of spread level
    spread_mean_12m = spread_index.rolling(lookback_mean).mean()
    below_mean = (spread_index < spread_mean_12m).astype(float)

    # Long when spread narrowing OR spread below 12m mean
    signal = ((inv_roc > 0) | (below_mean == 1)).astype(float)
    signal = signal.fillna(0)

    return signal.values, spread_index.values, inv_roc.values


# --------------------------------------------------------------------------
# 5) INDICATOR 4 — Commodity Momentum (Inflation Proxy)
# --------------------------------------------------------------------------
# 6-month momentum of BCOM index.
# Z-score of 6m momentum vs. its long-term distribution (rolling 36m).
# FLAT when z-score > 1.5 (inflationary shock), else LONG.

def compute_commodity_signal(df, mom_window=6, zscore_lookback=36, threshold=1.5):
    """
    6-month commodity momentum z-score.
    FLAT if z-score > 1.5 (inflation shock), else LONG.
    """
    price = df['commodities_Price']
    mom   = price.pct_change(mom_window)

    roll_mean = mom.rolling(zscore_lookback).mean()
    roll_std  = mom.rolling(zscore_lookback).std()

    zscore = (mom - roll_mean) / roll_std.replace(0, np.nan)

    # Flat when commodity momentum spike > 1.5 std above long-run mean
    signal = (zscore <= threshold).astype(float)
    signal = signal.fillna(1)  # default LONG when insufficient data

    return signal.values, zscore.values


# --------------------------------------------------------------------------
# 6) INDICATOR 5 — Realized Volatility Regime Filter
# --------------------------------------------------------------------------
# Rolling 3m std of SPX returns.
# Z-score vs 36m lookback.
# FLAT if z-score > 1.0 (panic regime), else LONG.

def compute_vol_regime_signal(df, vol_window=3, zscore_lookback=36, threshold=1.0):
    """
    Realized volatility z-score filter.
    FLAT when z > 1.0 (high-volatility panic regime), else LONG.
    """
    ret = df['spx_Return']

    realized_vol = ret.rolling(vol_window).std() * np.sqrt(12)  # annualized

    roll_mean = realized_vol.rolling(zscore_lookback).mean()
    roll_std  = realized_vol.rolling(zscore_lookback).std()

    zscore = (realized_vol - roll_mean) / roll_std.replace(0, np.nan)

    signal = (zscore <= threshold).astype(float)
    signal = signal.fillna(1)

    return signal.values, zscore.values


# --------------------------------------------------------------------------
# 7) COMPOSITE SIGNAL — MAJORITY VOTE (3 of 5)
# --------------------------------------------------------------------------

def run_signal_model(bcom_path, spx_path, treasury_path, ig_path,
                     weights=None, buy_threshold=0.30, sell_threshold=0.20):
    """
    Full 5-Indicator M1 pipeline.
    Final signal = majority vote (≥3 longs out of 5 → BUY, else FLAT/SELL).
    """
    print("=" * 70)
    print("  5-INDICATOR M1 — Composite Primary Strategy")
    print("=" * 70)

    print("\n[1/6] Loading data...")
    df = load_data(bcom_path, spx_path, treasury_path, ig_path)
    print(f"       {len(df)} months: {df['Date'].min().strftime('%Y-%m')} → {df['Date'].max().strftime('%Y-%m')}")

    print("[2/6] Indicator 1: AR(3) Autoregressive Price Dynamics...")
    ar3_sig, ar3_forecast = compute_ar3_signal(df)
    df['I1_AR3']         = ar3_sig
    df['I1_AR3_Forecast'] = ar3_forecast

    print("[3/6] Indicator 2: MACD Velocity...")
    macd_sig, macd_line, macd_vel = compute_macd_signal(df)
    df['I2_MACD']     = macd_sig
    df['I2_MACD_Line'] = macd_line
    df['I2_MACD_Vel']  = macd_vel

    print("[4/6] Indicator 3: Credit Yield Spread...")
    cs_sig, cs_index, cs_roc = compute_credit_spread_signal(df)
    df['I3_CreditSpread']    = cs_sig
    df['I3_Spread_Index']    = cs_index
    df['I3_Spread_ROC_Inv']  = cs_roc

    print("[5/6] Indicator 4: Commodity Momentum (Inflation Proxy)...")
    com_sig, com_z = compute_commodity_signal(df)
    df['I4_CommodityMom']   = com_sig
    df['I4_Commodity_ZScore'] = com_z

    print("[6/6] Indicator 5: Realized Volatility Regime...")
    vol_sig, vol_z = compute_vol_regime_signal(df)
    df['I5_VolRegime']       = vol_sig
    df['I5_Vol_ZScore']      = vol_z

    # Composite vote
    indicator_cols = ['I1_AR3', 'I2_MACD', 'I3_CreditSpread', 'I4_CommodityMom', 'I5_VolRegime']
    df['Vote_Sum']       = df[indicator_cols].sum(axis=1)
    df['Composite_Score'] = df['Vote_Sum'] / 5.0  # normalized 0-1

    # Signal: ≥3 votes → BUY, else SELL (cash)
    def assign_signal(row):
        # Need all indicators to have data (non-NaN)
        vals = [row[c] for c in indicator_cols]
        if any(pd.isna(v) for v in vals):
            return None
        return 'BUY' if sum(vals) >= 3 else 'SELL'

    df['Signal'] = df.apply(assign_signal, axis=1)

    valid = df.dropna(subset=['Signal'])
    sig_counts = valid['Signal'].value_counts()
    print(f"\n  Signal distribution:")
    for s in ['BUY', 'SELL']:
        cnt = sig_counts.get(s, 0)
        print(f"    {s:6s}: {cnt:3d}  ({cnt/len(valid)*100:.1f}%)")

    # Individual indicator agreement
    print(f"\n  Indicator LONG rates:")
    for col in indicator_cols:
        rate = df[col].mean()
        print(f"    {col:<25s}: {rate:.1%} long")

    print("\nDone!")
    return df


def get_latest_signal(df):
    latest = df.dropna(subset=['Signal']).iloc[-1]
    print("\n" + "=" * 60)
    print(f"LATEST SIGNAL: {latest['Signal']}")
    print("=" * 60)
    print(f"Date:            {latest['Date'].strftime('%Y-%m-%d')}")
    print(f"Composite Score: {latest['Composite_Score']:.2f}  (votes: {int(latest['Vote_Sum'])}/5)")
    print(f"\nIndicator Breakdown:")
    ind_names = {
        'I1_AR3':          'AR(3) Autoregressive',
        'I2_MACD':         'MACD Velocity',
        'I3_CreditSpread': 'Credit Spread',
        'I4_CommodityMom': 'Commodity Momentum',
        'I5_VolRegime':    'Vol Regime Filter',
    }
    for col, name in ind_names.items():
        v = int(latest[col])
        status = "LONG ✓" if v == 1 else "FLAT  ✗"
        print(f"  {name:<28s}: {status}")
    print("=" * 60)
    return latest


# =============================================================================
# PART 2: BACKTEST MODULE
# =============================================================================

ASSET_NAMES = ['spx', 'rates', 'commodities', 'credit']


def map_signals_to_positions(df):
    df = df.copy()
    df['Position'] = df['Signal'].map({'BUY': 1, 'SELL': 0})
    df['Position'] = df['Position'].ffill().fillna(0)
    return df


def calculate_strategy_returns(df):
    """
    Strategy = fully invested in SPX when BUY (position=1), cash when SELL.
    Forward SPX return used (no look-ahead).
    """
    df = df.copy()
    df['Forward_Return'] = df['spx_Return'].shift(-1)
    df['BH_Return']      = df['Forward_Return']

    df['Strategy_Return'] = df['Position'] * df['Forward_Return']

    df['SPX_Price']         = df['spx_Price']
    df['SPX_Return']        = df['spx_Return']
    df['Treasury10Y_Return'] = df['rates_Return']

    df['Strategy_Cumulative'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()
    df['BH_Cumulative']       = (1 + df['BH_Return'].fillna(0)).cumprod()

    return df


# ──── Benchmarks ────────────────────────────────────────────────────────────

def calculate_benchmark_6040(df):
    df = df.copy()
    fwd_spx      = df['SPX_Return'].shift(-1)
    fwd_treasury = df['Treasury10Y_Return'].shift(-1)
    df['Bench_6040_Return']      = 0.60 * fwd_spx + 0.40 * fwd_treasury
    df['Bench_6040_Cumulative']  = (1 + df['Bench_6040_Return'].fillna(0)).cumprod()
    return df


def calculate_benchmark_ew(df):
    df = df.copy()
    fwd = sum(0.25 * df[f'{a}_Return'].shift(-1) for a in ASSET_NAMES)
    df['Bench_EW_Return']     = fwd
    df['Bench_EW_Cumulative'] = (1 + df['Bench_EW_Return'].fillna(0)).cumprod()
    return df


def calculate_benchmark_sma(df, short_window=10, long_window=12):
    df = df.copy()
    df['SMA_Short']    = df['SPX_Price'].rolling(short_window).mean()
    df['SMA_Long']     = df['SPX_Price'].rolling(long_window).mean()
    df['SMA_Position'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1, 0)
    fwd_spx = df['SPX_Return'].shift(-1)
    df['Bench_SMA_Return']      = df['SMA_Position'] * fwd_spx
    df['Bench_SMA_Cumulative']  = (1 + df['Bench_SMA_Return'].fillna(0)).cumprod()
    return df


def calculate_benchmark_random(df, n_simulations=1000, seed=42):
    df = df.copy()
    rng = np.random.RandomState(seed)

    valid_signals = df.dropna(subset=['Signal'])
    signal_counts = valid_signals['Signal'].value_counts(normalize=True)
    signal_labels = signal_counts.index.tolist()
    signal_probs  = signal_counts.values.tolist()

    n_periods  = len(valid_signals)
    fwd_returns = df['Forward_Return'].values

    sim_sharpes          = []
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

        sharpe = (sim_returns.mean() * 12) / (sim_returns.std() * np.sqrt(12)) if sim_returns.std() > 0 else 0
        sim_sharpes.append(sharpe)
        sim_cumulative_returns.append(np.cumprod(1 + sim_returns))

    sim_sharpes = np.array(sim_sharpes)
    median_idx  = np.argsort(sim_sharpes)[n_simulations // 2]
    median_cum  = sim_cumulative_returns[median_idx]

    valid_indices = valid_signals.index.tolist()
    df['Bench_Random_Return']      = np.nan
    df['Bench_Random_Cumulative']  = np.nan

    median_returns = np.diff(median_cum, prepend=1.0) / np.concatenate([[1.0], median_cum[:-1]])

    for j, idx in enumerate(valid_indices):
        if j < len(median_returns):
            df.loc[idx, 'Bench_Random_Return']     = median_returns[j]
            df.loc[idx, 'Bench_Random_Cumulative'] = median_cum[j]

    df['Bench_Random_Cumulative'] = df['Bench_Random_Cumulative'].ffill().fillna(1.0)
    df.attrs['random_sim_sharpes']         = sim_sharpes
    df.attrs['random_sim_median_sharpe']   = np.median(sim_sharpes)
    df.attrs['random_sim_95th_sharpe']     = np.percentile(sim_sharpes, 95)
    df.attrs['random_sim_5th_sharpe']      = np.percentile(sim_sharpes, 5)

    return df


def calculate_all_benchmarks(df):
    print("  Computing benchmarks...")
    df = calculate_benchmark_6040(df)
    df = calculate_benchmark_ew(df)
    df = calculate_benchmark_sma(df)
    print("    → Random Signal Baseline (1,000 simulations)")
    df = calculate_benchmark_random(df)
    return df


# ──── Performance Metrics ───────────────────────────────────────────────────

def calculate_max_drawdown(cum_series):
    rolling_max = cum_series.expanding().max()
    return ((cum_series - rolling_max) / rolling_max).min()


def compute_information_ratio(strat_rets, bh_rets):
    """IR = mean(active returns) / std(active returns) * sqrt(12)"""
    active = strat_rets - bh_rets
    if active.std() < 1e-10:
        return 0.0
    return (active.mean() / active.std()) * np.sqrt(12)


def compute_indicator_correlations(df, valid):
    """Correlation of each indicator signal with SPX forward return."""
    indicator_cols = ['I1_AR3', 'I2_MACD', 'I3_CreditSpread', 'I4_CommodityMom', 'I5_VolRegime']
    results = {}
    for col in indicator_cols:
        if col in valid.columns:
            mask = valid[col].notna() & valid['Forward_Return'].notna()
            if mask.sum() > 5:
                results[col] = valid.loc[mask, col].corr(valid.loc[mask, 'Forward_Return'])
            else:
                results[col] = np.nan
    return results


def calculate_performance_metrics(df, periods_per_year=12):
    valid     = df.dropna(subset=['Strategy_Return', 'Forward_Return']).copy()
    strat_rets = valid['Strategy_Return']
    bh_rets    = valid['BH_Return']

    def compute_metrics(returns, label):
        n_periods  = len(returns)
        cumulative = (1 + returns).prod()
        n_years    = n_periods / periods_per_year
        ann_return = cumulative ** (1 / n_years) - 1 if n_years > 0 else 0
        ann_vol    = returns.std() * np.sqrt(periods_per_year)
        sharpe     = ann_return / ann_vol if ann_vol > 0 else 0
        cum_series = (1 + returns).cumprod()
        max_dd     = calculate_max_drawdown(cum_series)
        win_rate   = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

        return {
            'Label':              label,
            'Total Periods':      n_periods,
            'Annualized Return':  ann_return,
            'Volatility':         ann_vol,
            'Sharpe Ratio':       sharpe,
            'Max Drawdown':       max_dd,
            'Win Rate':           win_rate,
            'Cumulative Return':  cumulative - 1,
        }

    strat_metrics = compute_metrics(strat_rets, '5-Indicator M1')
    bh_metrics    = compute_metrics(bh_rets,    'Buy & Hold SPX')

    # Information Ratio
    strat_metrics['Information Ratio'] = compute_information_ratio(strat_rets, bh_rets)

    # Correlation of composite signal with SPX
    mask = valid['Composite_Score'].notna() & valid['Forward_Return'].notna()
    strat_metrics['Corr (Composite vs SPX)'] = (
        valid.loc[mask, 'Composite_Score'].corr(valid.loc[mask, 'Forward_Return'])
        if mask.sum() > 5 else np.nan
    )

    bench_metrics = {}

    b6040 = valid['Bench_6040_Return'].dropna()
    if len(b6040) > 0:
        bench_metrics['60/40 Portfolio'] = compute_metrics(b6040, '60/40 Portfolio')
        bench_metrics['60/40 Portfolio']['Information Ratio'] = compute_information_ratio(b6040, bh_rets.loc[b6040.index])

    bew = valid['Bench_EW_Return'].dropna()
    if len(bew) > 0:
        bench_metrics['Equal-Weight'] = compute_metrics(bew, 'Equal-Weight (25%)')
        bench_metrics['Equal-Weight']['Information Ratio'] = compute_information_ratio(bew, bh_rets.loc[bew.index])

    bsma = valid['Bench_SMA_Return'].dropna()
    if len(bsma) > 0:
        bench_metrics['SMA Crossover'] = compute_metrics(bsma, 'SMA Crossover')
        bench_metrics['SMA Crossover']['Information Ratio'] = compute_information_ratio(bsma, bh_rets.loc[bsma.index])

    brand = valid['Bench_Random_Return'].dropna()
    if len(brand) > 0:
        bench_metrics['Random Baseline'] = compute_metrics(brand, 'Random Baseline')
        bench_metrics['Random Baseline']['Information Ratio'] = compute_information_ratio(brand, bh_rets.loc[brand.index])

    # Time-in-market
    strat_metrics['Time in Market']    = (valid['Position'] != 0).sum() / len(valid)
    strat_metrics['Periods Long/Flat'] = f"{(valid['Position']==1).sum()} / {(valid['Position']==0).sum()}"

    # Per-indicator correlations
    ind_corrs = compute_indicator_correlations(df, valid)
    for col, corr in ind_corrs.items():
        strat_metrics[f'Corr {col} vs SPX'] = corr

    return strat_metrics, bh_metrics, bench_metrics


def print_performance(strat_metrics, bh_metrics, bench_metrics=None):
    all_benchmarks = {'5-Indicator M1': strat_metrics, 'Buy & Hold SPX': bh_metrics}
    if bench_metrics:
        all_benchmarks.update(bench_metrics)

    labels    = list(all_benchmarks.keys())
    col_width = 20

    print(f"\n{'Metric':<35}", end='')
    for label in labels:
        print(f" {label[:col_width-2]:<{col_width}}", end='')
    print()
    print("-" * (35 + col_width * len(labels)))

    numeric_keys = [
        ('Annualized Return',   '{:.2%}'),
        ('Volatility',          '{:.2%}'),
        ('Sharpe Ratio',        '{:.3f}'),
        ('Max Drawdown',        '{:.2%}'),
        ('Win Rate',            '{:.2%}'),
        ('Cumulative Return',   '{:.2%}'),
        ('Information Ratio',   '{:.3f}'),
    ]

    for key, fmt in numeric_keys:
        print(f"  {key:<33}", end='')
        for label in labels:
            val = all_benchmarks[label].get(key, None)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                s = 'N/A'
            else:
                try:
                    s = fmt.format(val)
                except Exception:
                    s = str(val)
            print(f" {s:<{col_width}}", end='')
        print()

    print()
    print("Strategy-Specific Metrics:")
    print("-" * 60)
    extra_keys = ['Corr (Composite vs SPX)', 'Time in Market', 'Periods Long/Flat']
    for key in extra_keys:
        val = strat_metrics.get(key, None)
        if val is not None:
            if isinstance(val, float) and not np.isnan(val):
                s = f"{val:.4f}" if abs(val) < 10 else f"{val:.2f}"
            else:
                s = str(val)
            print(f"  {key:<38} {s}")

    # Per-indicator correlations
    print("\n  Per-Indicator Correlation with SPX Forward Return:")
    ind_cols = ['I1_AR3', 'I2_MACD', 'I3_CreditSpread', 'I4_CommodityMom', 'I5_VolRegime']
    for col in ind_cols:
        key = f'Corr {col} vs SPX'
        val = strat_metrics.get(key, None)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            print(f"    {col:<28} {val:+.4f}")


# ──── Classification Metrics ────────────────────────────────────────────────

def calculate_signal_accuracy(df):
    valid = df.dropna(subset=['Signal', 'Forward_Return']).copy()

    buy_signals  = valid[valid['Signal'] == 'BUY']
    n_buy        = len(buy_signals)
    n_buy_correct = (buy_signals['Forward_Return'] > 0).sum() if n_buy > 0 else 0
    precision    = n_buy_correct / n_buy if n_buy > 0 else 0

    sell_signals  = valid[valid['Signal'] == 'SELL']
    n_sell        = len(sell_signals)
    n_sell_correct = (sell_signals['Forward_Return'] < 0).sum() if n_sell > 0 else 0

    positive_periods = valid[valid['Forward_Return'] > 0]
    n_positive = len(positive_periods)
    n_caught   = len(positive_periods[positive_periods['Signal'] == 'BUY'])
    recall     = n_caught / n_positive if n_positive > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    results = {
        'BUY Signals':             n_buy,
        'SELL Signals':            n_sell,
        'Precision (BUY)':         precision,
        'Recall (BUY)':            recall,
        'F1 Score (BUY)':          f1,
        'SELL Precision':          n_sell_correct / n_sell if n_sell > 0 else 0,
        'Overall Accuracy':        (n_buy_correct + n_sell_correct) / (n_buy + n_sell) if (n_buy + n_sell) > 0 else 0,
    }

    print("\n" + "=" * 70)
    print("SIGNAL CLASSIFICATION METRICS")
    print("=" * 70)
    fmt_map = {
        'BUY Signals': '{:d}',
        'SELL Signals': '{:d}',
        'Precision (BUY)': '{:.3f}',
        'Recall (BUY)': '{:.3f}',
        'F1 Score (BUY)': '{:.3f}',
        'SELL Precision': '{:.3f}',
        'Overall Accuracy': '{:.3f}',
    }
    for k, v in results.items():
        fmt = fmt_map.get(k, '{}')
        try:
            s = fmt.format(v)
        except Exception:
            s = str(v)
        print(f"  {k:<35} {s}")
    print("=" * 70)
    return results


# ──── Meta-labels ────────────────────────────────────────────────────────────

def generate_meta_labels(df):
    df = df.copy()
    df['Meta_Label'] = np.nan
    df.loc[(df['Signal'] == 'BUY')  & (df['Forward_Return'] > 0), 'Meta_Label'] = 1
    df.loc[(df['Signal'] == 'BUY')  & (df['Forward_Return'] <= 0), 'Meta_Label'] = 0
    df.loc[(df['Signal'] == 'SELL') & (df['Forward_Return'] < 0), 'Meta_Label'] = 1
    df.loc[(df['Signal'] == 'SELL') & (df['Forward_Return'] >= 0), 'Meta_Label'] = 0

    meta_valid = df.dropna(subset=['Meta_Label'])
    n_total = len(meta_valid)
    if n_total > 0:
        n_correct   = (meta_valid['Meta_Label'] == 1).sum()
        n_incorrect = (meta_valid['Meta_Label'] == 0).sum()
        print(f"\nMeta-Labels Generated: {n_correct}/{n_total} correct ({n_correct/n_total*100:.1f}%)")
    return df


def calculate_rolling_metrics(df, window=12):
    df = df.copy()
    rolling_mean = df['Strategy_Return'].rolling(window).mean() * 12
    rolling_std  = df['Strategy_Return'].rolling(window).std() * np.sqrt(12)
    df['Rolling_Sharpe'] = rolling_mean / rolling_std
    df['Signal_Hit']     = np.nan
    df.loc[(df['Signal'] == 'BUY')  & (df['Forward_Return'] > 0), 'Signal_Hit'] = 1
    df.loc[(df['Signal'] == 'BUY')  & (df['Forward_Return'] <= 0), 'Signal_Hit'] = 0
    df.loc[(df['Signal'] == 'SELL') & (df['Forward_Return'] < 0), 'Signal_Hit'] = 1
    df.loc[(df['Signal'] == 'SELL') & (df['Forward_Return'] >= 0), 'Signal_Hit'] = 0
    df['Rolling_HitRate'] = df['Signal_Hit'].rolling(window, min_periods=1).mean()
    return df


# ──── Train/Test Split ───────────────────────────────────────────────────────

def train_test_split(df, train_ratio=0.60):
    valid_df  = df.dropna(subset=['Signal']).copy()
    n         = len(valid_df)
    split_idx = int(n * train_ratio)
    split_date = valid_df.iloc[split_idx]['Date']

    df = df.copy()
    df['Set'] = 'pre-signal'
    df.loc[df['Date'].isin(valid_df.iloc[:split_idx]['Date']), 'Set'] = 'train'
    df.loc[df['Date'].isin(valid_df.iloc[split_idx:]['Date']), 'Set'] = 'test'

    train_df = df[df['Set'] == 'train'].copy()
    test_df  = df[df['Set'] == 'test'].copy()

    print(f"\n  Train/Test Split (sequential):")
    print(f"    Train: {len(train_df)} periods  |  {train_df['Date'].min().strftime('%Y-%m')} to {train_df['Date'].max().strftime('%Y-%m')}")
    print(f"    Test:  {len(test_df)} periods   |  {test_df['Date'].min().strftime('%Y-%m')} to {test_df['Date'].max().strftime('%Y-%m')}")
    print(f"    Split date: {split_date.strftime('%Y-%m-%d')}")

    return df, train_df, test_df


# ──── Random Significance ────────────────────────────────────────────────────

def print_random_significance(df, strategy_sharpe):
    if 'random_sim_sharpes' not in df.attrs:
        return
    sim_sharpes = df.attrs['random_sim_sharpes']
    percentile  = (sim_sharpes < strategy_sharpe).sum() / len(sim_sharpes) * 100
    p_value     = 1 - percentile / 100

    print("\n" + "=" * 70)
    print("  STATISTICAL SIGNIFICANCE: M1 vs RANDOM SIGNALS")
    print("=" * 70)
    print(f"  M1 Strategy Sharpe:      {strategy_sharpe:.3f}")
    print(f"  Random Median Sharpe:    {np.median(sim_sharpes):.3f}")
    print(f"  M1 Percentile Rank:      {percentile:.1f}th percentile")
    print(f"  p-value (one-tailed):    {p_value:.3f}")
    if p_value < 0.05:
        print(f"  Result:                  SIGNIFICANT (p < 0.05)")
    elif p_value < 0.10:
        print(f"  Result:                  MARGINALLY SIGNIFICANT (p < 0.10)")
    else:
        print(f"  Result:                  NOT SIGNIFICANT (p = {p_value:.3f})")
    print("=" * 70)


# =============================================================================
# PART 3: MAIN BACKTEST RUNNER
# =============================================================================

def run_backtest(df, periods_per_year=12, train_ratio=0.60):
    print("\n" + "#" * 70)
    print("#  RUNNING BACKTEST — 5-INDICATOR M1 MODEL")
    print("#" * 70)

    df = map_signals_to_positions(df)
    df = calculate_strategy_returns(df)
    df = calculate_all_benchmarks(df)
    df = generate_meta_labels(df)
    df = calculate_rolling_metrics(df)
    df, train_df, test_df = train_test_split(df, train_ratio)

    print("\n" + "=" * 70)
    print("  IN-SAMPLE (TRAIN) PERFORMANCE")
    print("=" * 70)
    train_strat, train_bh, train_bench = calculate_performance_metrics(train_df, periods_per_year)
    print_performance(train_strat, train_bh, train_bench)
    print("\n--- In-Sample Classification Metrics ---")
    calculate_signal_accuracy(train_df)

    print("\n" + "=" * 70)
    print("  OUT-OF-SAMPLE (TEST) PERFORMANCE  ← Primary evaluation")
    print("=" * 70)
    test_strat, test_bh, test_bench = calculate_performance_metrics(test_df, periods_per_year)
    print_performance(test_strat, test_bh, test_bench)
    print("\n--- Out-of-Sample Classification Metrics ---")
    calculate_signal_accuracy(test_df)

    print_random_significance(df, test_strat['Sharpe Ratio'])

    for label, subset in [('Train', train_df), ('Test', test_df)]:
        meta_valid = subset.dropna(subset=['Meta_Label'])
        n = len(meta_valid)
        if n > 0:
            n1 = (meta_valid['Meta_Label'] == 1).sum()
            print(f"\n  Meta-Label Distribution ({label}): {n1}/{n} correct ({n1/n*100:.1f}%)")

    print("\n" + "#" * 70)
    print("#  BACKTEST COMPLETE")
    print("#" * 70)

    return df, train_df, test_df


# =============================================================================
# PART 4: RUN EVERYTHING
# =============================================================================

if __name__ == "__main__":

    DATA_DIR = '/mnt/user-data/uploads'

    BCOM_PATH     = f'{DATA_DIR}/BCOM_INDEX__3_.xlsx'
    SPX_PATH      = f'{DATA_DIR}/SNP500__2_.xlsx'
    TREASURY_PATH = f'{DATA_DIR}/US_10Y_treasury_bonds__2_.xlsx'
    IG_PATH       = f'{DATA_DIR}/US_IG_corporate_bonds__1_.xlsx'

    # Step 1: Build signals
    results = run_signal_model(
        bcom_path     = BCOM_PATH,
        spx_path      = SPX_PATH,
        treasury_path = TREASURY_PATH,
        ig_path       = IG_PATH,
    )

    # Step 2: Latest signal
    latest = get_latest_signal(results)

    # Step 3: Full backtest
    full_df, train_df, test_df = run_backtest(results)

    # Step 4: Save results
    output_cols = [
        'Date', 'spx_Price', 'spx_Return',
        'I1_AR3', 'I1_AR3_Forecast',
        'I2_MACD', 'I2_MACD_Line', 'I2_MACD_Vel',
        'I3_CreditSpread', 'I3_Spread_Index', 'I3_Spread_ROC_Inv',
        'I4_CommodityMom', 'I4_Commodity_ZScore',
        'I5_VolRegime', 'I5_Vol_ZScore',
        'Vote_Sum', 'Composite_Score', 'Signal', 'Position',
        'Forward_Return', 'Strategy_Return', 'BH_Return',
        'Strategy_Cumulative', 'BH_Cumulative',
        'Bench_6040_Return', 'Bench_6040_Cumulative',
        'Bench_EW_Return', 'Bench_EW_Cumulative',
        'Bench_SMA_Return', 'Bench_SMA_Cumulative',
        'Bench_Random_Return', 'Bench_Random_Cumulative',
        'Meta_Label', 'Rolling_Sharpe', 'Rolling_HitRate', 'Set'
    ]
    existing = [c for c in output_cols if c in full_df.columns]
    out_path = '/mnt/user-data/outputs/backtest_results_5indicator.csv'
    full_df[existing].to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
