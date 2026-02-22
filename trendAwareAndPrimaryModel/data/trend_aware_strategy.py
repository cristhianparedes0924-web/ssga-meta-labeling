import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from sklearn.ensemble import GradientBoostingRegressor

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
REPORTS_DIR = DATA_DIR / 'reports'


def _ensure_within_project(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f'Path escapes project folder: {resolved}') from exc
    return resolved

# Update these filenames to match your local files
FILES = {
    'spx': RAW_DIR / 'spx.xlsx',
    'rates': RAW_DIR / 'treasury_10y.xlsx',
    'commodities': RAW_DIR / 'bcom.xlsx',
    'credit': RAW_DIR / 'corp_bonds.xlsx',
}

# The 5 Core Features (No Mania Override)
ML_FEATURES = ['rates_phase', 'credit_phase', 'spx_phase', 'commodities_phase', 'spx_roc_3m']

# Risk Constraints
MIN_WEIGHT = 0.00
MAX_WEIGHT = 1.00

# ==========================================
# 2. DATA LOADING & PROCESSING
# ==========================================
def load_data(file_map):
    print("Loading data...")
    prices = pd.DataFrame()

    for name, filepath in file_map.items():
        try:
            safe_path = _ensure_within_project(Path(filepath))
            raw = pd.read_excel(safe_path, header=None, dtype=object)

            header_row = -1
            for i, row in raw.iterrows():
                values = [str(v).strip().lower() for v in row.tolist() if pd.notna(v)]
                if 'date' in values and ('px_last' in values or 'close' in values):
                    header_row = int(i)
                    break

            if header_row == -1:
                print(f"Skipping {name}: Header not found.")
                continue

            header = raw.iloc[header_row].tolist()
            df = raw.iloc[header_row + 1 :].copy()
            df.columns = header
            df = df.loc[:, [col for col in df.columns if pd.notna(col)]]
            df.columns = [str(c).strip() for c in df.columns]

            date_col = next((c for c in df.columns if c.lower() == 'date'), None)
            close_col = next((c for c in df.columns if c.upper() == 'PX_LAST' or 'close' in c.lower()), None)

            if date_col and close_col:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df[close_col] = pd.to_numeric(df[close_col], errors='coerce')

                series = (
                    df[[date_col, close_col]]
                    .dropna(subset=[date_col])
                    .drop_duplicates(subset=[date_col], keep='last')
                    .set_index(date_col)
                    .sort_index()
                    .rename(columns={close_col: name})
                )

                if prices.empty:
                    prices = series
                else:
                    prices = prices.join(series, how='outer')
        except Exception as e:
            print(f"Error reading {name}: {e}")

    # CRITICAL: Sort Index to ensure correct time sequence
    prices.sort_index(inplace=True)
    return prices.ffill().dropna()


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
    start_window = 24  # 2 years warm-up

    for i in range(len(idx)):
        X_curr = X.iloc[i : i + 1]

        # Base Allocation (Equal Weight)
        allocation = pd.Series(0.25, index=prices.columns)

        # ML Logic (Only if enough history)
        if i >= start_window:
            X_train = X[ML_FEATURES].iloc[:i]
            y_train = targets.iloc[:i]

            preds = {}
            for asset in prices.columns:
                # Lightweight Gradient Boosting (Trend Aware)
                model = GradientBoostingRegressor(
                    n_estimators=30,
                    max_depth=2,
                    learning_rate=0.1,
                    random_state=42,
                )
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
    cagr = strat_cum.iloc[-1] ** (12 / len(strat_cum)) - 1
    sharpe = (strat_ret.mean() * 12) / (strat_ret.std() * np.sqrt(12))
    dd = (strat_cum / strat_cum.cummax() - 1).min()

    print("\n" + "=" * 50)
    print("ORIGINAL 5 STRATEGY PERFORMANCE (1997-2025)")
    print("=" * 50)
    print(f"{'Metric':<20} | {'Strategy':<10} | {'S&P 500':<10}")
    print("-" * 50)
    print(f"{'Total Return':<20} | {total_ret:.0%}      | {spx_cum.iloc[-1] - 1:.0%}")
    print(f"{'CAGR':<20} | {cagr:.2%}     | {(spx_cum.iloc[-1]) ** (12 / len(spx_cum)) - 1:.2%}")
    print(f"{'Sharpe Ratio':<20} | {sharpe:.2f}       | {(spx_ret.mean() * 12) / (spx_ret.std() * np.sqrt(12)):.2f}")
    print(f"{'Max Drawdown':<20} | {dd:.2%}    | {(spx_cum / spx_cum.cummax() - 1).min():.2%}")
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
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    chart_path = _ensure_within_project(REPORTS_DIR / 'strategy_performance.png')
    plt.savefig(chart_path)
    print(f"\n[+] Chart saved as '{chart_path}'")

    # Save Weights
    weights_path = _ensure_within_project(REPORTS_DIR / 'monthly_allocations.csv')
    weights.to_csv(weights_path)
    print(f"[+] Weights saved as '{weights_path}'")


if __name__ == "__main__":
    run_strategy()
