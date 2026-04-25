import pandas as pd
import numpy as np

def create_indicators(df):
    """
    Calculates the 5 SSGA signals based on the raw price data.
    """
    # Create a copy to avoid warnings
    df = df.copy()

    # ---------------------------------------------------------
    # 1. MOMENTUM (12-month return of S&P 500)
    # Logic: Stocks trending up tend to continue.
    # ---------------------------------------------------------
    df['Z1_Mom'] = df['SPX_Price'].pct_change(12)

    # ---------------------------------------------------------
    # 2. VALUE (Real Yield: 10Y Yield - CPI Proxy)
    # Using BCOM year-over-year change as a rough inflation proxy
    # ---------------------------------------------------------
    df['Inflation_Proxy'] = df['BCOM_Price'].pct_change(12)
    df['Z2_Value'] = df['Treasury10Y_Price'] - df['Inflation_Proxy']

    # ---------------------------------------------------------
    # 3. CARRY (Corporate Bond Return - Risk Free Return)
    # Simple proxy: IG Corp Return - Treasury Return (12 month rolling)
    # ---------------------------------------------------------
    df['Corp_Ret_12M'] = df['IG_Corp_Price'].pct_change(12)
    df['Treasury_Ret_12M'] = df['Treasury10Y_Price'].pct_change(12)
    df['Z3_Carry'] = df['Corp_Ret_12M'] - df['Treasury_Ret_12M']

    # ---------------------------------------------------------
    # 4. VOLATILITY (Negative 12-month std dev of S&P returns)
    # Logic: High vol is bad for risk assets.
    # ---------------------------------------------------------
    df['Z5_Vol'] = -1 * df['SPX_Price'].pct_change(1).rolling(12).std()

    # ---------------------------------------------------------
    # 5. TREND (Price vs 10-month Moving Average)
    # ---------------------------------------------------------
    df['MA_10'] = df['SPX_Price'].rolling(10).mean()
    df['Z4_Trend'] = (df['SPX_Price'] - df['MA_10']) / df['MA_10']

    # Drop the first 12 rows (NaNs due to rolling windows)
    df = df.dropna().reset_index(drop=True)

    return df