#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
four_bar_patterns_returns_pf.py
Fetch OHLC from Binance sequentially with progress bars, classify 4-bar windows,
compute average forward returns (6,12,18,24 bars) and profit factor, 
and save a single CSV for all symbols including human-readable patterns.
"""

import pandas as pd
import asyncio
import aiohttp
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
import math

BASE_URL = "https://api.binance.com/api/v3/klines"

# -----------------------------
# Fetch a chunk of klines safely
# -----------------------------
async def fetch_klines(session, symbol, interval, start_str, limit=1000, retries=3):
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(start_str.timestamp() * 1000),
        "limit": limit
    }
    for attempt in range(retries):
        try:
            async with session.get(BASE_URL, params=params, timeout=10) as resp:
                return await resp.json()
        except Exception as e:
            print(f"Error fetching {symbol} at {start_str} (attempt {attempt+1}): {e}")
            await asyncio.sleep(2)
    print(f"Failed to fetch {symbol} at {start_str} after {retries} attempts.")
    return []

# -----------------------------
# Fetch full historical OHLC with progress bar
# -----------------------------
async def get_historical_ohlc(symbol, interval='1h', years=4):
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=365*years)
    all_data = []

    async with aiohttp.ClientSession() as session:
        total_hours = years * 365 * 24
        iterations = math.ceil(total_hours / 1000)
        pbar = tqdm(total=iterations, desc=f"{symbol} fetching", ncols=80)
        
        while start_time < end_time:
            chunk = await fetch_klines(session, symbol, interval, start_time)
            if not chunk:
                break
            all_data.extend(chunk)
            last_time = chunk[-1][0]
            start_time = datetime.fromtimestamp(last_time / 1000, tz=timezone.utc) + timedelta(hours=1)
            await asyncio.sleep(0.05)
            pbar.update(1)
        pbar.close()

    df = pd.DataFrame(all_data, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','quote_asset_volume','number_of_trades',
        'taker_buy_base','taker_buy_quote','ignore'
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    numeric_cols = ['open','high','low','close','volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df

# -----------------------------
# Categorize 4-bar adjacent patterns
# -----------------------------
def get_4bar_adjacent_class(df):
    class_ids = []
    pattern_defs = []

    for i in range(3, len(df)):
        window = df.iloc[i-3:i+1]
        bits = []
        for col in ['open','high','low','close']:
            bits.append(int(window[col].iloc[1] > window[col].iloc[0]))
            bits.append(int(window[col].iloc[2] > window[col].iloc[1]))
            bits.append(int(window[col].iloc[3] > window[col].iloc[2]))
        
        class_id = 0
        for bit in bits:
            class_id = (class_id << 1) | bit
        
        class_ids.append(class_id)
        pattern_defs.append(bits)

    class_ids = [None, None, None] + class_ids
    pattern_defs = [None, None, None] + pattern_defs

    df['4bar_class'] = class_ids
    df['pattern_bits'] = pattern_defs
    return df

# -----------------------------
# Convert bit pattern to human-readable format
# -----------------------------
def bits_to_readable(bits):
    if bits is None:
        return None
    attrs = ['open','high','low','close']
    readable = {}
    for i, attr in enumerate(attrs):
        readable[attr] = ['above' if b==1 else 'below_or_equal' for b in bits[i*3:(i+1)*3]]
    return readable

# -----------------------------
# Compute forward returns
# -----------------------------
def add_forward_returns(df, horizons=[6,12,18,24]):
    for h in horizons:
        df[f'return_{h}b'] = (df['close'].shift(-h) - df['close']) / df['close'] * 100
    return df

# -----------------------------
# Process a symbol: fetch, classify, returns
# -----------------------------
async def process_symbol(symbol, interval='1h', years=4, horizons=[6,12,18,24]):
    df = await get_historical_ohlc(symbol, interval, years)
    df_classed = get_4bar_adjacent_class(df)
    df_classed = add_forward_returns(df_classed, horizons)
    
    # Drop rows with missing 4bar_class or returns
    valid = df_classed.dropna(subset=['4bar_class'] + [f'return_{h}b' for h in horizons])
    valid['4bar_class'] = valid['4bar_class'].astype(int)
    return valid

# -----------------------------
# Aggregate across symbols
# -----------------------------
async def main(symbols, interval='1h', years=4, horizons=[6,12,18,24]):
    aggregated = pd.DataFrame()

    for symbol in symbols:
        df_symbol = await process_symbol(symbol, interval, years, horizons)
        aggregated = pd.concat([aggregated, df_symbol], ignore_index=True)

    # Group by 4bar_class
    agg_funcs = {}
    for h in horizons:
        agg_funcs[f'return_{h}b'] = 'mean'
        agg_funcs[f'pf_{h}b'] = lambda x: x[x>0].sum() / abs(x[x<0].sum()) if abs(x[x<0].sum())>0 else float('inf')

    # Compute profit factor for each horizon
    pf_df = pd.DataFrame()
    for h in horizons:
        pf = aggregated.groupby('4bar_class')[f'return_{h}b'].apply(lambda x: x[x>0].sum() / abs(x[x<0].sum()) if abs(x[x<0].sum())>0 else float('inf'))
        pf_df[f'pf_{h}b'] = pf

    # Compute average returns
    returns_df = aggregated.groupby('4bar_class')[[f'return_{h}b' for h in horizons]].mean()

    # Merge returns and profit factor
    result = returns_df.merge(pf_df, left_index=True, right_index=True).reset_index()

    # Add human-readable pattern (first occurrence)
    pattern_map = aggregated.groupby('4bar_class')['pattern_bits'].first().apply(bits_to_readable)
    result['pattern'] = result['4bar_class'].map(pattern_map)

    # Save CSV
    result.to_csv("4bar_class_returns_pf.csv", index=False)
    print("\nCSV saved as 4bar_class_returns_pf.csv")
    print(result.head(10))

# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":
    symbols = ['BTCUSDT','ETHUSDT','BNBUSDT']
    asyncio.run(main(symbols))
