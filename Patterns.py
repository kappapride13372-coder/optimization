#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
four_bar_patterns_incremental.py
Fetch OHLC from Binance, classify 4-bar windows,
compute average forward returns (6,12,18,24 bars), profit factor,
count occurrences, filter patterns >=300 times, and write results incrementally to CSV.
Safe for low-RAM droplets.
"""

import pandas as pd
import asyncio
import aiohttp
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
import math
import os

BASE_URL = "https://api.binance.com/api/v3/klines"

# -----------------------------
# Fetch OHLC chunk safely
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
# Get historical OHLC for one symbol
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
    # Keep only essential columns
    df = df[['open','high','low','close']]
    df = df.astype(float)
    return df

# -----------------------------
# Classify 4-bar adjacent patterns
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
# Forward returns
# -----------------------------
def add_forward_returns(df, horizons=[6,12,18,24]):
    for h in horizons:
        df[f'return_{h}b'] = (df['close'].shift(-h) - df['close']) / df['close'] * 100
    return df

# -----------------------------
# Bits to human-readable
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
# Process one symbol and return DataFrame
# -----------------------------
async def process_symbol(symbol, interval='1h', years=4, horizons=[6,12,18,24]):
    df = await get_historical_ohlc(symbol, interval, years)
    df_classed = get_4bar_adjacent_class(df)
    df_classed = add_forward_returns(df_classed, horizons)
    valid = df_classed.dropna(subset=['4bar_class'] + [f'return_{h}b' for h in horizons]).copy()
    valid['4bar_class'] = valid['4bar_class'].astype(int)
    return valid

# -----------------------------
# Main incremental analysis
# -----------------------------
async def main(symbols, interval='1h', years=4, horizons=[6,12,18,24], min_occurrences=300):
    output_file = "4bar_class_complete.csv"
    # Remove old CSV
    if os.path.exists(output_file):
        os.remove(output_file)

    for symbol in symbols:
        df_symbol = await process_symbol(symbol, interval, years, horizons)
        # Count occurrences
        occurrences = df_symbol['4bar_class'].value_counts()
        valid_classes = occurrences[occurrences >= min_occurrences].index
        df_symbol = df_symbol[df_symbol['4bar_class'].isin(valid_classes)]

        grouped = df_symbol.groupby('4bar_class')
        result_rows = []
        for cls in tqdm(grouped.groups.keys(), desc=f"Analyzing {symbol}", ncols=80):
            group = grouped.get_group(cls)
            row = {'4bar_class': cls, 'occurrences': len(group)}
            for h in horizons:
                returns = group[f'return_{h}b']
                row[f'return_{h}b'] = returns.mean()
                pos_sum = returns[returns>0].sum()
                neg_sum = returns[returns<0].sum()
                row[f'pf_{h}b'] = pos_sum / abs(neg_sum) if abs(neg_sum)>0 else float('inf')
            row['pattern'] = bits_to_readable(group['pattern_bits'].iloc[0])
            result_rows.append(row)

        result_df = pd.DataFrame(result_rows)
        # Append to CSV incrementally
        header = not os.path.exists(output_file)
        result_df.to_csv(output_file, mode='a', index=False, header=header)

    print(f"\nCSV saved as {output_file}")
    input("\nScan completed. Press Enter to exit...")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    symbols = ['BTCUSDT','ETHUSDT','BNBUSDT']
    asyncio.run(main(symbols))
