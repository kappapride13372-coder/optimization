#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
four_bar_patterns_async.py
Fetch OHLC data from Binance asynchronously for multiple symbols, categorize 4-bar rolling windows 
using adjacent comparisons, and print the top 10 most frequent patterns in a human-readable format.
"""

import pandas as pd
import asyncio
import aiohttp
from datetime import datetime, timedelta
from collections import Counter
import time

BASE_URL = "https://api.binance.com/api/v3/klines"

# -----------------------------
# Fetch OHLC chunk
# -----------------------------
async def fetch_klines(session, symbol, interval, start_str, limit=1000):
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(start_str.timestamp() * 1000),
        "limit": limit
    }
    async with session.get(BASE_URL, params=params) as resp:
        return await resp.json()

# -----------------------------
# Fetch full historical OHLC
# -----------------------------
async def get_historical_ohlc(symbol, interval='1h', years=4):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=365*years)
    all_data = []

    async with aiohttp.ClientSession() as session:
        while start_time < end_time:
            chunk = await fetch_klines(session, symbol, interval, start_time)
            if not chunk:
                break
            all_data.extend(chunk)
            last_time = chunk[-1][0]
            start_time = datetime.utcfromtimestamp(last_time / 1000) + timedelta(hours=1)
            await asyncio.sleep(0.1)  # slight delay to avoid rate limits

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
    df['4bar_pattern'] = pattern_defs
    return df

# -----------------------------
# Convert bit pattern to readable format
# -----------------------------
def format_pattern_readable(bits):
    if bits is None:
        return None
    attrs = ['open','high','low','close']
    readable = {}
    for i, attr in enumerate(attrs):
        readable[attr] = ['above' if b==1 else 'below_or_equal' for b in bits[i*3:(i+1)*3]]
    return readable

# -----------------------------
# Print top N patterns
# -----------------------------
def print_top_patterns_readable(df, top_n=10):
    valid_patterns = df['4bar_class'].dropna().astype(int)
    valid_defs = df['4bar_pattern'].dropna()
    
    counter = Counter(valid_patterns)
    top_patterns = counter.most_common(top_n)
    
    print(f"\nTop {top_n} 4-bar patterns (readable):")
    for class_id, count in top_patterns:
        idx = valid_patterns[valid_patterns == class_id].index[0]
        bits = valid_defs.loc[idx]
        readable = format_pattern_readable(bits)
        print(f"Class ID: {class_id}, Occurrences: {count}, Definition: {readable}")

# -----------------------------
# Async main function
# -----------------------------
async def main(symbols, interval='1h', years=4):
    tasks = []
    for symbol in symbols:
        print(f"Fetching {symbol}...")
        tasks.append(get_historical_ohlc(symbol, interval, years))
    
    results = await asyncio.gather(*tasks)
    ohlc_data = dict(zip(symbols, results))
    return ohlc_data

# -----------------------------
# Run the script
# -----------------------------
if __name__ == "__main__":
    symbols = ['BTCUSDT','ETHUSDT','BNBUSDT']
    ohlc_data = asyncio.run(main(symbols))

    for symbol, df in ohlc_data.items():
        print(f"\n{symbol} data fetched: {df.shape[0]} rows")
        df_classed = get_4bar_adjacent_class(df)
        print_top_patterns_readable(df_classed, top_n=10)
