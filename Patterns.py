"""
Memory-efficient Keltner Channel optimizer with aggregated CSV and multiple equity graphs

- Shows progress while downloading OHLCV bars per symbol
- Processes one symbol at a time
- Aggregates metrics across symbols (numeric only, avoids column duplication)
- Generates CSV including profit factor
- Plots top 5 equity curves by Sortino ratio and top 5 by total return
- Prevents crashes on small droplets by clearing memory and forcing garbage collection
- Ensures rolling windows are integers
- Console waits for Enter to exit
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ccxt
import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm, trange
import os
import gc

# --------------------------- CONFIG ---------------------------
SYMBOLS = [ "HMSTRUSDT","BBUSDT","ACTUSDT","HOOKUSDT","SXTUSDT","FLOWUSDT","MUBARAKUSDT","DEXEUSDT",
    "1000CATUSDT","THETAUSDT","COOKIEUSDT","AVAXUSDT","LQTYUSDT","EPICUSDT","ACXUSDT","CTSIUSDT",
    "GMTUSDT","QNTUSDT","ARKUSDT","ONGUSDT","WIFUSDT","CYBERUSDT","PORTALUSDT","FIDAUSDT",
    "PIXELUSDT","NEXOUSDT","FORMUSDT","CUSDT","BANANAUSDT","KNCUSDT","LRCUSDT","JASMYUSDT",
    "XAIUSDT","EGLDUSDT","TOWNSUSDT","ILVUSDT","DODOUSDT","QIUSDT","HOLOUSDT","EDUUSDT",
    "VELODROMEUSDT","INITUSDT","MANTAUSDT","BICOUSDT","OPENUSDT","BEAMXUSDT","C98USDT","RDNTUSDT",
    "OXTUSDT","ACEUSDT","PHAUSDT","SKLUSDT","AIXBTUSDT","HYPERUSDT","KAIAUSDT","DOTUSDT","FTTUSDT",
    "ZECUSDT","BONKUSDT","NEARUSDT","PYTHUSDT","PHBUSDT","TNSRUSDT","SFPUSDT","AXLUSDT","AEVOUSDT",
    ]
TIMEFRAME = '4h'
EMA_LENGTHS = [30, 60, 90, 120]
ATR_LENGTHS = [30, 60, 90, 120]
MULTIPLIERS = [x * 0.5 for x in range(2, 7)]
INITIAL_CAPITAL = 10000.0
COMMISSION = 0.001
SLIPPAGE_PCT = 0.0005
STOPLOSS_PCT = 0.05
MIN_BARS = 200
TOP_N_PLOT = 5
OUTPUT_DIR = 'keltner_optimizer_out'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------- HELPERS ---------------------------

def fetch_ohlcv(symbol, timeframe, start_date='2022-01-01'):
    exchange = ccxt.binance({'enableRateLimit': True})
    since_ms = int(pd.to_datetime(start_date).timestamp() * 1000)
    all_bars = []
    limit = 1000
    print(f'Fetching data for {symbol}...')
    pbar = tqdm(desc=f'{symbol} OHLCV', unit='bars')
    while True:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        if not bars:
            break
        all_bars += bars
        since_ms = bars[-1][0] + 1
        pbar.update(len(bars))
        if len(bars) < limit:
            break
    pbar.close()
    df = pd.DataFrame(all_bars, columns=['timestamp','open','high','low','close','volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    df = df[['open','high','low','close','volume']]
    df = df[~df.index.duplicated(keep='first')]
    return df

def ema(series, length):
    length = int(length)
    return series.ewm(span=length, adjust=False).mean()

def atr(df, length):
    length = int(length)
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean()

def compute_keltner(df, ema_len, atr_len, mult):
    middle = ema(df['close'], ema_len)
    a = atr(df, atr_len)
    upper = middle + mult * a
    lower = middle - mult * a
    df_kc = df.copy()
    df_kc['kc_middle'] = middle
    df_kc['kc_upper'] = upper
    df_kc['kc_lower'] = lower
    return df_kc

def generate_signals(df):
    close = df['close']
    upper = df['kc_upper']
    middle = df['kc_middle']
    long_entry = (close.shift(1) <= upper.shift(1)) & (close > upper)
    long_exit = (close.shift(1) >= middle.shift(1)) & (close < middle)
    signals = pd.DataFrame(index=df.index)
    signals['entry'] = long_entry.astype(int)
    signals['exit'] = long_exit.astype(int)
    return signals

def backtest(df, signals, return_equity_curve=False):
    prices = df['close']
    position = 0
    cash = INITIAL_CAPITAL
    shares = 0.0
    equity_curve = []
    trade_returns = []
    entry_price = None
    stoploss_price = None

    for i in range(len(df)):
        price = prices.iat[i]
        if position == 1 and price <= stoploss_price:
            cash += shares * price
            cash -= cash * COMMISSION
            trade_returns.append((price - entry_price)/entry_price)
            position, shares, entry_price, stoploss_price = 0, 0, None, None
        if signals['entry'].iat[i] and position == 0:
            entry_price = price*(1+SLIPPAGE_PCT)
            shares = cash/entry_price
            cash -= shares*entry_price + cash*COMMISSION
            position = 1
            stoploss_price = entry_price*(1-STOPLOSS_PCT)
        elif signals['exit'].iat[i] and position == 1:
            exit_price = price*(1-SLIPPAGE_PCT)
            cash += shares*exit_price
            cash -= cash*COMMISSION
            trade_returns.append((exit_price - entry_price)/entry_price)
            position, shares, entry_price, stoploss_price = 0, 0, None, None
        equity_curve.append(cash + shares*price if shares else cash)

    eq = pd.Series(equity_curve, index=df.index)
    total_return = eq.iloc[-1]/eq.iloc[0]-1
    days = (eq.index[-1]-eq.index[0]).total_seconds()/86400
    years = days/365.25
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/years)-1 if years>0 else np.nan
    max_dd = ((eq.cummax()-eq)/eq.cummax()).max()
    daily_rets = eq.resample('1D').last().pct_change().dropna()
    mean_ret, std_ret = daily_rets.mean(), daily_rets.std()
    downside_std = daily_rets[daily_rets<0].std()
    sharpe = mean_ret/std_ret*np.sqrt(252) if std_ret>0 else np.nan
    sortino = mean_ret/downside_std*np.sqrt(252) if downside_std>0 else np.nan
    calmar = cagr/max_dd if max_dd>0 else np.nan
    win_rate = np.mean(np.array(trade_returns)>0) if trade_returns else np.nan
    gross_profit = sum([r for r in trade_returns if r>0])
    gross_loss = abs(sum([r for r in trade_returns if r<0]))
    profit_factor = gross_profit/gross_loss if gross_loss>0 else np.nan

    metrics = {'total_return': total_return, 'cagr': cagr, 'max_drawdown': max_dd,
               'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar, 'trades': len(trade_returns),
               'win_rate': win_rate, 'profit_factor': profit_factor}
    if return_equity_curve:
        return eq, metrics
    else:
        return metrics


def run_grid():
    all_results = []

    # Fetch all symbols once with progress bars
    data_dict = {symbol: fetch_ohlcv(symbol, TIMEFRAME) for symbol in SYMBOLS}

    # Grid search per symbol
    for symbol, df in data_dict.items():
        combos = list(product(EMA_LENGTHS, ATR_LENGTHS, MULTIPLIERS))
        for ema_len, atr_len, mult in tqdm(combos, desc=f'Grid {symbol}'):
            if len(df) < max(ema_len, atr_len)+MIN_BARS:
                continue
            df_kc = compute_keltner(df, ema_len, atr_len, mult)
            signals = generate_signals(df_kc)
            metrics = backtest(df_kc, signals, return_equity_curve=False)
            row = {'symbol': symbol, 'ema_len': ema_len, 'atr_len': atr_len, 'multiplier': mult}
            row.update(metrics)
            all_results.append(row)

            # Clean up
            del df_kc, signals, metrics
            gc.collect()

    df_all = pd.DataFrame(all_results)

    # Aggregate only numeric metrics and avoid duplicating groupby columns
    group_cols = ['ema_len','atr_len','multiplier']
    numeric_cols = df_all.select_dtypes(include=[np.number]).columns.difference(group_cols)
    df_agg = df_all.groupby(group_cols)[numeric_cols].mean().reset_index()
    df_agg.to_csv(os.path.join(OUTPUT_DIR,'results_aggregated.csv'), index=False)

    df_first = data_dict[SYMBOLS[0]]

    # Top 5 by Sortino ratio
    top_sortino = df_agg.sort_values('sortino', ascending=False).head(TOP_N_PLOT)
    plt.figure(figsize=(12,8))
    for _, row in top_sortino.iterrows():
        df_kc = compute_keltner(df_first, row.
