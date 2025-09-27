"""
Keltner Channel strategy optimizer for Binance historical data

Usage:
  - Edit SYMBOLS, TIMEFRAME, START_DATE as needed (defaults: ['BTC/USDT'], '4h', '2022-01-01')
  - Install requirements: pip install ccxt pandas numpy scipy tqdm matplotlib
  - Run: python keltner_channel_optimizer.py

What it does:
  1. Downloads OHLCV from Binance (public REST via ccxt)
  2. Computes EMA (middle), ATR, and Keltner bands
  3. Generates breakout signals (long on close above upper band, exit on close below mean band or stoploss)
  4. Vectorized backtest (fixed position sizing: fraction of equity)
  5. Grid search over EMA_length, ATR_length, multiplier
  6. Tests multiple symbols and aggregates results across them
  7. Outputs CSVs with metrics aggregated by parameter set (summed/averaged)
  8. Plots a single PNG containing all equity curves across parameter combos (symbols aggregated)

Notes:
  - This is a simple, educational framework. Extend with commissions, slippage,
    position sizing, risk management, walk-forward validation, cross-validation.
  - Use out-of-sample / walk-forward testing before deploying live.

"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# --------------------------- USER CONFIG ---------------------------
SYMBOLS = ['BTC/USDT', 'ETH/USDT']  # Binance pairs to test simultaneously
TIMEFRAME = '4h'                    # ccxt timeframe
START_DATE = '2022-01-01'           # UTC start (YYYY-MM-DD)
END_DATE = None                     # None for now -> uses most recent available
EXCHANGE = 'binance'

# Parameter grid to search
EMA_LENGTHS = [30, 60, 90, 120]
ATR_LENGTHS = [30, 60, 90, 120]
MULTIPLIERS = [x * 0.5 for x in range(2, 7)]  # 1.0 to 3.0 in 0.5 steps

# Backtest settings
INITIAL_CAPITAL = 10000.0    # USD
RISK_PER_TRADE = 0.01        # fraction of equity risked per trade (used for sizing if stoploss used)
COMMISSION = 0.00075         # proportion per trade (example, adjust to Binance futures/spot fees)
SLIPPAGE_PCT = 0.0005        # fraction of price lost to slippage on entries/exits
STOPLOSS_PCT = 0.05          # initial stoploss of 5% from entry
MIN_BARS = 200               # minimum bars required for indicator lengths

# Output
OUTPUT_DIR = 'keltner_optimizer_out'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------- HELPERS ---------------------------

def fetch_ohlcv_ccxt(symbol, timeframe, since_iso):
    exchange = getattr(ccxt, EXCHANGE)({'enableRateLimit': True})
    since_ms = int(pd.to_datetime(since_iso).timestamp() * 1000)
    all_bars = []
    limit = 1000
    while True:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        if not bars:
            break
        all_bars += bars
        since_ms = bars[-1][0] + 1
        if len(bars) < limit:
            break
    df = pd.DataFrame(all_bars, columns=['timestamp','open','high','low','close','volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('datetime')
    df = df[~df.index.duplicated(keep='first')]
    return df


def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def atr(df, length):
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean()


def compute_keltner(df, ema_len, atr_len, mult):
    middle = ema(df['close'], ema_len)
    a = atr(df, atr_len)
    upper = middle + mult * a
    lower = middle - mult * a
    out = df.copy()
    out['kc_middle'] = middle
    out['kc_upper'] = upper
    out['kc_lower'] = lower
    return out


def generate_signals(df):
    close = df['close']
    upper = df['kc_upper']
    middle = df['kc_middle']

    long_entry = (close.shift(1) <= upper.shift(1)) & (close > upper)  # entry: close above upper band
    long_exit = (close.shift(1) >= middle.shift(1)) & (close < middle)  # exit: close below mean band

    signals = pd.DataFrame(index=df.index)
    signals['entry'] = long_entry.astype(int)
    signals['exit'] = long_exit.astype(int)
    return signals


def backtest_vectorized(df, signals, initial_capital=INITIAL_CAPITAL):
    prices = df['close']
    entries = signals['entry']
    exits = signals['exit']

    position = 0
    equity = initial_capital
    cash = initial_capital
    shares = 0.0

    equity_curve = []
    trade_returns = []
    entry_price = None
    stoploss_price = None

    for i in range(len(df)):
        price = prices.iat[i]

        # check stoploss
        if position == 1 and price <= stoploss_price:
            cash += shares * price  # exit at stoploss
            commission = equity * COMMISSION
            cash -= commission
            trade_return = (price - entry_price) / entry_price if entry_price else 0
            trade_returns.append(trade_return)
            shares = 0
            position = 0
            stoploss_price = None
            entry_price = None

        # normal entry
        if entries.iat[i] and position == 0:
            entry_price = price * (1 + SLIPPAGE_PCT)
            shares = cash / entry_price
            commission = equity * COMMISSION
            cash -= shares * entry_price + commission
            position = 1
            stoploss_price = entry_price * (1 - STOPLOSS_PCT)

        # normal exit (close below mean band)
        elif exits.iat[i] and position == 1:
            exit_price = price * (1 - SLIPPAGE_PCT)
            cash += shares * exit_price
            commission = equity * COMMISSION
            cash -= commission
            trade_return = (exit_price - entry_price) / entry_price if entry_price else 0
            trade_returns.append(trade_return)
            shares = 0
            position = 0
            stoploss_price = None
            entry_price = None

        current_value = cash + (shares * price if shares else 0)
        equity = current_value
        equity_curve.append(equity)

    eq = pd.Series(equity_curve, index=df.index)
    total_return = eq.iloc[-1] / eq.iloc[0] - 1
    days = (eq.index[-1] - eq.index[0]).total_seconds() / (3600*24)
    years = days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    rolling_max = eq.cummax()
    drawdown = (eq - rolling_max) / rolling_max
    max_dd = drawdown.min()

    daily_rets = eq.resample('1D').last().pct_change().dropna()
    mean_ret = daily_rets.mean()
    std_ret = daily_rets.std()
    downside_std = daily_rets[daily_rets < 0].std()

    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else np.nan
    sortino = (mean_ret / downside_std) * np.sqrt(252) if downside_std > 0 else np.nan
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    volatility = std_ret * np.sqrt(252)

    trades = int(signals['entry'].sum())
    win_rate = (np.array(trade_returns) > 0).mean() if trade_returns else np.nan

    metrics = {
        'equity_curve': eq,
        'total_return': total_return,
        'cagr': cagr,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'volatility': volatility,
        'trades': trades,
        'win_rate': win_rate
    }

    return eq, metrics
