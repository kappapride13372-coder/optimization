"""
Memory-efficient Keltner Channel optimizer with full historical data for droplets

Features:
- 4h timeframe with full history
- Grid search over EMA lengths, ATR lengths, multipliers
- Entry: close above upper band
- Exit: close below mean band OR 5% stoploss
- Processes one symbol at a time to save memory
- Metrics: CAGR, Sharpe, Sortino, Calmar, max DD, win rate, profit factor
- Only top 10 combos per symbol by profit factor are plotted
- Headless Matplotlib for droplets
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ccxt
import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
import os

# --------------------------- CONFIG ---------------------------
SYMBOLS = ['BTC/USDT', 'ETH/USDT']
TIMEFRAME = '4h'
EMA_LENGTHS = [30, 60, 90, 120]
ATR_LENGTHS = [30, 60, 90, 120]
MULTIPLIERS = [x * 0.5 for x in range(2, 7)]
INITIAL_CAPITAL = 10000.0
COMMISSION = 0.00075
SLIPPAGE_PCT = 0.0005
STOPLOSS_PCT = 0.05
MIN_BARS = 200
TOP_N_PLOT = 10
OUTPUT_DIR = 'keltner_optimizer_out'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------- HELPERS ---------------------------

def fetch_ohlcv(symbol, timeframe, start_date='2022-01-01'):
    exchange = ccxt.binance({'enableRateLimit': True})
    since_ms = int(pd.to_datetime(start_date).timestamp() * 1000)
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
    df.set_index('datetime', inplace=True)
    df = df[['open','high','low','close','volume']]
    df = df[~df.index.duplicated(keep='first')]
    return df


def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def atr(df, length):
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
    for symbol in SYMBOLS:
        print(f'Fetching {symbol}...')
        df = fetch_ohlcv(symbol, TIMEFRAME)
        results = []
        combos = list(product(EMA_LENGTHS, ATR_LENGTHS, MULTIPLIERS))
        for ema_len, atr_len, mult in tqdm(combos, desc=f'Grid {symbol}'):
            if len(df) < max(ema_len, atr_len)+MIN_BARS:
                continue
            df_kc = compute_keltner(df, ema_len, atr_len, mult)
            signals = generate_signals(df_kc)
            metrics = backtest(df_kc, signals, return_equity_curve=False)
            row = {'symbol': symbol, 'ema_len': ema_len, 'atr_len': atr_len, 'multiplier': mult}
            row.update(metrics)
            results.append(row)
            del df_kc, signals  # free memory
        df_results = pd.DataFrame(results)
        df_results.to_csv(os.path.join(OUTPUT_DIR,f'results_{symbol.replace("/","_")}.csv'), index=False)

        # plot top N by profit factor
        topN = df_results.sort_values('profit_factor', ascending=False).head(TOP_N_PLOT)
        plt.figure(figsize=(12,8))
        for _, row in topN.iterrows():
            df_kc = compute_keltner(df, row.ema_len, row.atr_len, row.multiplier)
            signals = generate_signals(df_kc)
            eq_curve, _ = backtest(df_kc, signals, return_equity_curve=True)
            plt.plot(eq_curve.index, eq_curve.values/eq_curve.iloc[0], label=f"EMA{row.ema_len}_ATR{row.atr_len}_M{row.multiplier}")
            del df_kc, signals, eq_curve
        plt.title(f'Top {TOP_N_PLOT} Equity Curves for {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Normalized Equity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR,f'top{TOP_N_PLOT}_equity_{symbol.replace("/","_")}.png'), dpi=150)
        plt.close()


if __name__ == '__main__':
    run_grid()
    print('Done.')
