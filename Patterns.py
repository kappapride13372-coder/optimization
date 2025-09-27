"""
Fully working Keltner Channel strategy optimizer for Binance on headless servers

Features:
- 4h timeframe
- Grid search over EMA lengths, ATR lengths, multipliers
- Entry: close above upper band
- Exit: close below mean band OR 5% stoploss
- Multi-symbol support
- Risk and performance metrics (CAGR, Sharpe, Sortino, Calmar, max DD, win rate)
- Generates CSVs per symbol and aggregated
- Generates one PNG with all equity curves
- Matplotlib uses Agg backend for droplet/headless compatibility
"""

import matplotlib
matplotlib.use('Agg')  # headless backend
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
START_DATE = '2022-01-01'
EMA_LENGTHS = [30, 60, 90, 120]
ATR_LENGTHS = [30, 60, 90, 120]
MULTIPLIERS = [x * 0.5 for x in range(2, 7)]
INITIAL_CAPITAL = 10000.0
COMMISSION = 0.00075
SLIPPAGE_PCT = 0.0005
STOPLOSS_PCT = 0.05
MIN_BARS = 200
OUTPUT_DIR = 'keltner_optimizer_out'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------- HELPERS ---------------------------

def fetch_ohlcv(symbol, timeframe, since_iso):
    exchange = ccxt.binance({'enableRateLimit': True})
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
    df.set_index('datetime', inplace=True)
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

    long_entry = (close.shift(1) <= upper.shift(1)) & (close > upper)  # entry: close above upper
    long_exit = (close.shift(1) >= middle.shift(1)) & (close < middle)  # exit: close below mean

    signals = pd.DataFrame(index=df.index)
    signals['entry'] = long_entry.astype(int)
    signals['exit'] = long_exit.astype(int)
    return signals


def backtest(df, signals):
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

        # stoploss check
        if position == 1 and price <= stoploss_price:
            cash += shares * price
            cash -= cash * COMMISSION
            trade_returns.append((price - entry_price) / entry_price)
            position, shares, entry_price, stoploss_price = 0, 0, None, None

        # entry
        if signals['entry'].iat[i] and position == 0:
            entry_price = price * (1 + SLIPPAGE_PCT)
            shares = cash / entry_price
            cash -= shares * entry_price + cash * COMMISSION
            position = 1
            stoploss_price = entry_price * (1 - STOPLOSS_PCT)

        # exit
        elif signals['exit'].iat[i] and position == 1:
            exit_price = price * (1 - SLIPPAGE_PCT)
            cash += shares * exit_price
            cash -= cash * COMMISSION
            trade_returns.append((exit_price - entry_price) / entry_price)
            position, shares, entry_price, stoploss_price = 0, 0, None, None

        equity_curve.append(cash + shares * price if shares else cash)

    eq = pd.Series(equity_curve, index=df.index)
    total_return = eq.iloc[-1]/eq.iloc[0] - 1
    days = (eq.index[-1]-eq.index[0]).total_seconds()/86400
    years = days/365.25
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/years) -1 if years>0 else np.nan
    max_dd = ((eq.cummax() - eq)/eq.cummax()).max()
    daily_rets = eq.resample('1D').last().pct_change().dropna()
    mean_ret, std_ret = daily_rets.mean(), daily_rets.std()
    downside_std = daily_rets[daily_rets<0].std()
    sharpe = mean_ret/std_ret*np.sqrt(252) if std_ret>0 else np.nan
    sortino = mean_ret/downside_std*np.sqrt(252) if downside_std>0 else np.nan
    calmar = cagr/max_dd if max_dd>0 else np.nan
    win_rate = np.mean(np.array(trade_returns)>0) if trade_returns else np.nan

    metrics = {'equity_curve': eq, 'total_return': total_return, 'cagr': cagr, 'max_drawdown': max_dd,
               'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar, 'trades': len(trade_returns), 'win_rate': win_rate}
    return eq, metrics


def run_grid():
    results = []
    equity_curves = []
    combos = list(product(EMA_LENGTHS, ATR_LENGTHS, MULTIPLIERS))

    for symbol in SYMBOLS:
        print(f'Fetching {symbol}...')
        df = fetch_ohlcv(symbol, TIMEFRAME, START_DATE)
        for ema_len, atr_len, mult in tqdm(combos, desc=f'Grid {symbol}'):
            if len(df) < max(ema_len, atr_len)+MIN_BARS:
                continue
            df_kc = compute_keltner(df, ema_len, atr_len, mult)
            signals = generate_signals(df_kc)
            eq, metrics = backtest(df_kc, signals)

            results.append({'symbol': symbol, 'ema_len': ema_len, 'atr_len': atr_len, 'multiplier': mult,
                            'total_return': metrics['total_return'], 'cagr': metrics['cagr'], 'max_drawdown': metrics['max_drawdown'],
                            'sharpe': metrics['sharpe'], 'sortino': metrics['sortino'], 'calmar': metrics['calmar'],
                            'trades': metrics['trades'], 'win_rate': metrics['win_rate']})
            equity_curves.append((f'{symbol}_ema{ema_len}_atr{atr_len}_m{mult}', metrics['equity_curve']/metrics['equity_curve'].iloc[0]))

    df_all = pd.DataFrame(results)
    df_agg = df_all.groupby(['ema_len','atr_len','multiplier']).mean().reset_index()
    return df_all, df_agg, equity_curves


def plot_equity_curves(equity_curves):
    plt.figure(figsize=(12,8))
    for label, eq in equity_curves:
        plt.plot(eq.index, eq.values, alpha=0.3, linewidth=1)
    plt.title('Equity Curves for All Parameter Combos')
    plt.xlabel('Date')
    plt.ylabel('Normalized Equity')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, 'equity_curves.png')
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f'Equity curves saved to {out_png}')


if __name__ == '__main__':
    all_results, agg_results, equity_curves = run_grid()
    all_results.to_csv(os.path.join(OUTPUT_DIR,'results_per_symbol.csv'), index=False)
    agg_results.to_csv(os.path.join(OUTPUT_DIR,'results_aggregated.csv'), index=False)
    plot_equity_curves(equity_curves)
    if not agg_results.empty:
        print('\nBest aggregated params (by CAGR):')
        print(agg_results.sort_values('cagr', ascending=False).iloc[0])
    print('\nDone.')
