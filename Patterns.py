"""
Memory-efficient Keltner Channel strategy optimizer with safe data fetching for Binance droplets

Features:
- Fetches OHLCV in chunks with local CSV caching
- 4h timeframe (can be changed)
- Grid search over EMA lengths, ATR lengths, multipliers
- Entry: close above upper band
- Exit: close below mean band OR 5% stoploss
- Multi-symbol support
- Metrics: CAGR, Sharpe, Sortino, Calmar, max DD, win rate, profit factor
- Only top 10 combos by profit factor are plotted
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
import time

# --------------------------- CONFIG ---------------------------
SYMBOLS = ['BTC/USDT', 'ETH/USDT']
TIMEFRAME = '4h'
LOOKBACK_DAYS = 365  # fetch only last 1 year to save memory
EMA_LENGTHS = [30, 60, 90, 120]
ATR_LENGTHS = [30, 60, 90, 120]
MULTIPLIERS = [x * 0.5 for x in range(2, 7)]
INITIAL_CAPITAL = 10000.0
COMMISSION = 0.001
SLIPPAGE_PCT = 0.0005
STOPLOSS_PCT = 0.05
MIN_BARS = 200
OUTPUT_DIR = 'keltner_optimizer_out'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------- HELPERS ---------------------------

def fetch_ohlcv(symbol, timeframe, lookback_days=LOOKBACK_DAYS):
    exchange = ccxt.binance({'enableRateLimit': True})
    since_iso = (pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    symbol_file = f'{OUTPUT_DIR}/{symbol.replace("/","_")}_ohlcv.csv'

    if os.path.exists(symbol_file):
        df = pd.read_csv(symbol_file, index_col='datetime', parse_dates=True)
        print(f'Loaded cached data for {symbol} from {symbol_file}')
        return df

    print(f'Fetching {symbol} OHLCV from Binance...')
    since_ms = int(pd.to_datetime(since_iso).timestamp() * 1000)
    all_bars = []
    limit = 1000
    while True:
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        except ccxt.NetworkError as e:
            print(f'Network error: {e}, retrying in 5s...')
            time.sleep(5)
            continue
        if not bars:
            break
        all_bars += bars
        since_ms = bars[-1][0] + 1
        if len(bars) < limit:
            break
        time.sleep(0.2)  # avoid rate limit

    df = pd.DataFrame(all_bars, columns=['timestamp','open','high','low','close','volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    df = df[['open','high','low','close','volume']]
    df = df[~df.index.duplicated(keep='first')]
    df.to_csv(symbol_file)
    print(f'Saved fetched data for {symbol} to {symbol_file}')
    return df

# EMA, ATR, compute_keltner, generate_signals, backtest remain the same as previous memory-efficient version
# run_grid and plot_top10 functions remain the same, using fetch_ohlcv with caching and LOOKBACK_DAYS limit



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

    metrics = {'equity_curve': eq, 'total_return': total_return, 'cagr': cagr, 'max_drawdown': max_dd,
               'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar, 'trades': len(trade_returns),
               'win_rate': win_rate, 'profit_factor': profit_factor}
    if return_equity_curve:
        return eq, metrics
    else:
        return metrics


def run_grid():
    results = []
    combos = list(product(EMA_LENGTHS, ATR_LENGTHS, MULTIPLIERS))

    for symbol in SYMBOLS:
        print(f'Fetching {symbol}...')
        df = fetch_ohlcv(symbol, TIMEFRAME, START_DATE)
        for ema_len, atr_len, mult in tqdm(combos, desc=f'Grid {symbol}'):
            if len(df) < max(ema_len, atr_len)+MIN_BARS:
                continue
            df_kc = compute_keltner(df, ema_len, atr_len, mult)
            signals = generate_signals(df_kc)
            metrics = backtest(df_kc, signals, return_equity_curve=False)
            row = {'symbol': symbol, 'ema_len': ema_len, 'atr_len': atr_len, 'multiplier': mult}
            row.update(metrics)
            results.append(row)

    df_all = pd.DataFrame(results)
    df_agg = df_all.groupby(['ema_len','atr_len','multiplier']).mean().reset_index()
    return df_all, df_agg, df


def plot_top10(df_all, df_agg, df_full):
    top10 = df_agg.sort_values('profit_factor', ascending=False).head(10)
    plt.figure(figsize=(12,8))
    for _, row in top10.iterrows():
        metrics = backtest(df_full, generate_signals(compute_keltner(df_full, row.ema_len, row.atr_len, row.multiplier)), return_equity_curve=True)
        eq_curve, _ = metrics
        plt.plot(eq_curve.index, eq_curve.values/eq_curve.iloc[0], label=f"EMA{row.ema_len}_ATR{row.atr_len}_M{row.multiplier}")
    plt.title('Top 10 Equity Curves by Profit Factor')
    plt.xlabel('Date')
    plt.ylabel('Normalized Equity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, 'top10_equity_curves.png')
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f'Top 10 equity curves saved to {out_png}')


if __name__ == '__main__':
    all_results, agg_results, df_full = run_grid()
    all_results.to_csv(os.path.join(OUTPUT_DIR,'results_per_symbol.csv'), index=False)
    agg_results.to_csv(os.path.join(OUTPUT_DIR,'results_aggregated.csv'), index=False)
    plot_top10(all_results, agg_results, df_full)
    print('\nDone.')
