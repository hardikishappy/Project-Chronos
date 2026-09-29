import pandas as pd
import numpy as np

panel = pd.read_csv(r"c:\Users\hardi\Downloads\Project Chronos\chronos_lean_workspace\data\consolidated_panel.csv", parse_dates=['date'])
panel.sort_values(['symbol', 'date'], inplace=True)
panel['fwd_ret'] = panel.groupby('symbol')['Close'].shift(-1) / panel['Close'] - 1.0

# Base factor: Volatility-Adjusted Momentum (20d)
m20 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(20, min_periods=5).mean())
s20 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(20, min_periods=5).std().fillna(1e-4))
f1 = (panel['Close'] - m20) / (s20 + 1e-6)

# Range factor (5d)
f3_raw = (panel['Close'] - panel['Low']) / (panel['High'] - panel['Low'] + 1e-4) - 0.5
f3 = panel.groupby('symbol')['Close'].transform(lambda s: f3_raw.rolling(5).mean())

# Combine
alpha_raw = f1 * 0.7 + f3 * 0.3
panel['raw_alpha'] = alpha_raw

for decay_days in [1, 2, 4, 6, 8, 12]:
    if decay_days == 1:
        panel['decayed_alpha'] = panel['raw_alpha']
    else:
        # Linear decay over decay_days: weights [1, 2, ..., K]
        weights = np.arange(1, decay_days + 1, dtype=float)
        weights /= weights.sum()
        panel['decayed_alpha'] = panel.groupby('symbol')['raw_alpha'].transform(
            lambda s: s.rolling(decay_days, min_periods=1).apply(lambda x: np.dot(x, weights[-len(x):]) if len(x) > 0 else np.nan, raw=True)
        )

    df = panel.copy()
    dates = sorted(df['date'].unique())[:-1]
    daily_returns = []
    turnover_list = []
    prev_w = {}
    
    for dt in dates:
        day = df[df['date'] == dt].dropna(subset=['decayed_alpha', 'fwd_ret'])
        if len(day) < 5:
            continue
        scores = day['decayed_alpha'].values
        scores = np.clip(scores, np.percentile(scores, 5), np.percentile(scores, 95))
        demeaned = scores - np.mean(scores)
        sum_abs = np.sum(np.abs(demeaned))
        if sum_abs == 0:
            continue
        w = demeaned / sum_abs
        ret = np.sum(w * day['fwd_ret'].values)
        daily_returns.append(ret)
        
        curr_w = dict(zip(day['symbol'], w))
        all_syms = set(prev_w.keys()).union(curr_w.keys())
        to = 0.5 * sum(abs(curr_w.get(s, 0.0) - prev_w.get(s, 0.0)) for s in all_syms)
        turnover_list.append(to)
        prev_w = curr_w
        
    r = np.array(daily_returns)
    sharpe = (np.mean(r) / np.std(r)) * np.sqrt(252)
    cum = np.cumprod(1 + r)
    peaks = np.maximum.accumulate(cum)
    dd = np.max((peaks - cum) / peaks)
    ann_ret = (cum[-1] ** (252.0 / len(r))) - 1.0
    avg_to = np.mean(turnover_list)
    print(f"[Decay = {decay_days:2d}] Sharpe: {sharpe:.3f} | Max DD: {dd*100:.2f}% | AnnRet: {ann_ret*100:.2f}% | Turnover: {avg_to*100:.2f}%")
