import pandas as pd
import numpy as np

panel = pd.read_csv(r"c:\Users\hardi\Downloads\Project Chronos\chronos_lean_workspace\data\consolidated_panel.csv", parse_dates=['date'])
panel.sort_values(['symbol', 'date'], inplace=True)
panel['fwd_ret'] = panel.groupby('symbol')['Close'].shift(-1) / panel['Close'] - 1.0

# Calculate candidate factors
print("Testing factor formulas...")

def evaluate_factor(name, alpha_series):
    df = panel.copy()
    df['alpha'] = alpha_series
    
    dates = sorted(df['date'].unique())[:-1] # drop last date
    daily_returns = []
    
    for dt in dates:
        day = df[df['date'] == dt]
        valid = day.dropna(subset=['alpha', 'fwd_ret'])
        if len(valid) < 5:
            continue
        scores = valid['alpha'].values
        demeaned = scores - np.mean(scores)
        sum_abs = np.sum(np.abs(demeaned))
        if sum_abs == 0:
            continue
        w = demeaned / sum_abs
        ret = np.sum(w * valid['fwd_ret'].values)
        daily_returns.append(ret)
        
    r = np.array(daily_returns)
    if len(r) == 0 or np.std(r) == 0:
        return None
    sharpe = (np.mean(r) / np.std(r)) * np.sqrt(252)
    cum = np.cumprod(1 + r)
    peaks = np.maximum.accumulate(cum)
    dd = np.max((peaks - cum) / peaks)
    ann_ret = (cum[-1] ** (252.0 / len(r))) - 1.0
    return {
        "name": name,
        "sharpe": round(sharpe, 3),
        "dd": round(dd * 100, 2),
        "ann_ret": round(ann_ret * 100, 2),
        "returns": r
    }

# 1. Trend Breakout: (Close - ts_mean(Close, 40)) / ts_std_dev(Close, 40)
mean40 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(40, min_periods=10).mean())
std40 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(40, min_periods=10).std().fillna(1e-4))
f_trend40 = (panel['Close'] - mean40) / std40

# 2. Dual Horizon Momentum: ts_rank(returns, 126) - ts_rank(returns, 10)
ret_col = panel.groupby('symbol')['Close'].pct_change().fillna(0)
def ts_rank_fast(s, w):
    return s.rolling(w, min_periods=5).apply(lambda x: (np.sum(x <= x[-1]) - 1) / max(len(x) - 1, 1), raw=True)

rank_126 = panel.groupby('symbol')['Close'].pct_change().transform(lambda s: s.rolling(60).mean() / (s.rolling(60).std() + 1e-4))
rank_5 = panel.groupby('symbol')['Close'].pct_change().transform(lambda s: s.rolling(5).mean() / (s.rolling(5).std() + 1e-4))
f_mom_spread = rank_126 - rank_5

# 3. High-Low Volatility Breakout: (Close - Low) / (High - Low + 1e-4) - 0.5
f_range = (panel['Close'] - panel['Low']) / (panel['High'] - panel['Low'] + 1e-4) - 0.5
f_range_smooth = panel.groupby('symbol')['Close'].transform(lambda s: f_range.rolling(5).mean())

# 4. Volatility-Adjusted Momentum: (Close / ts_mean(Close, 20) - 1) / (ts_std_dev(Close, 20) / Close)
mean20 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(20, min_periods=5).mean())
std20 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(20, min_periods=5).std().fillna(1e-4))
f_vol_adj_mom = (panel['Close'] - mean20) / (std20 + 1e-6)

# 5. Volume-Weighted Moving Average Trend: Close / VWAP - 1
vwap = (panel['Close'] * panel['Volume']).groupby(panel['symbol']).rolling(20, min_periods=5).sum() / (panel['Volume'].groupby(panel['symbol']).rolling(20, min_periods=5).sum() + 1e-4)
vwap = vwap.reset_index(level=0, drop=True)
f_vwap_trend = (panel['Close'] - vwap) / (vwap + 1e-4)

# 6. Sector-Neutralized Momentum: group_rank(ts_rank(returns, 60), sector) - group_rank(ts_rank(returns, 5), sector)
panel['mom60'] = panel.groupby('symbol')['Close'].pct_change(60).fillna(0)
panel['mom5'] = panel.groupby('symbol')['Close'].pct_change(5).fillna(0)
panel['sec_rank60'] = panel.groupby(['date', 'sector'])['mom60'].rank(pct=True)
panel['sec_rank5'] = panel.groupby(['date', 'sector'])['mom5'].rank(pct=True)
f_sec_mom = panel['sec_rank60'] - panel['sec_rank5']

factors = [
    ("Trend_Channel_40", f_trend40),
    ("Mom_Spread_Sharpe", f_mom_spread),
    ("Range_Breakout_5d", f_range_smooth),
    ("Vol_Adjusted_Mom_20d", f_vol_adj_mom),
    ("VWAP_Trend_20d", f_vwap_trend),
    ("Sector_Neutral_Mom", f_sec_mom)
]

for name, s in factors:
    res = evaluate_factor(name, s)
    if res:
        print(f"[{res['name']:<25}] Sharpe: {res['sharpe']:<6} | DD: {res['dd']}% | AnnRet: {res['ann_ret']}%")
