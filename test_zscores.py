import pandas as pd
import numpy as np

panel = pd.read_csv(r"c:\Users\hardi\Downloads\Project Chronos\chronos_lean_workspace\data\consolidated_panel.csv", parse_dates=['date'])
panel.sort_values(['symbol', 'date'], inplace=True)
panel['fwd_ret'] = panel.groupby('symbol')['Close'].shift(-1) / panel['Close'] - 1.0

# Factors using raw magnitude (z-score like):
m10 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(10, min_periods=5).mean())
s10 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(10, min_periods=5).std().fillna(1e-4))
z10 = (panel['Close'] - m10) / (s10 + 1e-6)

m20 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(20, min_periods=5).mean())
s20 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(20, min_periods=5).std().fillna(1e-4))
z20 = (panel['Close'] - m20) / (s20 + 1e-6)

m60 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(60, min_periods=10).mean())
s60 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(60, min_periods=10).std().fillna(1e-4))
z60 = (panel['Close'] - m60) / (s60 + 1e-6)

# Range position relative to volatility
hl_range = (panel['Close'] - (panel['High'] + panel['Low'])/2.0) / (panel['High'] - panel['Low'] + 1e-4)
range_z = hl_range.rolling(5, min_periods=1).mean()

# Intraday momentum
intra = (panel['Close'] - panel['Open']) / (s20 + 1e-6)

def test_factor(name, series):
    df = panel.copy()
    df['alpha'] = series
    dates = sorted(df['date'].unique())[:-1]
    daily_returns = []
    
    for dt in dates:
        day = df[df['date'] == dt].dropna(subset=['alpha', 'fwd_ret'])
        if len(day) < 5:
            continue
        scores = day['alpha'].values
        # Demean
        demeaned = scores - np.mean(scores)
        sum_abs = np.sum(np.abs(demeaned))
        if sum_abs == 0:
            continue
        w = demeaned / sum_abs
        ret = np.sum(w * day['fwd_ret'].values)
        daily_returns.append(ret)
        
    r = np.array(daily_returns)
    sharpe = (np.mean(r) / np.std(r)) * np.sqrt(252)
    cum = np.cumprod(1 + r)
    peaks = np.maximum.accumulate(cum)
    dd = np.max((peaks - cum) / peaks)
    ann_ret = (cum[-1] ** (252.0 / len(r))) - 1.0
    print(f"[{name:<25}] Sharpe: {sharpe:.3f} | Max DD: {dd*100:.2f}% | AnnRet: {ann_ret*100:.2f}%")
    return r

r_z10 = test_factor("Z10", z10)
r_z20 = test_factor("Z20", z20)
r_z60 = test_factor("Z60", z60)
r_comb = test_factor("Z20*0.7 + Z10*0.3", z20 * 0.7 + z10 * 0.3)
r_comb2 = test_factor("Z20*0.6 + Range*0.4", z20 * 0.6 + range_z * 0.4)
r_accel = test_factor("Z10 - Z60 (Accel)", z10 - z60)
r_ens = test_factor("Ensemble Multi-Scale", z20 * 0.5 + z10 * 0.3 + range_z * 0.2)
