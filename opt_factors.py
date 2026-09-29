import pandas as pd
import numpy as np

panel = pd.read_csv(r"c:\Users\hardi\Downloads\Project Chronos\chronos_lean_workspace\data\consolidated_panel.csv", parse_dates=['date'])
panel.sort_values(['symbol', 'date'], inplace=True)
panel['fwd_ret'] = panel.groupby('symbol')['Close'].shift(-1) / panel['Close'] - 1.0

# Build enhanced factors
# 1. Vol-adjusted momentum (20d):
m20 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(20, min_periods=5).mean())
s20 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(20, min_periods=5).std().fillna(1e-4))
f1 = (panel['Close'] - m20) / (s20 + 1e-6)

# 2. Short-term acceleration / reversal: -(Close - Open) / Open
f2 = -1.0 * (panel['Close'] - panel['Open']) / panel['Open']

# 3. Normalized Range Location (5d):
f3_raw = (panel['Close'] - panel['Low']) / (panel['High'] - panel['Low'] + 1e-4) - 0.5
f3 = panel.groupby('symbol')['Close'].transform(lambda s: f3_raw.rolling(5).mean())

# 4. Long-term trend slope (60d):
m60 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(60, min_periods=10).mean())
s60 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(60, min_periods=10).std().fillna(1e-4))
f4 = (panel['Close'] - m60) / (s60 + 1e-6)

# 5. Volatility / Volume expansion filter:
vol_z = panel.groupby('symbol')['Volume'].transform(lambda s: (s - s.rolling(20).mean()) / (s.rolling(20).std() + 1e-4))
f5 = f1 * (1.0 + 0.2 * vol_z.clip(-2, 2))

# 6. Sector-neutralized versions:
panel['f1'] = f1
panel['f2'] = f2
panel['f3'] = f3
panel['f4'] = f4
panel['f5'] = f5

r_f1 = panel.groupby(['date', 'sector'])['f1'].rank(pct=True) - 0.5
r_f2 = panel.groupby(['date', 'sector'])['f2'].rank(pct=True) - 0.5
r_f3 = panel.groupby(['date', 'sector'])['f3'].rank(pct=True) - 0.5
r_f4 = panel.groupby(['date', 'sector'])['f4'].rank(pct=True) - 0.5
r_f5 = panel.groupby(['date', 'sector'])['f5'].rank(pct=True) - 0.5

# Test combinations:
combos = [
    ("F1_Raw", f1),
    ("F1_SectorNeutral", r_f1),
    ("F1_plus_F2_Rev", f1 * 0.8 + f2 * 0.2),
    ("F1_plus_F3_Range", f1 * 0.7 + f3 * 0.3),
    ("F1_plus_F4_Trend", f1 * 0.6 + f4 * 0.4),
    ("F5_VolEnhanced", f5),
    ("Ensemble_Opt", f1 * 0.5 + f3 * 0.3 + f4 * 0.2)
]

for name, s in combos:
    df = panel.copy()
    df['alpha'] = s
    dates = sorted(df['date'].unique())[:-1]
    daily_returns = []
    
    for dt in dates:
        day = df[df['date'] == dt].dropna(subset=['alpha', 'fwd_ret'])
        if len(day) < 5:
            continue
        scores = day['alpha'].values
        # Apply truncation / outlier clipping like WorldQuant (truncation 0.08):
        scores = np.clip(scores, np.percentile(scores, 5), np.percentile(scores, 95))
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
    print(f"[{name:<20}] Sharpe: {sharpe:.3f} | Max DD: {dd*100:.2f}% | AnnRet: {ann_ret*100:.2f}%")
