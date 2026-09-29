import pandas as pd
import numpy as np

panel = pd.read_csv(r"c:\Users\hardi\Downloads\Project Chronos\chronos_lean_workspace\data\consolidated_panel.csv", parse_dates=['date'])
panel.sort_values(['symbol', 'date'], inplace=True)
panel['fwd_ret'] = panel.groupby('symbol')['Close'].shift(-1) / panel['Close'] - 1.0

# Factor 1: Volatility-Adjusted Momentum 20d: (Close - Mean20) / Std20
mean20 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(20, min_periods=5).mean())
std20 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(20, min_periods=5).std().fillna(1e-4))
f1 = (panel['Close'] - mean20) / (std20 + 1e-6)

# Factor 2: Trend Channel 40d: (Close - Mean40) / Std40
mean40 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(40, min_periods=10).mean())
std40 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(40, min_periods=10).std().fillna(1e-4))
f2 = (panel['Close'] - mean40) / (std40 + 1e-6)

# Factor 3: Short-term Range Breakout 5d
f3_raw = (panel['Close'] - panel['Low']) / (panel['High'] - panel['Low'] + 1e-4) - 0.5
f3 = panel.groupby('symbol')['Close'].transform(lambda s: f3_raw.rolling(5).mean())

# Factor 4: 60-day Risk-Adjusted Momentum
ret20 = panel.groupby('symbol')['Close'].pct_change(20).fillna(0)
vol20 = panel.groupby('symbol')['Close'].pct_change().transform(lambda s: s.rolling(20).std().fillna(1e-4))
f4 = ret20 / (vol20 + 1e-4)

# Factor 5: VWAP Reversion / Trend Interaction:
vwap = (panel['Close'] * panel['Volume']).groupby(panel['symbol']).rolling(20, min_periods=5).sum() / (panel['Volume'].groupby(panel['symbol']).rolling(20, min_periods=5).sum() + 1e-4)
vwap = vwap.reset_index(level=0, drop=True)
f5 = (panel['Close'] - vwap) / (vwap + 1e-4)

# Assign factors to panel columns
panel['f1'] = f1
panel['f2'] = f2
panel['f3'] = f3
panel['f4'] = f4
panel['f5'] = f5

# Rank each factor cross-sectionally
panel['r_f1'] = panel.groupby('date')['f1'].rank(pct=True) - 0.5
panel['r_f2'] = panel.groupby('date')['f2'].rank(pct=True) - 0.5
panel['r_f3'] = panel.groupby('date')['f3'].rank(pct=True) - 0.5
panel['r_f4'] = panel.groupby('date')['f4'].rank(pct=True) - 0.5
panel['r_f5'] = panel.groupby('date')['f5'].rank(pct=True) - 0.5

# Test individual correlations
corr = panel[['r_f1', 'r_f2', 'r_f3', 'r_f4', 'r_f5']].corr()
print("Factor Correlation Matrix:")
print(corr)

# Ensemble tests:
ensembles = [
    ("Ensemble_1_2", panel['r_f1'] * 0.6 + panel['r_f2'] * 0.4),
    ("Ensemble_1_3", panel['r_f1'] * 0.6 + panel['r_f3'] * 0.4),
    ("Ensemble_1_4", panel['r_f1'] * 0.5 + panel['r_f4'] * 0.5),
    ("Ensemble_1_2_4", panel['r_f1'] * 0.4 + panel['r_f2'] * 0.3 + panel['r_f4'] * 0.3),
    ("Ensemble_All", panel['r_f1'] * 0.3 + panel['r_f2'] * 0.25 + panel['r_f3'] * 0.15 + panel['r_f4'] * 0.3)
]

def eval_series(name, alpha_series):
    df = panel.copy()
    df['alpha'] = alpha_series
    dates = sorted(df['date'].unique())[:-1]
    daily_returns = []
    
    for dt in dates:
        day = df[df['date'] == dt].dropna(subset=['alpha', 'fwd_ret'])
        if len(day) < 5:
            continue
        scores = day['alpha'].values
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

print("\nEnsemble Performance:")
for name, ens in ensembles:
    eval_series(name, ens)
