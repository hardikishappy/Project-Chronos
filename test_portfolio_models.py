import pandas as pd
import numpy as np

panel = pd.read_csv(r"c:\Users\hardi\Downloads\Project Chronos\chronos_lean_workspace\data\consolidated_panel.csv", parse_dates=['date'])
panel.sort_values(['symbol', 'date'], inplace=True)
panel['fwd_ret'] = panel.groupby('symbol')['Close'].shift(-1) / panel['Close'] - 1.0

# 1. Residual Momentum (Price vs MA20 / Vol20)
m20 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(20, min_periods=5).mean())
s20 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(20, min_periods=5).std().fillna(1e-4))
sig_mom = (panel['Close'] - m20) / (s20 + 1e-6)

# 2. Reversal on Intraday Stretch
intraday = (panel['Close'] - panel['Open']) / panel['Open']
# Short after extreme down day or up day?
# If we test short-term mean reversion on 3-day return:
ret3 = panel.groupby('symbol')['Close'].pct_change(3).fillna(0)
sig_rev3 = -1.0 * ret3 / (s20 / m20 + 1e-4)

# 3. Range location:
sig_range = (panel['Close'] - panel['Low']) / (panel['High'] - panel['Low'] + 1e-4) - 0.5

# 4. Volatility contraction to expansion:
# Lower 10d vol relative to 60d vol (squeeze before move)
vol10 = panel.groupby('symbol')['Close'].pct_change().transform(lambda s: s.rolling(10).std().fillna(1e-4))
vol60 = panel.groupby('symbol')['Close'].pct_change().transform(lambda s: s.rolling(60).std().fillna(1e-4))
vol_ratio = vol10 / (vol60 + 1e-4)

# Test various weightings
panel['sig_mom'] = sig_mom
panel['sig_range'] = sig_range
panel['sig_rev3'] = sig_rev3

# Cross-sectional percentile ranks
panel['r_mom'] = panel.groupby('date')['sig_mom'].rank(pct=True) - 0.5
panel['r_range'] = panel.groupby('date')['sig_range'].rank(pct=True) - 0.5
panel['r_rev3'] = panel.groupby('date')['sig_rev3'].rank(pct=True) - 0.5

# Alpha 1: Vol-Adjusted Trend (F1)
# Alpha 2: Range Breakout (F2)
# Alpha 3: High-Conviction Top/Bottom Quantile Filter
# What if we only trade the top & bottom quintiles (long top 20%, short bottom 20%)?
def eval_alpha(name, alpha_col, quintile_filter=False):
    df = panel.copy()
    dates = sorted(df['date'].unique())[:-1]
    daily_returns = []
    
    for dt in dates:
        day = df[df['date'] == dt].dropna(subset=[alpha_col, 'fwd_ret'])
        if len(day) < 5:
            continue
        scores = day[alpha_col].values
        
        if quintile_filter:
            # Only long top 25%, short bottom 25%, zero out middle 50%
            q_high = np.percentile(scores, 75)
            q_low = np.percentile(scores, 25)
            filtered_scores = np.where(scores >= q_high, scores - q_high, np.where(scores <= q_low, scores - q_low, 0.0))
            scores = filtered_scores
            
        demeaned = scores - np.mean(scores)
        sum_abs = np.sum(np.abs(demeaned))
        if sum_abs < 1e-7:
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
    print(f"[{name:<30}] Sharpe: {sharpe:.3f} | Max DD: {dd*100:.2f}% | AnnRet: {ann_ret*100:.2f}%")

panel['ens_1'] = panel['r_mom'] * 0.7 + panel['r_range'] * 0.3
panel['ens_2'] = panel['r_mom'] * 0.6 + panel['r_range'] * 0.2 + panel['r_rev3'] * 0.2
panel['ens_3'] = np.sign(panel['r_mom']) * (np.abs(panel['r_mom']) ** 1.5) + panel['r_range'] * 0.3

eval_alpha("Ens1_Linear", 'ens_1', quintile_filter=False)
eval_alpha("Ens1_Quintile", 'ens_1', quintile_filter=True)
eval_alpha("Ens2_MultiFactor", 'ens_2', quintile_filter=False)
eval_alpha("Ens2_Quintile", 'ens_2', quintile_filter=True)
eval_alpha("Ens3_Convex", 'ens_3', quintile_filter=False)
eval_alpha("Ens3_Convex_Quintile", 'ens_3', quintile_filter=True)
