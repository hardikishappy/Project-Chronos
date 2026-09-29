import pandas as pd
import numpy as np

panel = pd.read_csv(r"c:\Users\hardi\Downloads\Project Chronos\chronos_lean_workspace\data\consolidated_panel.csv", parse_dates=['date'])
panel.sort_values(['symbol', 'date'], inplace=True)
panel['fwd_ret'] = panel.groupby('symbol')['Close'].shift(-1) / panel['Close'] - 1.0

m10 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(10, min_periods=5).mean())
s10 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(10, min_periods=5).std().fillna(1e-4))
z10 = (panel['Close'] - m10) / (s10 + 1e-6)

m20 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(20, min_periods=5).mean())
s20 = panel.groupby('symbol')['Close'].transform(lambda s: s.rolling(20, min_periods=5).std().fillna(1e-4))
z20 = (panel['Close'] - m20) / (s20 + 1e-6)

hl_range = (panel['Close'] - (panel['High'] + panel['Low'])/2.0) / (panel['High'] - panel['Low'] + 1e-4)
range_z = hl_range.rolling(5, min_periods=1).mean()

# Alpha core:
alpha = z20 * 0.5 + z10 * 0.3 + range_z * 0.2

# Market trend filter (Equal-weight universe close above 100d MA)
market_mean = panel.groupby('date')['Close'].mean()
market_ma = market_mean.rolling(50, min_periods=10).mean()
market_trend = (market_mean > market_ma).astype(float)
market_trend_dict = market_trend.to_dict()

# Test various enhancements:
df = panel.copy()
df['alpha'] = alpha
dates = sorted(df['date'].unique())[:-1]

for mode in ['standard', 'market_regime', 'vol_scaled', 'top_bottom_scaled']:
    daily_returns = []
    
    recent_vols = []
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
        
        # Scaling modifications:
        if mode == 'market_regime':
            trend = market_trend_dict.get(dt, 1.0)
            mult = 1.0 if trend == 1.0 else 0.5
            w = w * mult
        elif mode == 'top_bottom_scaled':
            # Concentrate on the top 10 and bottom 10 highest conviction
            rank = np.argsort(demeaned)
            top_bottom_mask = (rank < 8) | (rank >= len(rank) - 8)
            w = np.where(top_bottom_mask, demeaned, 0.0)
            if np.sum(np.abs(w)) > 0:
                w = w / np.sum(np.abs(w))
        elif mode == 'vol_scaled':
            # Inverse volatility per asset: w_i = (demeaned_i / vol_i) / sum
            asset_vols = s20.loc[day.index].values
            w = (demeaned / (asset_vols + 1e-4))
            if np.sum(np.abs(w)) > 0:
                w = w / np.sum(np.abs(w))
                
        ret = np.sum(w * day['fwd_ret'].values)
        daily_returns.append(ret)
        
    r = np.array(daily_returns)
    sharpe = (np.mean(r) / np.std(r)) * np.sqrt(252)
    cum = np.cumprod(1 + r)
    peaks = np.maximum.accumulate(cum)
    dd = np.max((peaks - cum) / peaks)
    ann_ret = (cum[-1] ** (252.0 / len(r))) - 1.0
    print(f"[{mode:<20}] Sharpe: {sharpe:.3f} | Max DD: {dd*100:.2f}% | AnnRet: {ann_ret*100:.2f}%")
