import pandas as pd
import numpy as np
from runner import LeanRunner
from transpiler import FactorEvaluator

runner = LeanRunner()

alphas_to_test = [
    ("Alpha_Momentum_Spread", "momentum_spread"),
    ("Alpha_Intraday_Rev", "intraday_return"),
    ("Alpha_Vol_Ratio", "vol_ratio"),
    ("Alpha_Decay_Linear", "ts_decay_linear"),
    ("Alpha_wpZvNlv1", "wpZvNlv1")
]

results = []
for aid, expr in alphas_to_test:
    res = runner.run_backtest(aid, expr, start_date="2021-01-01", end_date="2024-01-01")
    results.append(res)

df_res = pd.DataFrame(results)
print("\n" + "="*80)
print("ALPHA CANDIDATE EVALUATION SUMMARY")
print("="*80)
print(df_res[['AlphaId', 'SharpeRatio', 'SortinoRatio', 'MaxDrawdown', 'AnnualizedReturn', 'AverageTurnover']])
