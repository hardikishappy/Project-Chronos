import json

with open(r"c:\Users\hardi\Downloads\Project Chronos\data\accepted_alphas.json", "r") as f:
    alphas = json.load(f)

for i, (aid, a) in enumerate(alphas.items(), 1):
    print(f"{i:2d}. [{aid}] Sharpe={a['original_sharpe']} | Ret={a['original_returns']} | DD={a['original_drawdown']}")
    print(f"    {a['expression']}")
