import os
import json

workspace_dir = r"c:\Users\hardi\Downloads\Project Chronos\chronos_lean_workspace"
os.makedirs(os.path.join(workspace_dir, "data", "equity", "usa", "daily"), exist_ok=True)
os.makedirs(os.path.join(workspace_dir, "data", "equity", "usa", "map_files"), exist_ok=True)
os.makedirs(os.path.join(workspace_dir, "data", "equity", "usa", "factor_files"), exist_ok=True)
os.makedirs(os.path.join(workspace_dir, "strategies"), exist_ok=True)
os.makedirs(os.path.join(workspace_dir, "results"), exist_ok=True)

lean_config = {
    "data-folder": "data",
    "default-language": "python",
    "job-organization-id": "chronos-local-org",
    "algorithm-type-name": "BasicTemplateAlgorithm",
    "algorithm-language": "Python",
    "algorithm-location": "main.py",
    "data-provider": "QuantConnect.Lean.Engine.DataFeeds.DefaultDataProvider",
    "map-file-provider": "QuantConnect.Data.Auxiliary.LocalDiskMapFileProvider",
    "factor-file-provider": "QuantConnect.Data.Auxiliary.LocalDiskFactorFileProvider",
    "environments": {
        "backtesting": {
            "live-mode": False,
            "setup-handler": "QuantConnect.Lean.Engine.Setup.ConsoleSetupHandler",
            "result-handler": "QuantConnect.Lean.Engine.Results.BacktestingResultHandler",
            "data-feed-handler": "QuantConnect.Lean.Engine.DataFeeds.FileSystemDataFeed",
            "real-time-handler": "QuantConnect.Lean.Engine.RealTime.BacktestingRealTimeHandler",
            "transaction-handler": "QuantConnect.Lean.Engine.TransactionHandlers.BacktestingTransactionHandler"
        }
    },
    "parameters": {
        "start-date": "2021-01-01",
        "end-date": "2024-01-01",
        "initial-cash": 100000.0,
        "fee-model": "zero",
        "slippage-model": "zero"
    }
}

config_path = os.path.join(workspace_dir, "lean.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(lean_config, f, indent=2)

print(f"Scaffolded LEAN workspace successfully at: {workspace_dir}")
print(f"Created config: {config_path}")
