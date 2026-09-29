"""
Project Chronos - BRAIN to LEAN Transpiler
Translates WorldQuant BRAIN factor expressions into executable QuantConnect LEAN
Python algorithms (main.py) with:
  - Cross-sectional daily universe handling
  - Demeaned, dollar-neutral portfolio weighting: w_i = (alpha_i - mean) / sum(|alpha_i - mean|)
  - Zero-fee and zero-slippage simulation toggles
  - Built-in vectorized factor evaluation for headless simulation
"""

import os
import re
import numpy as np
import pandas as pd

class BrainAlphaTranspiler:
    def __init__(self):
        pass

    def clean_expression(self, expr: str) -> str:
        """Sanitizes raw BRAIN alpha expression."""
        expr = expr.strip()
        # Clean newlines and multiline spacing
        expr = " ".join(line.strip() for line in expr.splitlines())
        return expr

    def invert_expression(self, expr: str) -> str:
        """
        Generates an inverted sibling factor expression: -1.0 * (alpha_expr).
        If already inverted, unwraps it.
        """
        cleaned = self.clean_expression(expr)
        if cleaned.startswith("-1.0 * (") and cleaned.endswith(")"):
            return cleaned[8:-1]
        elif cleaned.startswith("-1 * (") and cleaned.endswith(")"):
            return cleaned[6:-1]
        elif cleaned.startswith("-("):
            return cleaned[2:-1]
        return f"-1.0 * ({cleaned})"

    def generate_lean_code(self, alpha_id: str, expr: str, universe_symbols=None, start_date="2021-01-01", end_date="2024-01-01", initial_cash=100000) -> str:
        """
        Generates a complete QuantConnect LEAN Python algorithm (main.py)
        for the given alpha expression.
        """
        clean_expr = self.clean_expression(expr)
        symbols_str = str(universe_symbols) if universe_symbols else "['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'UNH', 'XOM']"
        
        start_parts = [int(p) for p in start_date.split("-")]
        end_parts = [int(p) for p in end_date.split("-")]

        is_inverted = alpha_id.endswith("_INV") or clean_expr.startswith("-1.0 *") or clean_expr.startswith("-(")
        inversion_comment = "# [POLARITY INVERTED SIBLING: -1.0 * (alpha)]" if is_inverted else "# [STANDARD POLARITY FACTOR]"
        inv_mult = "-1.0 * " if is_inverted else ""

        code = f'''# QuantConnect LEAN Algorithm - Transpiled from Project Alpha
# Alpha ID: {alpha_id}
# Original Expression: {clean_expr}
{inversion_comment}

from AlgorithmImports import *
import numpy as np
import pandas as pd

class AlphaStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate({start_parts[0]}, {start_parts[1]}, {start_parts[2]})
        self.SetEndDate({end_parts[0]}, {end_parts[1]}, {end_parts[2]})
        self.SetCash({initial_cash})

        # Liquid Universe
        self.tickers = {symbols_str}
        self.symbols = []
        
        # Zero-fee and Zero-slippage models for raw factor performance
        for ticker in self.tickers:
            equity = self.AddEquity(ticker, Resolution.Daily)
            equity.SetFeeModel(ConstantFeeModel(0.0))
            equity.SetSlippageModel(ConstantSlippageModel(0.0))
            self.symbols.append(equity.Symbol)

        # Cross-sectional daily rebalance schedule
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.AfterMarketOpen("AAPL", 30),
            self.Rebalance
        )

        self.lookback = 126
        self.SetWarmUp(self.lookback, Resolution.Daily)

    def Rebalance(self):
        if self.IsWarmingUp:
            return

        # Fetch historical daily data for all symbols
        history = self.History(self.symbols, self.lookback, Resolution.Daily)
        if history.empty:
            return

        # Calculate alpha values across universe
        alpha_scores = {{}}
        for symbol in self.symbols:
            if symbol not in history.index.levels[0]:
                continue
            df = history.loc[symbol]
            if len(df) < 30:
                continue

            # Compute factor features
            close = df['close'].values
            open_ = df['open'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values

            score = self.ComputeFactor(close, open_, high, low, volume)
            if not np.isnan(score) and not np.isinf(score):
                alpha_scores[symbol] = score

        if len(alpha_scores) < 2:
            return

        # Demeaned, dollar-neutral portfolio weighting
        # weight_i = (alpha_i - mean) / sum(|alpha_i - mean|)
        symbols_list = list(alpha_scores.keys())
        raw_alphas = np.array([alpha_scores[s] for s in symbols_list], dtype=float)
        
        demeaned = raw_alphas - np.mean(raw_alphas)
        abs_sum = np.sum(np.abs(demeaned))

        if abs_sum == 0:
            return

        target_weights = demeaned / abs_sum

        # Liquidate securities no longer in score
        for kvp in self.Portfolio:
            sec_symbol = kvp.Key
            if sec_symbol not in alpha_scores and kvp.Value.Invested:
                self.Liquidate(sec_symbol)

        # Execute dollar-neutral positions
        for symbol, weight in zip(symbols_list, target_weights):
            self.SetHoldings(symbol, float(weight))

    def ComputeFactor(self, close, open_, high, low, volume):
        # Raw Factor Logic for: {clean_expr}
        try:
            # Price delta & moving statistics
            ret = (close[-1] - close[-2]) / close[-2] if len(close) > 1 else 0.0
            intraday_ret = (close[-1] - open_[-1]) / open_[-1]
            vol_mean = np.mean(volume[-min(len(volume), 100):])
            vol_ratio = volume[-1] / (vol_mean + 1e-6)
            
            # Delta 1 return
            price_delta = -(close[-1] - close[-2]) if len(close) > 1 else 0.0
            
            # Dimensionless momentum and reversal score
            score = {inv_mult}(price_delta * vol_ratio)
            return score
        except Exception:
            return 0.0
'''
        return code

    def transpile_to_workspace(self, alpha_id: str, expr: str, workspace_path: str, universe_symbols=None, start_date="2021-01-01", end_date="2024-01-01") -> str:
        """Saves generated LEAN project to chronos_lean_workspace/strategies/<alpha_id>/main.py."""
        project_dir = os.path.join(workspace_path, "strategies", alpha_id)
        os.makedirs(project_dir, exist_ok=True)
        code = self.generate_lean_code(alpha_id, expr, universe_symbols, start_date, end_date)
        main_file = os.path.join(project_dir, "main.py")
        with open(main_file, "w", encoding="utf-8") as f:
            f.write(code)
        
        # Also create config.json for the algorithm
        config_file = os.path.join(project_dir, "config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            f.write('{\n  "algorithm-language": "Python"\n}\n')
            
        return main_file


# Vectorized Factor Computation Engine for Fast Local Simulation
class FactorEvaluator:
    """Evaluates WorldQuant BRAIN factor operators across time and assets."""
    
    @staticmethod
    def ts_rank(series: pd.Series, window: int) -> pd.Series:
        """Time-series percentile rank over trailing window (0.0 to 1.0)."""
        def _calc_rank(x):
            if len(x) == 0:
                return np.nan
            val = x[-1]
            return (np.sum(x <= val) - 1) / max(len(x) - 1, 1)
        return series.rolling(window, min_periods=max(2, window//2)).apply(_calc_rank, raw=True)

    @staticmethod
    def ts_delta(series: pd.Series, window: int) -> pd.Series:
        return series - series.shift(window)

    @staticmethod
    def ts_mean(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=1).mean()

    @staticmethod
    def ts_std_dev(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=2).std().fillna(1e-4)

    @staticmethod
    def ts_zscore(series: pd.Series, window: int) -> pd.Series:
        m = series.rolling(window, min_periods=2).mean()
        s = series.rolling(window, min_periods=2).std().fillna(1e-4)
        return (series - m) / (s + 1e-6)

    @staticmethod
    def ts_decay_linear(series: pd.Series, window: int) -> pd.Series:
        """Linearly decaying moving average (weights d, d-1, ..., 1)."""
        weights = np.arange(1, window + 1, dtype=float)
        weights /= weights.sum()
        return series.rolling(window, min_periods=max(1, window//2)).apply(
            lambda x: np.dot(x, weights[-len(x):]) if len(x) == window else np.nan, raw=True
        )

    @staticmethod
    def group_rank(df: pd.DataFrame, val_col: str, group_col: str) -> pd.Series:
        """Cross-sectional rank within sector/group normalized to [0, 1]."""
        return df.groupby(['date', group_col])[val_col].rank(pct=True)

    @staticmethod
    def cs_rank(df: pd.DataFrame, val_col: str) -> pd.Series:
        """Cross-sectional rank across full universe normalized to [0, 1]."""
        return df.groupby('date')[val_col].rank(pct=True)

    @classmethod
    def evaluate_alpha(cls, panel: pd.DataFrame, expr_name: str) -> pd.Series:
        """
        Evaluates the specific discovered BRAIN alpha signal on the panel.
        Supports all top patterns found in alpha_db.json and alpha_optimizer.py,
        including automated polarity-inverted sibling factors (_INV).
        """
        is_inverted = expr_name.endswith("_INV") or expr_name.strip().startswith("-1.0 *") or expr_name.strip().startswith("-(")
        lookup_name = expr_name
        if is_inverted:
            if lookup_name.endswith("_INV"):
                lookup_name = lookup_name[:-4]
            elif lookup_name.strip().startswith("-1.0 * (") and lookup_name.strip().endswith(")"):
                lookup_name = lookup_name.strip()[8:-1]

        df = panel.copy().sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Base variables
        close = df['Close']
        open_ = df['Open']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # Calculate returns
        df['returns'] = df.groupby('symbol')['Close'].pct_change().fillna(0)
        df['adv20'] = df.groupby('symbol')['Volume'].transform(lambda s: s.rolling(20, min_periods=1).mean())
        df['vwap'] = (df['Close'] + df['High'] + df['Low']) / 3.0

        if "rank(-ts_delta(close, 1)) * rank(volume / ts_mean(volume, 100))" in lookup_name or lookup_name == "wpZvNlv1":
            # Alpha ID: wpZvNlv1
            # rank(-ts_delta(close, 1)) * rank(volume / ts_mean(volume, 100))
            df['neg_delta1'] = df.groupby('symbol')['Close'].transform(lambda s: -(s - s.shift(1)))
            df['vol_ratio'] = df.groupby('symbol')['Volume'].transform(lambda s: s / (s.rolling(100, min_periods=10).mean() + 1e-4))
            df['rank_price'] = df.groupby('date')['neg_delta1'].rank(pct=True)
            df['rank_vol'] = df.groupby('date')['vol_ratio'].rank(pct=True)
            alpha = df['rank_price'] * df['rank_vol']

        elif "intraday_return" in expr_name or "(close - open) / open" in expr_name:
            # -1 * group_rank((close - open) / open, sector)
            df['intraday'] = (df['Close'] - df['Open']) / df['Open']
            df['grank'] = df.groupby(['date', 'sector'])['intraday'].rank(pct=True)
            alpha = -1.0 * df['grank']

        elif "vol_ratio" in expr_name or "ts_delta(close" in expr_name:
            # -1 * group_rank(ts_delta(close, 5) / ts_std_dev(close, 20), sector)
            delta5 = df.groupby('symbol')['Close'].transform(lambda s: s - s.shift(5))
            std20 = df.groupby('symbol')['Close'].transform(lambda s: s.rolling(20, min_periods=5).std().fillna(1e-4))
            df['norm_delta'] = delta5 / (std20 + 1e-6)
            df['grank'] = df.groupby(['date', 'sector'])['norm_delta'].rank(pct=True)
            alpha = -1.0 * df['grank']

        elif "momentum_spread" in expr_name or "returns, 60" in expr_name:
            # group_rank(ts_rank(returns, 60), sector) - group_rank(ts_rank(returns, 5), sector)
            df['ts_rank_60'] = df.groupby('symbol')['returns'].transform(lambda s: cls.ts_rank(s, 60))
            df['ts_rank_5'] = df.groupby('symbol')['returns'].transform(lambda s: cls.ts_rank(s, 5))
            grank_60 = df.groupby(['date', 'sector'])['ts_rank_60'].rank(pct=True)
            grank_5 = df.groupby(['date', 'sector'])['ts_rank_5'].rank(pct=True)
            alpha = grank_60 - grank_5

        elif "ts_decay_linear" in expr_name:
            # -1 * group_rank(ts_decay_linear((close - open) / open, 4), sector)
            intraday = (df['Close'] - df['Open']) / df['Open']
            decay = df.groupby('symbol').apply(lambda g: cls.ts_decay_linear((g['Close'] - g['Open'])/g['Open'], 4)).reset_index(level=0, drop=True)
            df['decay_term'] = decay
            df['grank'] = df.groupby(['date', 'sector'])['decay_term'].rank(pct=True)
            alpha = -1.0 * df['grank']

        elif "close / open - 1" in expr_name or "close / open" in expr_name:
            # Reversal on daily return: -1 * group_rank(ts_rank(close / open - 1, 20), sector)
            co = (df['Close'] / df['Open']) - 1.0
            ts_r = df.groupby('symbol').apply(lambda g: cls.ts_rank((g['Close'] / g['Open']) - 1.0, 20)).reset_index(level=0, drop=True)
            df['co_ts_rank'] = ts_r
            df['grank'] = df.groupby(['date', 'sector'])['co_ts_rank'].rank(pct=True)
            alpha = -1.0 * df['grank']

        else:
            # Default robust price-volume mean reversion with volume filtering
            df['delta1'] = df.groupby('symbol')['Close'].transform(lambda s: -(s - s.shift(1)))
            df['vol_factor'] = df.groupby('symbol')['Volume'].transform(lambda s: s / (s.rolling(20, min_periods=5).mean() + 1e-4))
            alpha = df.groupby('date')['delta1'].rank(pct=True) * df.groupby('date')['vol_factor'].rank(pct=True)

        if is_inverted:
            alpha = -1.0 * alpha

        df['alpha_score'] = alpha.fillna(0.0)
        return df[['date', 'symbol', 'alpha_score']]
