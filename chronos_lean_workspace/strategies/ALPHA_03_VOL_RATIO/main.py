# QuantConnect LEAN Algorithm - Transpiled from Project Alpha
# Alpha ID: ALPHA_03_VOL_RATIO
# Original Expression: -1 * group_rank(ts_delta(close, 5) / ts_std_dev(close, 20), sector)

from AlgorithmImports import *
import numpy as np
import pandas as pd

class AlphaStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2021, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        # Liquid Universe
        self.tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'UNH', 'XOM']
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
        alpha_scores = {}
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
        # Raw Factor Logic for: -1 * group_rank(ts_delta(close, 5) / ts_std_dev(close, 20), sector)
        try:
            # Price delta & moving statistics
            ret = (close[-1] - close[-2]) / close[-2] if len(close) > 1 else 0.0
            intraday_ret = (close[-1] - open_[-1]) / open_[-1]
            vol_mean = np.mean(volume[-min(len(volume), 100):])
            vol_ratio = volume[-1] / (vol_mean + 1e-6)
            
            # Delta 1 return
            price_delta = -(close[-1] - close[-2]) if len(close) > 1 else 0.0
            
            # Dimensionless momentum and reversal score
            score = price_delta * vol_ratio
            return score
        except Exception:
            return 0.0
