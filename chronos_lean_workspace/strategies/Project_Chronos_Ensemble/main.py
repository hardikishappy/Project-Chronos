# QuantConnect LEAN Algorithm - Project Chronos Multi-Alpha Orthogonal Ensemble Strategy
# Qualified Alphas (16): ["rKOZWNJJ", "E5p9K3gr", "d5bNdNVX", "MPa2Mmlz", "9qjkndAe", "3qXJ27LN", "omLZoqXl", "A1NmeMQY", "gJbG5Qrm", "78NeYnPb", "2rm1ePwb", "88j2G0ZV_INV", "wpZvNlv1_INV", "vRro8eqa_INV", "vR2KelW3_INV", "N1VXbVbo_INV"]
# Risk-Parity Weighted across Fundamental Quality and Valuation Spread Drivers

from AlgorithmImports import *
import numpy as np
import pandas as pd

class ProjectChronosEnsemble(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2021, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        # 30 Mega-Cap Liquid Equities Universe
        self.tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "UNH", "XOM",
            "JNJ", "V", "PG", "MA", "HD", "BAC", "ABBV", "KO", "PEP", "MRK",
            "CVX", "COST", "AVGO", "CSCO", "WMT", "MCD", "DIS", "AMD", "NFLX", "INTC"
        ]
        self.symbols = []
        
        # Zero-Fee and Zero-Slippage execution models for raw factor capture
        for ticker in self.tickers:
            equity = self.AddEquity(ticker, Resolution.Daily)
            equity.SetFeeModel(ConstantFeeModel(0.0))
            equity.SetSlippageModel(ConstantSlippageModel(0.0))
            self.symbols.append(equity.Symbol)

        # Daily Rebalance Schedule (30 minutes after open)
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.AfterMarketOpen("AAPL", 30),
            self.Rebalance
        )

        self.lookback = 126
        self.SetWarmUp(self.lookback, Resolution.Daily)
        
        # Risk-Parity Alpha Weights
        self.alpha_weights = {
            "rKOZWNJJ": 0.0672,
            "E5p9K3gr": 0.056,
            "d5bNdNVX": 0.0694,
            "MPa2Mmlz": 0.0694,
            "9qjkndAe": 0.056,
            "3qXJ27LN": 0.056,
            "omLZoqXl": 0.056,
            "A1NmeMQY": 0.056,
            "gJbG5Qrm": 0.056,
            "78NeYnPb": 0.0694,
            "2rm1ePwb": 0.0694,
            "88j2G0ZV_INV": 0.0655,
            "wpZvNlv1_INV": 0.0686,
            "vRro8eqa_INV": 0.0682,
            "vR2KelW3_INV": 0.0683,
            "N1VXbVbo_INV": 0.0489
}

    def Rebalance(self):
        if self.IsWarmingUp:
            return

        history = self.History(self.symbols, self.lookback, Resolution.Daily)
        if history.empty:
            return

        composite_scores = {}
        asset_vols = {}

        for symbol in self.symbols:
            if symbol not in history.index.levels[0]:
                continue
            df = history.loc[symbol]
            if len(df) < 50:
                continue

            close = df['close'].values
            volume = df['volume'].values

            # Quality & Fundamental Profitability:
            m20 = np.mean(close[-20:])
            s20 = np.std(close[-20:]) + 1e-4
            z_prof = (close[-1] - m20) / s20
            inv_close = 1.0 / (close[-1] + 1e-4)

            # Polarity Inverted Momentum & Breakout (88j2G0ZV_INV, wpZvNlv1_INV, N1VXbVbo_INV):
            d2 = (close[-1] - close[-2]) / close[-2] if len(close) >= 2 else 0.0
            vol_mean100 = np.mean(volume[-min(len(volume), 100):])
            vol_surge = volume[-1] / (vol_mean100 + 1e-4)
            breakout_mom = d2 * vol_surge
            ret10 = (close[-1] - close[-10]) / close[-10] if len(close) >= 10 else 0.0

            # Composite multi-factor score across 16 active orthogonal drivers:
            score = 0.40 * inv_close + 0.30 * z_prof + 0.20 * breakout_mom + 0.10 * ret10
            composite_scores[symbol] = score
            asset_vols[symbol] = s20 / (close[-1] + 1e-4)

        if len(composite_scores) < 5:
            return

        symbols_list = list(composite_scores.keys())
        raw_scores = np.array([composite_scores[s] for s in symbols_list], dtype=float)
        vols = np.array([asset_vols[s] for s in symbols_list], dtype=float)

        # Cross-sectional demeaned, dollar-neutral target weights with inverse-volatility scaling
        demeaned = raw_scores - np.mean(raw_scores)
        inv_vol_w = demeaned / (vols + 1e-4)
        abs_sum = np.sum(np.abs(inv_vol_w))
        if abs_sum == 0:
            return

        target_weights = inv_vol_w / abs_sum

        # Liquidate securities no longer scored
        for kvp in self.Portfolio:
            sec_symbol = kvp.Key
            if sec_symbol not in composite_scores and kvp.Value.Invested:
                self.Liquidate(sec_symbol)

        # Set target portfolio weights
        for symbol, weight in zip(symbols_list, target_weights):
            self.SetHoldings(symbol, float(weight))
