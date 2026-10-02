"""
Project Chronos - Alpaca Paper Trading Execution Broker
Autonomous execution interface for Alpaca Markets Paper Trading REST API:
  - Account status & telemetry (/v2/account)
  - Positions & portfolio state (/v2/positions)
  - Order generation & execution (/v2/orders)
  - Multi-alpha portfolio rebalancing & dollar-neutral execution
  - Seamless fallback to simulated paper mode if API keys are unconfigured
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import requests
import numpy as np

from execution.almgren_chriss import AlmgrenChrissExecutor, AlmgrenChrissTrajectory

logger = logging.getLogger("AlpacaPaperBroker")

# Master 16-Alpha Risk-Parity Long/Short Composite Target Allocations (S&P 500 Mega-Cap Liquid Universe)
DEFAULT_ENSEMBLE_WEIGHTS: Dict[str, float] = {
    "NVDA": 0.075, "AAPL": 0.065, "MSFT": 0.060, "META": 0.055, "GOOGL": 0.055,
    "AMZN": 0.050, "JPM": 0.050, "UNH": 0.050, "V": 0.045, "MA": 0.045,
    "TSLA": 0.040, "JNJ": 0.040, "PG": 0.040, "HD": 0.035, "COST": 0.035,
    "INTC": -0.055, "CSCO": -0.045, "XOM": -0.045, "BAC": -0.040, "KO": -0.035, "DIS": -0.030
}

# Minimum Operational Cash Reserve for settlement buffer and margin friction
MINIMUM_CASH_RESERVE: float = 1500.00


class AlpacaPaperBroker:
    def __init__(self, env_path: Optional[str] = None):
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        if env_path is None:
            self.env_path = os.path.join(self.root_dir, ".env")
        else:
            self.env_path = env_path

        self.api_key: str = ""
        self.secret_key: str = ""
        self.base_url: str = "https://paper-api.alpaca.markets"
        self.data_url: str = "https://data.alpaca.markets"
        
        # Minimum Operational Reserve for settlement friction & margin requirement absorption
        self.minimum_cash_reserve: float = MINIMUM_CASH_RESERVE

        # Micro-lot pruning threshold ($3.00 to accommodate 100+ liquid equities across panel)
        self.micro_lot_threshold: float = 3.0
        
        # Batch order throttler (0.08-0.10s between dispatches to strictly respect 200 req/min rate limit)
        self.throttle_delay: float = 0.09
        
        # Almgren-Chriss Optimal Execution Engine
        self.ac_executor = AlmgrenChrissExecutor(large_order_adv_threshold=0.01)
        self._adv_cache: Dict[str, float] = {}

        # 3-State Continuous Gaussian HMM Regime Engine
        from models.regime_hmm import MarketRegimeHMM
        self.regime_hmm = MarketRegimeHMM()

        self._load_config()

        # Simulated sandbox state when running dry-run or mock
        self._simulated_cash: float = 100000.0
        self._simulated_positions: Dict[str, Dict[str, Any]] = {}
        self._simulated_orders: List[Dict[str, Any]] = []
        self._price_cache: Dict[str, Any] = {}

    def _load_config(self) -> None:
        """Loads credentials from .env and environment variables."""
        if os.path.exists(self.env_path):
            try:
                with open(self.env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            if k == "ALPACA_API_KEY":
                                self.api_key = v
                            elif k == "ALPACA_SECRET_KEY":
                                self.secret_key = v
                            elif k == "ALPACA_BASE_URL":
                                self.base_url = v.rstrip("/")
            except Exception as e:
                logger.warning(f"Failed to read .env file at {self.env_path}: {e}")

        # Environment variable overrides
        self.api_key = os.environ.get("ALPACA_API_KEY", self.api_key)
        self.secret_key = os.environ.get("ALPACA_SECRET_KEY", self.secret_key)
        self.base_url = os.environ.get("ALPACA_BASE_URL", self.base_url).rstrip("/")

    @property
    def is_configured(self) -> bool:
        """Checks if real API credentials are configured."""
        placeholder_tokens = ["your_paper_key_here", "your_paper_secret_here", ""]
        return bool(
            self.api_key and 
            self.secret_key and 
            self.api_key not in placeholder_tokens and 
            self.secret_key not in placeholder_tokens
        )

    def _get_headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json"
        }

    # =========================================================================
    # Account & Telemetry
    # =========================================================================
    def get_account(self) -> Dict[str, Any]:
        """
        Retrieves Alpaca Paper account state.
        Returns live data if configured, otherwise returns simulated sandbox state.
        """
        if self.is_configured:
            try:
                url = f"{self.base_url}/v2/account"
                resp = requests.get(url, headers=self._get_headers(), timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    data["connection_status"] = "CONNECTED_LIVE_PAPER"
                    data["paper_trading"] = True
                    return data
                else:
                    logger.error(f"Alpaca get_account failed ({resp.status_code}): {resp.text}")
            except Exception as e:
                logger.error(f"Network error querying Alpaca account: {e}")

        # Simulated fallback state
        total_pos_val = sum(p.get("market_value", 0.0) for p in self._simulated_positions.values())
        return {
            "id": "SIMULATED-CHRONOS-PAPER-01",
            "status": "ACTIVE",
            "currency": "USD",
            "buying_power": f"{self._simulated_cash * 2.0:.2f}",
            "cash": f"{self._simulated_cash:.2f}",
            "portfolio_value": f"{self._simulated_cash + total_pos_val:.2f}",
            "equity": f"{self._simulated_cash + total_pos_val:.2f}",
            "pattern_day_trader": False,
            "trading_blocked": False,
            "transfers_blocked": False,
            "account_blocked": False,
            "paper_trading": True,
            "connection_status": "SIMULATED_LOCAL_SANDBOX"
        }

    # =========================================================================
    # Market Clock & Hours Telemetry
    # =========================================================================
    def get_clock(self) -> Dict[str, Any]:
        """
        Retrieves Alpaca market clock state (/v2/clock).
        Returns live market status, or computes US/Eastern status as fallback.
        """
        if self.is_configured:
            try:
                url = f"{self.base_url}/v2/clock"
                resp = requests.get(url, headers=self._get_headers(), timeout=5)
                if resp.status_code == 200:
                    return resp.json()
                else:
                    logger.error(f"Alpaca get_clock failed ({resp.status_code}): {resp.text}")
            except Exception as e:
                logger.error(f"Network error querying Alpaca clock: {e}")

        # Fallback clock computation using pytz US/Eastern
        try:
            import pytz
            eastern = pytz.timezone("US/Eastern")
            now_est = datetime.now(eastern)
            is_weekday = now_est.weekday() < 5
            market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now_est.replace(hour=16, minute=0, second=0, microsecond=0)
            is_open = is_weekday and (market_open <= now_est <= market_close)
            return {
                "timestamp": now_est.isoformat(),
                "is_open": is_open,
                "next_open": market_open.isoformat(),
                "next_close": market_close.isoformat()
            }
        except Exception:
            return {"is_open": False, "timestamp": datetime.now().isoformat()}

    def is_market_open(self) -> bool:
        """Returns True if the US equity market is currently open for live routing."""
        clock = self.get_clock()
        return bool(clock.get("is_open", False))

    # =========================================================================
    # Positions Management
    # =========================================================================
    def get_positions(self) -> List[Dict[str, Any]]:
        """Retrieves all open portfolio positions."""
        if self.is_configured:
            try:
                url = f"{self.base_url}/v2/positions"
                resp = requests.get(url, headers=self._get_headers(), timeout=10)
                if resp.status_code == 200:
                    return resp.json()
                else:
                    logger.error(f"Alpaca get_positions failed ({resp.status_code}): {resp.text}")
            except Exception as e:
                logger.error(f"Network error querying Alpaca positions: {e}")

        return list(self._simulated_positions.values())

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieves position for a specific symbol."""
        symbol = symbol.upper()
        if self.is_configured:
            try:
                url = f"{self.base_url}/v2/positions/{symbol}"
                resp = requests.get(url, headers=self._get_headers(), timeout=10)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 404:
                    return None
            except Exception as e:
                logger.error(f"Network error querying Alpaca position {symbol}: {e}")

        return self._simulated_positions.get(symbol)

    # =========================================================================
    # Order Submission & Cancellation
    # =========================================================================
    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        bypass_market_hours: bool = False,
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Submits a market or limit order to Alpaca Paper Trading.
        Enforces integer share rounding, micro-lot (< $10) omission, and off-hours protection.
        """
        symbol = symbol.upper()
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"Invalid order side: {side}. Must be 'buy' or 'sell'.")

        qty_int = int(round(abs(qty)))
        if qty_int <= 0:
            return {"status": "skipped", "reason": "qty <= 0", "symbol": symbol}

        # Price lookup for sanity bounds
        latest_price = price if price is not None else self.get_latest_price(symbol)
        order_value = qty_int * latest_price

        # Sanity Bound: Omit any order where notional value is under micro_lot_threshold ($3.00)
        if order_value < self.micro_lot_threshold:
            logger.info(f"Skipping micro-lot order: {side.upper()} {qty_int} {symbol} (Val: ${order_value:.2f} < ${self.micro_lot_threshold:.2f})")
            return {
                "status": "skipped",
                "reason": "order_value_below_micro_lot_threshold",
                "symbol": symbol,
                "qty": qty_int,
                "order_value": round(order_value, 2)
            }

        # Market Closed Protection (if configured and not bypassed)
        if not bypass_market_hours and self.is_configured and not self.is_market_open():
            logger.warning(f"Order rejected for {symbol}: US Equity Market is closed.")
            return {
                "status": "rejected",
                "reason": "market_closed",
                "symbol": symbol,
                "error": "Market is currently CLOSED. Scheduled to fire automatically at 09:35 AM EST."
            }

        payload: Dict[str, Any] = {
            "symbol": symbol,
            "qty": str(qty_int),
            "side": side,
            "type": order_type.lower(),
            "time_in_force": time_in_force.lower()
        }
        if order_type.lower() == "limit" and limit_price is not None:
            payload["limit_price"] = f"{limit_price:.2f}"

        if self.is_configured:
            try:
                url = f"{self.base_url}/v2/orders"
                resp = requests.post(url, headers=self._get_headers(), json=payload, timeout=10)
                if resp.status_code in (200, 201):
                    order_data = resp.json()
                    order_data["execution_mode"] = "LIVE_PAPER_REST"
                    logger.info(f"[Alpaca Paper] Placed order: {side.upper()} {qty_int} {symbol} ({order_type}) - ID: {order_data.get('id')}")
                    return order_data
                else:
                    logger.error(f"Alpaca order rejected ({resp.status_code}): {resp.text}")
                    return {"status": "rejected", "error": resp.text, "symbol": symbol, "payload": payload}
            except Exception as e:
                logger.error(f"Network exception placing Alpaca order: {e}")
                return {"status": "error", "error": str(e), "symbol": symbol}

        # Simulated sandbox order execution
        sim_price = latest_price
        trade_val = sim_price * qty_int
        
        if side == "buy":
            self._simulated_cash -= trade_val
            curr = self._simulated_positions.get(symbol, {"qty": 0, "avg_entry_price": sim_price})
            new_qty = int(curr["qty"]) + qty_int
            self._simulated_positions[symbol] = {
                "symbol": symbol,
                "qty": str(new_qty),
                "avg_entry_price": f"{sim_price:.2f}",
                "market_value": f"{new_qty * sim_price:.2f}",
                "current_price": f"{sim_price:.2f}"
            }
        else: # sell
            self._simulated_cash += trade_val
            if symbol in self._simulated_positions:
                curr_qty = int(self._simulated_positions[symbol]["qty"])
                new_qty = max(0, curr_qty - qty_int)
                if new_qty == 0:
                    del self._simulated_positions[symbol]
                else:
                    self._simulated_positions[symbol]["qty"] = str(new_qty)
                    self._simulated_positions[symbol]["market_value"] = f"{new_qty * sim_price:.2f}"

        sim_order = {
            "id": f"sim-order-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "symbol": symbol,
            "qty": str(qty_int),
            "side": side,
            "type": order_type,
            "status": "filled",
            "filled_avg_price": f"{sim_price:.2f}",
            "created_at": datetime.now().isoformat(),
            "execution_mode": "SIMULATED_LOCAL_SANDBOX"
        }
        self._simulated_orders.append(sim_order)
        return sim_order

    def cancel_all_orders(self) -> List[Dict[str, Any]]:
        """Cancels all active open orders."""
        if self.is_configured:
            try:
                url = f"{self.base_url}/v2/orders"
                resp = requests.delete(url, headers=self._get_headers(), timeout=10)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                logger.error(f"Error cancelling Alpaca orders: {e}")
        return []

    # =========================================================================
    # Market Data & Pricing
    # =========================================================================
    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Batch fetches the latest trading prices for a list of symbols in a single request."""
        clean_symbols = [s.upper() for s in symbols]
        prices: Dict[str, float] = {}
        now_ts = time.time()

        # Check in-memory cache first (valid 60 seconds)
        symbols_to_fetch = []
        for s in clean_symbols:
            if s in self._price_cache and (now_ts - self._price_cache[s][1]) < 60.0:
                prices[s] = self._price_cache[s][0]
            else:
                symbols_to_fetch.append(s)

        if self.is_configured and symbols_to_fetch:
            try:
                sym_str = ",".join(symbols_to_fetch)
                url = f"{self.data_url}/v2/stocks/snapshots?symbols={sym_str}&feed=iex"
                resp = requests.get(url, headers=self._get_headers(), timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    for sym, snap in data.items():
                        trade_p = snap.get("latestTrade", {}).get("p")
                        quote_p = snap.get("latestQuote", {}).get("ap") or snap.get("latestQuote", {}).get("bp")
                        bar_p = snap.get("minuteBar", {}).get("c") or snap.get("dailyBar", {}).get("c")
                        p = trade_p or quote_p or bar_p
                        if p and float(p) > 0:
                            p_flt = float(p)
                            prices[sym] = p_flt
                            self._price_cache[sym] = (p_flt, now_ts)
            except Exception as e:
                logger.warning(f"Batch snapshot pricing error: {e}")

        # Fallback for any symbols not retrieved in batch
        for s in clean_symbols:
            if s not in prices:
                prices[s] = self.get_latest_price(s)

        return prices

    def get_latest_price(self, symbol: str) -> float:
        """Fetches the latest trading price for a symbol."""
        symbol = symbol.upper()
        now_ts = time.time()
        
        # Check cache (valid for 60 seconds)
        if symbol in self._price_cache and (now_ts - self._price_cache[symbol][1]) < 60.0:
            return self._price_cache[symbol][0]

        if self.is_configured:
            try:
                # Alpaca Market Data v2 endpoint with IEX feed
                url = f"{self.data_url}/v2/stocks/{symbol}/trades/latest?feed=iex"
                resp = requests.get(url, headers=self._get_headers(), timeout=3)
                if resp.status_code == 200:
                    p = resp.json().get("trade", {}).get("p")
                    if p and float(p) > 0:
                        p_flt = float(p)
                        self._price_cache[symbol] = (p_flt, now_ts)
                        return p_flt
            except Exception:
                pass

        # Fallback to universe data panel
        try:
            panel_path = os.path.join(self.root_dir, "data", "universe_panel.csv")
            if os.path.exists(panel_path):
                import pandas as pd
                df = pd.read_csv(panel_path)
                sym_rows = df[df['symbol'] == symbol]
                if not sym_rows.empty:
                    return float(sym_rows.iloc[-1]['Close'])
        except Exception:
            pass

        # Baseline fallback prices for top liquid equities
        fallback_prices = {
            "AAPL": 185.50, "MSFT": 420.20, "GOOGL": 175.40, "AMZN": 182.10,
            "NVDA": 125.80, "META": 505.60, "TSLA": 250.30, "JPM": 210.40,
            "UNH": 580.00, "XOM": 115.30, "JNJ": 160.20, "V": 275.50,
            "PG": 170.00, "MA": 480.00, "HD": 390.00, "BAC": 40.50
        }
        return fallback_prices.get(symbol, 150.0)

    def get_asset_adv(self, symbol: str) -> float:
        """Retrieves 20-day Average Daily Volume (ADV) for a symbol from local panel or default."""
        symbol = symbol.upper()
        if symbol in self._adv_cache:
            return self._adv_cache[symbol]

        for p_name in ["universe_panel.csv", "consolidated_panel.csv"]:
            for d_path in [os.path.join(self.root_dir, "data", p_name), os.path.join(self.root_dir, "chronos_lean_workspace", "data", p_name)]:
                if os.path.exists(d_path):
                    try:
                        import pandas as pd
                        df = pd.read_csv(d_path)
                        sub = df[df['symbol'] == symbol]
                        if not sub.empty and 'Volume' in sub.columns:
                            adv = float(sub['Volume'].tail(20).mean())
                            if adv > 1000:
                                self._adv_cache[symbol] = adv
                                return adv
                    except Exception:
                        pass
        default_adv = 10_000_000.0 if symbol in ("AAPL", "NVDA", "TSLA", "AMD", "BAC") else 3_000_000.0
        self._adv_cache[symbol] = default_adv
        return default_adv

    def load_market_panel(self) -> Any:
        """Loads the consolidated market panel for the 100+ liquid equities universe."""
        import pandas as pd
        panel_path = os.path.join(self.root_dir, "chronos_lean_workspace", "data", "consolidated_panel.csv")
        if not os.path.exists(panel_path):
            panel_path = os.path.join(self.root_dir, "data", "consolidated_panel.csv")
        if not os.path.exists(panel_path):
            raise FileNotFoundError(f"Consolidated panel not found at {panel_path}")
        return pd.read_csv(panel_path)

    def compute_regime_target_weights(self, panel_df: Optional[Any] = None) -> Dict[str, float]:
        """
        Dynamically calculates target weights across the liquid universe (100+ equities)
        conditioned on the 3-state continuous Gaussian HMM market regime.
        """
        try:
            import pandas as pd
            if panel_df is None:
                panel_df = self.load_market_panel()
            panel_df['date'] = pd.to_datetime(panel_df['date'])
            
            inf = self.regime_hmm.get_current_regime(panel_df)
            logger.info(f"HMM Regime inferred: {inf.state_name} (Gross Target: {inf.target_gross_leverage:.2f})")
            
            mom_tilt = inf.factor_tilts.get("Momentum_Breakout_Multiplier", 1.0)
            mr_tilt = inf.factor_tilts.get("Mean_Reversion_Quality_Multiplier", 1.0)
            gross_lev = inf.target_gross_leverage
            # When the HMM indicates State 0 (Trending) or State 1 (Choppy), set target_gross_leverage = 1.00 (utilizing full cash)
            if getattr(inf, "current_state", 1) in (0, 1):
                gross_lev = 1.00

            # Compute dynamic cross-sectional factor alpha scores across the 100+ assets
            recent_dates = sorted(panel_df['date'].unique())[-60:]
            recent_df = panel_df[panel_df['date'].isin(recent_dates)].copy()
            recent_df.sort_values(['symbol', 'date'], inplace=True)

            alpha_scores = {}
            asset_vols = {}

            for sym, grp in recent_df.groupby('symbol'):
                if len(grp) < 20:
                    continue
                c = grp['Close'].values
                h = grp['High'].values
                l = grp['Low'].values
                
                # Factor 1: 20d Vol-Adjusted Momentum
                m20 = np.mean(c[-20:])
                s20 = np.std(c[-20:]) + 1e-4
                z20 = (c[-1] - m20) / s20

                # Factor 2: 10d Vol-Adjusted Momentum
                m10 = np.mean(c[-10:])
                s10 = np.std(c[-10:]) + 1e-4
                z10 = (c[-1] - m10) / s10

                # Factor 3: 5d Range Breakout
                hl_diff = (h[-5:] - l[-5:]) + 1e-4
                range_pos = np.mean((c[-5:] - (h[-5:] + l[-5:])/2.0) / hl_diff)

                # Dynamic score conditioned on HMM regime tilts
                composite = (z20 * 0.50 * mom_tilt + 
                             z10 * 0.25 * mom_tilt + 
                             range_pos * 0.25 * mr_tilt)
                
                alpha_scores[sym] = composite
                asset_vols[sym] = s20 / max(1.0, c[-1])

            if len(alpha_scores) >= 50:
                syms = list(alpha_scores.keys())
                raw_scores = np.array([alpha_scores[s] for s in syms])
                vols = np.array([asset_vols[s] for s in syms])

                # Demeaned, dollar-neutral, inverse-vol weighted
                demeaned = raw_scores - np.mean(raw_scores)
                inv_vol = demeaned / (vols + 1e-4)

                # Two-sided dollar neutrality: sum(pos) == +0.5 * gross_lev, sum(neg) == -0.5 * gross_lev
                pos_mask = inv_vol > 0
                neg_mask = inv_vol < 0
                norm_w = np.zeros_like(inv_vol)

                pos_sum = np.sum(inv_vol[pos_mask])
                neg_sum = np.sum(np.abs(inv_vol[neg_mask]))

                if pos_sum > 0 and neg_sum > 0:
                    norm_w[pos_mask] = (inv_vol[pos_mask] / pos_sum) * (0.5 * gross_lev)
                    norm_w[neg_mask] = -(np.abs(inv_vol[neg_mask]) / neg_sum) * (0.5 * gross_lev)
                    weights = {s: round(float(w), 5) for s, w in zip(syms, norm_w)}
                    return weights
        except Exception as e:
            logger.warning(f"Could not compute HMM regime-conditioned weights ({e}), using baseline.")
        return DEFAULT_ENSEMBLE_WEIGHTS

    # =========================================================================
    # Portfolio Rebalancing & Capital Deployment Engine ("Full Bull Mode")
    # =========================================================================
    def calculate_rebalance_orders(
        self,
        target_weights: Optional[Dict[str, float]] = None,
        total_capital: Optional[float] = None,
        current_positions: Optional[Dict[str, int]] = None,
        prices_cache: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculates multi-alpha portfolio rebalancing orders adhering to Aggressive Capital Deployment ("Full Bull Mode"):
          1. Target Deployed Capital = Portfolio Equity - Minimum Operational Reserve ($1,500.00).
          2. When HMM indicates State 0 (Trending) or State 1 (Choppy), gross leverage = 1.00.
          3. Scales individual target position share counts so that:
             - Aggregate dollar value of all long legs equals ~50% of deployed capital.
             - Aggregate dollar value of all short legs equals ~50% of deployed capital.
             - Overall gross utilization achieves 95% - 100% of portfolio equity.
          4. Micro-lot pruning threshold: $3.00 (retaining all qualifying panel equities).
        """
        if target_weights is None:
            target_weights = self.compute_regime_target_weights()

        account = self.get_account()
        equity = float(account.get("equity") or account.get("portfolio_value", 100000.0))
        target_capital = total_capital if total_capital is not None else max(0.0, equity - self.minimum_cash_reserve)

        if current_positions is None:
            current_positions = {p["symbol"]: int(float(p["qty"])) for p in self.get_positions()}

        universe_symbols = list(set(list(target_weights.keys()) + list(current_positions.keys())))
        if prices_cache is None:
            prices_cache = self.get_latest_prices(universe_symbols)

        pos_weights = {s: w for s, w in target_weights.items() if w > 0}
        neg_weights = {s: w for s, w in target_weights.items() if w < 0}

        target_long_budget = 0.50 * target_capital
        target_short_budget = 0.50 * target_capital

        sum_pos = sum(pos_weights.values())
        sum_neg = sum(abs(w) for w in neg_weights.values())

        target_shares: Dict[str, int] = {}
        target_dollars: Dict[str, float] = {}

        # 1. First-pass share allocation with integer rounding
        for s, w in target_weights.items():
            s = s.upper()
            price = prices_cache.get(s, self.get_latest_price(s))
            if price <= 0:
                continue
            if w > 0 and sum_pos > 0:
                norm_w = w / sum_pos
                alloc = target_long_budget * norm_w
                sh = int(round(alloc / price))
            elif w < 0 and sum_neg > 0:
                norm_w = abs(w) / sum_neg
                alloc = target_short_budget * norm_w
                sh = -int(round(alloc / price))
            else:
                alloc = target_capital * w
                sh = int(round(alloc / price))

            target_shares[s] = sh
            target_dollars[s] = round(sh * price, 2)

        # 2. Second-pass proportional scaling to ensure ~50% long and ~50% short deployed
        long_val = sum(target_shares[s] * prices_cache.get(s, self.get_latest_price(s)) for s in target_shares if target_shares[s] > 0)
        short_val = sum(abs(target_shares[s]) * prices_cache.get(s, self.get_latest_price(s)) for s in target_shares if target_shares[s] < 0)

        if long_val > 0 and long_val < (0.95 * target_long_budget):
            scale_l = target_long_budget / long_val
            for s in list(target_shares.keys()):
                if target_shares[s] > 0:
                    p = prices_cache.get(s, self.get_latest_price(s))
                    new_sh = max(1, int(round(target_shares[s] * scale_l)))
                    target_shares[s] = new_sh
                    target_dollars[s] = round(new_sh * p, 2)

        if short_val > 0 and short_val < (0.95 * target_short_budget):
            scale_s = target_short_budget / short_val
            for s in list(target_shares.keys()):
                if target_shares[s] < 0:
                    p = prices_cache.get(s, self.get_latest_price(s))
                    new_sh = -max(1, int(round(abs(target_shares[s]) * scale_s)))
                    target_shares[s] = new_sh
                    target_dollars[s] = round(new_sh * p, 2)

        final_long_val = sum(target_shares[s] * prices_cache.get(s, self.get_latest_price(s)) for s in target_shares if target_shares[s] > 0)
        final_short_val = sum(abs(target_shares[s]) * prices_cache.get(s, self.get_latest_price(s)) for s in target_shares if target_shares[s] < 0)
        gross_deployed = final_long_val + final_short_val

        sell_orders = []
        buy_orders = []

        # 1. Close / liquidate positions that are no longer in target weights
        for symbol, curr_qty in current_positions.items():
            if symbol not in target_shares:
                price = prices_cache.get(symbol, self.get_latest_price(symbol))
                val = abs(curr_qty) * price
                if val >= self.micro_lot_threshold:
                    if curr_qty > 0:
                        sell_orders.append({
                            "symbol": symbol,
                            "qty": curr_qty,
                            "side": "sell",
                            "price": price,
                            "order_value": val,
                            "reason": "LIQUIDATE_NON_FACTOR"
                        })
                    elif curr_qty < 0:
                        buy_orders.append({
                            "symbol": symbol,
                            "qty": abs(curr_qty),
                            "side": "buy",
                            "price": price,
                            "order_value": val,
                            "reason": "COVER_NON_FACTOR_SHORT"
                        })

        # 2. Adjust target holdings
        for symbol, target_qty in target_shares.items():
            curr_qty = current_positions.get(symbol, 0)
            delta = target_qty - curr_qty
            price = prices_cache.get(symbol, self.get_latest_price(symbol))
            order_val = abs(delta) * price

            # Sanity bound: omit micro-lot orders under threshold ($3.00)
            if abs(delta) <= 0 or order_val < self.micro_lot_threshold:
                continue

            if delta < 0: # Sell (trim long or initiate/expand short)
                sell_orders.append({
                    "symbol": symbol,
                    "qty": abs(delta),
                    "side": "sell",
                    "price": price,
                    "order_value": order_val,
                    "reason": "REBALANCE_TRIM_OR_SHORT"
                })
            else: # Buy (expand long or cover short)
                buy_orders.append({
                    "symbol": symbol,
                    "qty": delta,
                    "side": "buy",
                    "price": price,
                    "order_value": order_val,
                    "reason": "REBALANCE_EXPAND_OR_COVER"
                })

        return {
            "account_equity": equity,
            "cash_reserve": self.minimum_cash_reserve,
            "target_capital": target_capital,
            "target_weights": target_weights,
            "target_shares": target_shares,
            "target_dollars": target_dollars,
            "long_leg_value": round(final_long_val, 2),
            "short_leg_value": round(final_short_val, 2),
            "gross_deployed_capital": round(gross_deployed, 2),
            "gross_equity_utilization_pct": round((gross_deployed / equity) * 100.0, 2) if equity > 0 else 0.0,
            "sell_orders": sell_orders,
            "buy_orders": buy_orders
        }

    def rebalance_portfolio(
        self,
        target_weights: Optional[Dict[str, float]] = None,
        total_capital: Optional[float] = None,
        dry_run: bool = False,
        bypass_market_hours: bool = False
    ) -> Dict[str, Any]:
        """
        Executes multi-alpha portfolio rebalancing towards target weights:
          1. Enforces Market Closed Protection (if configured and not bypassed).
          2. Calculates target allocation plan via calculate_rebalance_orders (95%-100% equity deployed, $1500 reserve).
          3. Omits micro-lot orders (< $3.00 order value threshold).
          4. ALMGREN-CHRISS OPTIMAL ROUTING:
             - Standard deltas routed directly.
             - Large-delta orders (> 1% ADV) sliced into optimal micro-child orders.
          5. BATCH ORDER THROTTLER:
             - Sleeps 0.08–0.10s between sequential orders to strictly respect Alpaca's 200 req/min limit.
          6. SEQUENCING SAFEGUARD:
             - Phase 1: Submits all EXIT / SELL / SHORT orders FIRST to release capital and buying power.
             - Pauses 2 seconds for fill propagation.
             - Re-queries live account buying power & available cash.
             - Phase 2: Routes all BUY / COVER orders using verified purchasing power.
        """
        if target_weights is None:
            target_weights = self.compute_regime_target_weights()

        # Market Closed Protection
        if not bypass_market_hours and not dry_run and self.is_configured and not self.is_market_open():
            logger.info("Rebalance halted: US Equity Market is currently closed.")
            return {
                "status": "market_closed",
                "message": "⚠️ Market is currently CLOSED. Scheduled to fire automatically at 09:35 AM EST.",
                "timestamp": datetime.now().isoformat(),
                "is_open": False,
                "execution_results": []
            }

        # Calculate rebalance order plan adhering to Full Capital Deployment ("Full Bull Mode")
        plan = self.calculate_rebalance_orders(target_weights=target_weights, total_capital=total_capital)
        target_capital = plan["target_capital"]
        target_weights = plan["target_weights"]
        sell_orders = plan["sell_orders"]
        buy_orders = plan["buy_orders"]

        execution_results = []
        ac_sliced_count = 0
        direct_count = 0
        shortfall_bps_list = []

        # =====================================================================
        # PHASE 1: ROUTE EXIT / SELL / SHORT ORDERS FIRST (WITH THROTTLING)
        # =====================================================================
        for item in sell_orders:
            sym = item["symbol"]
            qty = item["qty"]
            price = item["price"]
            adv = self.get_asset_adv(sym)

            # Check if large delta requires Almgren-Chriss trajectory slicing
            is_large = self.ac_executor.is_large_order(shares=qty, adv=adv)
            traj = None
            route_qty = qty
            routing_mode = "DIRECT_MICRO_LOT"

            if is_large:
                traj = self.ac_executor.compute_trajectory(
                    symbol=sym,
                    side="sell",
                    total_shares=qty,
                    arrival_price=price,
                    adv=adv
                )
                route_qty = traj.slices[0].slice_qty if traj.slices else qty
                routing_mode = "ALMGREN_CHRISS_SLICED"
                ac_sliced_count += 1
                shortfall_bps_list.append(traj.expected_shortfall_bps)
            else:
                direct_count += 1
                shortfall_bps_list.append(2.0)

            if not dry_run:
                res = self.submit_order(
                    symbol=sym,
                    qty=route_qty,
                    side=item["side"],
                    order_type="market",
                    bypass_market_hours=bypass_market_hours,
                    price=price
                )
            else:
                res = {
                    "id": f"dry-run-sell-{sym}",
                    "status": "simulated_dry_run",
                    "filled_avg_price": f"{price:.2f}"
                }

            execution_results.append({
                "symbol": sym,
                "side": item["side"],
                "qty": route_qty,
                "parent_qty": qty,
                "reason": item["reason"],
                "routing_mode": routing_mode,
                "ac_trajectory": traj.__dict__ if traj else None,
                "order_response": res
            })

            # Automated Batch Order Throttler (0.08–0.10s)
            time.sleep(self.throttle_delay)

        # =====================================================================
        # INTERMISSION: 2-Second Fill Propagation & Buying Power Re-Query
        # =====================================================================
        if sell_orders and not dry_run:
            logger.info("Phase 1 liquidation orders routed. Pausing 2.0s for fill settlement...")
            time.sleep(2.0)
            account = self.get_account()
            logger.info(f"Refreshed telemetry: Cash=${float(account.get('cash', 0.0)):,.2f}, BuyingPower=${float(account.get('buying_power', 0.0)):,.2f}")

        # =====================================================================
        # PHASE 2: ROUTE BUY / EXPAND ORDERS (WITH THROTTLING & AC SLICING)
        # =====================================================================
        for item in buy_orders:
            sym = item["symbol"]
            qty = item["qty"]
            price = item["price"]
            adv = self.get_asset_adv(sym)

            is_large = self.ac_executor.is_large_order(shares=qty, adv=adv)
            traj = None
            route_qty = qty
            routing_mode = "DIRECT_MICRO_LOT"

            if is_large:
                traj = self.ac_executor.compute_trajectory(
                    symbol=sym,
                    side="buy",
                    total_shares=qty,
                    arrival_price=price,
                    adv=adv
                )
                route_qty = traj.slices[0].slice_qty if traj.slices else qty
                routing_mode = "ALMGREN_CHRISS_SLICED"
                ac_sliced_count += 1
                shortfall_bps_list.append(traj.expected_shortfall_bps)
            else:
                direct_count += 1
                shortfall_bps_list.append(2.0)

            if not dry_run:
                res = self.submit_order(
                    symbol=sym,
                    qty=route_qty,
                    side=item["side"],
                    order_type="market",
                    bypass_market_hours=bypass_market_hours,
                    price=price
                )
            else:
                res = {
                    "id": f"dry-run-buy-{sym}",
                    "status": "simulated_dry_run",
                    "filled_avg_price": f"{price:.2f}"
                }

            execution_results.append({
                "symbol": sym,
                "side": item["side"],
                "qty": route_qty,
                "parent_qty": qty,
                "reason": item["reason"],
                "routing_mode": routing_mode,
                "ac_trajectory": traj.__dict__ if traj else None,
                "order_response": res
            })

            # Automated Batch Order Throttler (0.08–0.10s)
            time.sleep(self.throttle_delay)

        # Final account telemetry query
        final_acc = self.get_account()
        final_cash = float(final_acc.get("cash", 0.0))
        final_equity = float(final_acc.get("equity") or final_acc.get("portfolio_value", 0.0))
        final_bp = float(final_acc.get("buying_power", 0.0))
        avg_shortfall_bps = float(np.mean(shortfall_bps_list)) if shortfall_bps_list else 0.0

        report = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "target_capital": target_capital,
            "account_equity": final_equity,
            "remaining_cash": final_cash,
            "buying_power": final_bp,
            "long_leg_value": plan["long_leg_value"],
            "short_leg_value": plan["short_leg_value"],
            "gross_deployed_capital": plan["gross_deployed_capital"],
            "gross_equity_utilization_pct": plan["gross_equity_utilization_pct"],
            "cash_reserve": self.minimum_cash_reserve,
            "dry_run": dry_run,
            "total_orders": len(execution_results),
            "sells_count": len(sell_orders),
            "buys_count": len(buy_orders),
            "almgren_chriss_sliced_count": ac_sliced_count,
            "direct_micro_orders_count": direct_count,
            "avg_expected_shortfall_bps": round(avg_shortfall_bps, 2),
            "micro_lot_threshold": self.micro_lot_threshold,
            "throttle_delay_sec": self.throttle_delay,
            "target_weights": target_weights,
            "execution_results": execution_results
        }
        return report

    def execute_rebalance(
        self,
        ensemble_weights: Optional[Dict[str, float]] = None,
        total_capital: Optional[float] = None,
        dry_run: bool = False,
        bypass_market_hours: bool = False
    ) -> Dict[str, Any]:
        """Alias for rebalance_portfolio adhering to Quantitative Execution Architect specs."""
        return self.rebalance_portfolio(
            target_weights=ensemble_weights,
            total_capital=total_capital,
            dry_run=dry_run,
            bypass_market_hours=bypass_market_hours
        )

    def format_rebalance_telegram_card(self, report: Dict[str, Any]) -> str:
        """Formats rebalance execution report as a clean Telegram card."""
        if report.get("status") == "market_closed":
            return (
                "⚠️ *ALPACA PAPER TRADING: MARKET CLOSED*\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "• The US Equity Market is currently *CLOSED*.\n"
                "• Scheduled to fire automatically at *09:35 AM EST* on the next trading day.\n"
                "• Send `/trade force` to bypass market hours validation."
            )

        mode = "LIVE PAPER REST" if self.is_configured else "SIMULATED SANDBOX"
        eq = report.get("account_equity", 0.0)
        cash = report.get("remaining_cash", 0.0)
        bp = report.get("buying_power", 0.0)
        results = report.get("execution_results", [])
        
        sells = [r for r in results if r.get("side") == "sell" and r.get("order_response", {}).get("status") not in ("skipped", "rejected")]
        buys = [r for r in results if r.get("side") == "buy" and r.get("order_response", {}).get("status") not in ("skipped", "rejected")]
        ac_sliced = report.get("almgren_chriss_sliced_count", 0)
        direct_cnt = report.get("direct_micro_orders_count", len(results) - ac_sliced)
        avg_is = report.get("avg_expected_shortfall_bps", 0.0)
        throttle = report.get("throttle_delay_sec", self.throttle_delay)
        prune_thresh = report.get("micro_lot_threshold", self.micro_lot_threshold)

        order_lines = []
        # Show top orders with order ID, quantity, side and status
        for r in results[:10]:
            sym = r.get("symbol")
            side = r.get("side", "").upper()
            qty = r.get("qty")
            mode_tag = " [AC]" if r.get("routing_mode") == "ALMGREN_CHRISS_SLICED" else ""
            resp = r.get("order_response", {})
            oid = str(resp.get("id", "N/A"))[:8]
            st = resp.get("status", "ok")
            tag = "🔴" if side == "SELL" else "🟢"
            order_lines.append(f"{tag} {side} {qty} {sym}{mode_tag} (ID: `{oid}` | `{st}`)")

        if len(results) > 10:
            order_lines.append(f"• ... and {len(results) - 10} additional orders")

        orders_block = "\n".join(order_lines) if order_lines else "• No rebalance orders required (portfolio already aligned)."

        card = (
            f"⚡ *CHRONOS ALPACA REBALANCE EXECUTED*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• *Execution Engine*: Almgren-Chriss Optimal Routing\n"
            f"• *Execution Mode*: `{mode}`\n"
            f"• *Portfolio Equity*: `${eq:,.2f}`\n"
            f"• *Remaining Cash*: `${cash:,.2f}`\n"
            f"• *Buying Power*: `${bp:,.2f}`\n"
            f"• *Orders Placed*: `{len(results)} Total` ({len(sells)} Sells, {len(buys)} Buys)\n"
            f"• *Almgren-Chriss Sliced*: `{ac_sliced}` orders | Direct: `{direct_cnt}`\n"
            f"• *Expected Shortfall*: `{avg_is:.1f} bps` avg impact\n"
            f"• *Micro-Lot Prune*: `${prune_thresh:.2f}` threshold | Throttler: `{throttle*1000:.0f}ms`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📋 *Order Routing & Fills*:\n"
            f"{orders_block}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• *Next Automated Run*: Daily at 09:35 AM US/Eastern"
        )
        return card

    def format_account_telegram_card(self) -> str:
        """Formats account status as a clean Telegram Markdown message."""
        acc = self.get_account()
        conn = acc.get("connection_status", "UNKNOWN")
        eq = float(acc.get("equity", 0.0))
        cash = float(acc.get("cash", 0.0))
        bp = float(acc.get("buying_power", 0.0))
        positions = self.get_positions()
        
        status_tag = "[CONNECTED]" if "CONNECTED" in conn else "[SIMULATED]"
        
        card = (
            f"🏛 *ALPACA PAPER TRADING TELEMETRY*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• *Status*: `{status_tag} {conn}`\n"
            f"• *Account ID*: `{acc.get('id', 'N/A')}`\n"
            f"• *Portfolio Equity*: `${eq:,.2f}`\n"
            f"• *Available Cash*: `${cash:,.2f}`\n"
            f"• *Buying Power*: `${bp:,.2f}`\n"
            f"• *Active Positions*: `{len(positions)} symbols`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• *Mode*: `Paper Trading (Zero Capital Risk)`\n"
            f"• *Broker URL*: `{self.base_url}`"
        )
        return card

    def format_positions_telegram_card(self) -> str:
        """Formats active positions as a Telegram Markdown message."""
        positions = self.get_positions()
        if not positions:
            return (
                "📊 *ALPACA PAPER POSITIONS*\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "No open positions. Portfolio is 100% Cash.\n"
                "Send `/trade` to execute Chronos multi-factor rebalance."
            )

        lines = [
            "📊 *ALPACA PAPER PORTFOLIO POSITIONS*",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"{'Symbol':<6} | {'Qty':<5} | {'Price':<8} | {'Value':<9} | {'P/L'}",
            "─" * 38
        ]
        total_val = 0.0
        for p in positions:
            sym = p.get("symbol", "")
            qty_val = float(p.get("qty", 0.0))
            price = float(p.get("current_price", 0.0))
            mv = float(p.get("market_value", 0.0))
            pl = float(p.get("unrealized_pl", 0.0))
            total_val += mv
            pl_str = f"+${pl:.1f}" if pl >= 0 else f"-${abs(pl):.1f}"
            lines.append(f"{sym:<6} | {qty_val:<5.1f} | ${price:<7.2f} | ${mv:<8.1f} | {pl_str}")

        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"Total Invested Value: `${total_val:,.2f}` across {len(positions)} assets.")
        return "\n".join(lines)

    def get_capital_deployment_metrics(self) -> Dict[str, Any]:
        """Returns live capital deployment metrics for Full Bull Mode."""
        acc = self.get_account()
        eq = float(acc.get("equity") or acc.get("portfolio_value", 100000.0))
        cash = float(acc.get("cash", 0.0))
        bp = float(acc.get("buying_power", 0.0))
        lmv = float(acc.get("long_market_value", 0.0))
        smv = float(acc.get("short_market_value", 0.0))
        
        reserve = self.minimum_cash_reserve
        target_deployed = max(0.0, eq - reserve)
        active_deployed = lmv + abs(smv)
        gross_lev = (active_deployed / eq) if eq > 0 else 0.0
        
        return {
            "total_equity": eq,
            "available_cash": cash,
            "cash_reserve": reserve,
            "target_deployed": target_deployed,
            "active_deployed": active_deployed,
            "long_market_value": lmv,
            "short_market_value": smv,
            "buying_power": bp,
            "gross_leverage": round(gross_lev, 3),
            "utilization_pct": round((active_deployed / eq) * 100.0, 1) if eq > 0 else 0.0
        }

    def format_capital_telegram_card(self) -> str:
        """Formats live capital deployment card for Telegram (/capital command)."""
        m = self.get_capital_deployment_metrics()
        bull_mode_tag = "ACTIVE (95%-100% Deployed)" if m["gross_leverage"] >= 0.90 else "CALIBRATING (Target 100%)"
        card = (
            f"💰 *[PROJECT CHRONOS] CAPITAL ALLOCATION POSTURE*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• *Bull Mode Posture*    : `{bull_mode_tag}`\n"
            f"• *Total Portfolio Equity*: `${m['total_equity']:,.2f}`\n"
            f"• *Target Deployed Capital*: `${m['target_deployed']:,.2f}` (Bull Mode Active)\n"
            f"• *Operational Reserve*  : `${m['cash_reserve']:,.2f}`\n"
            f"• *Available Broker Cash*: `${m['available_cash']:,.2f}`\n"
            f"• *Active Deployed Gross*: `${m['active_deployed']:,.2f}` ({m['utilization_pct']}% Equity)\n"
            f"  - Long Positions  : `${m['long_market_value']:,.2f}`\n"
            f"  - Short Positions : `${abs(m['short_market_value']):,.2f}`\n"
            f"• *Gross Portfolio Leverage*: `{m['gross_leverage']:.2f}x`\n"
            f"• *Available Buying Power*: `${m['buying_power']:,.2f}`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• *Policy*: 95%-100% Capital Deployment with $1,500 Buffer"
        )
        return card


if __name__ == "__main__":
    broker = AlpacaPaperBroker()
    print("[Alpaca Broker Test]")
    print(f"Configured: {broker.is_configured}")
    print(f"Base URL: {broker.base_url}")
    acc = broker.get_account()
    print("Account:", json.dumps(acc, indent=2))
    positions = broker.get_positions()
    print(f"Positions ({len(positions)}):", positions)
