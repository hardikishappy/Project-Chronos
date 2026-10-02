"""
Project Chronos - Almgren-Chriss Optimal Execution Engine
Implementation of the Almgren-Chriss (2000) optimal liquidation / acquisition framework
with transient (temporary) and permanent market impact for large-order routing.

Mathematical Framework:
  1. Permanent Price Impact: g(v) = gamma * v
  2. Temporary / Transient Impact: h(v) = eta * v, where v = n_k / tau
  3. Objective Function: min U(x) = E[x] + lambda * V[x]
  4. Hyperbolic Trajectory:
     x_j = (sinh(kappa * (T - t_j)) / sinh(kappa * T)) * X_0
     where kappa = sqrt(lambda * sigma^2 / eta)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

@dataclass
class ExecutionSlice:
    """Represents a discrete micro-child order slice."""
    step: int
    minute_offset: float
    slice_qty: int
    remaining_qty: int
    pct_of_parent: float
    trade_rate_per_min: float
    expected_price_impact_bps: float

@dataclass
class AlmgrenChrissTrajectory:
    """Complete optimal execution trajectory report."""
    symbol: str
    side: str
    total_shares: int
    arrival_price: float
    adv: float
    volatility: float
    time_horizon_min: float
    num_slices: int
    tau_min: float
    kappa: float
    lambda_risk: float
    eta: float
    gamma: float
    epsilon: float
    expected_shortfall_dollars: float
    expected_shortfall_bps: float
    shortfall_variance: float
    utility: float
    slices: List[ExecutionSlice]
    is_sliced: bool
    adv_fraction: float

class AlmgrenChrissExecutor:
    """
    Principal execution engine for large-delta rebalance orders.
    Calculates the calculus-of-variations optimal trade schedule to balance
    market impact costs (permanent + transient) against volatility timing risk.
    """

    def __init__(
        self,
        default_lambda: float = 1e-6,
        large_order_adv_threshold: float = 0.01, # 1% of ADV
        trading_minutes_per_day: float = 390.0
    ):
        self.default_lambda = default_lambda
        self.large_order_adv_threshold = large_order_adv_threshold
        self.trading_minutes_per_day = trading_minutes_per_day

    def is_large_order(self, shares: int, adv: float, threshold_pct: Optional[float] = None) -> bool:
        """Determines if order delta requires optimal Almgren-Chriss slicing vs direct routing."""
        thresh = threshold_pct if threshold_pct is not None else self.large_order_adv_threshold
        if adv <= 0:
            return abs(shares) >= 500
        adv_frac = abs(shares) / adv
        return adv_frac >= thresh

    def calibrate_parameters(
        self,
        arrival_price: float,
        adv: float,
        daily_vol_pct: float,
        tau_days: float
    ) -> Tuple[float, float, float, float]:
        """
        Calibrates Almgren-Chriss parameters:
          - sigma: daily price volatility (dollars / day^0.5)
          - gamma: permanent impact coefficient ($ / share)
          - eta: temporary / transient impact coefficient ($ * day / share)
          - epsilon: fixed transaction cost / half-spread ($ / share)
        """
        # Daily price volatility (dollars)
        sigma = max(0.01, arrival_price * max(0.005, daily_vol_pct))
        
        # Safe ADV bound
        safe_adv = max(10_000.0, float(adv))

        # Gamma (permanent price impact): typical empirical scaling gamma = 0.1 * sigma / ADV
        gamma = 0.1 * sigma / safe_adv

        # Eta (temporary price impact): typical scaling eta = 0.14 * (sigma * arrival_price) / (safe_adv * tau_days)
        # Or in dollar terms: 0.10 * sigma / (safe_adv * max(tau_days, 1e-4))
        eta = (0.10 * sigma) / (safe_adv * max(tau_days, 1e-4))

        # Epsilon: half bid-ask spread proxy (~3-5 bps for liquid US large-caps)
        epsilon = max(0.005, arrival_price * 0.0003)

        return sigma, gamma, eta, epsilon

    def compute_trajectory(
        self,
        symbol: str,
        side: str,
        total_shares: int,
        arrival_price: float,
        adv: float,
        daily_vol_pct: float = 0.015,
        time_horizon_min: float = 60.0,
        num_slices: int = 10,
        lambda_risk: Optional[float] = None,
        force_slicing: bool = False
    ) -> AlmgrenChrissTrajectory:
        """
        Computes the optimal Almgren-Chriss discrete execution trajectory:
          x_j = sinh(kappa * (T - t_j)) / sinh(kappa * T) * X_0
        """
        total_abs_shares = abs(int(total_shares))
        side = side.lower()
        if side not in ("buy", "sell"):
            side = "buy" if total_shares > 0 else "sell"

        adv_fraction = (total_abs_shares / adv) if adv > 0 else 0.0
        is_sliced = force_slicing or self.is_large_order(total_abs_shares, adv)

        # If not large enough to justify slicing, return single direct micro-lot order
        if not is_sliced or total_abs_shares <= 1 or num_slices <= 1:
            single_slice = ExecutionSlice(
                step=1,
                minute_offset=0.0,
                slice_qty=total_abs_shares,
                remaining_qty=0,
                pct_of_parent=100.0,
                trade_rate_per_min=float(total_abs_shares),
                expected_price_impact_bps=2.0
            )
            return AlmgrenChrissTrajectory(
                symbol=symbol,
                side=side,
                total_shares=total_abs_shares,
                arrival_price=arrival_price,
                adv=adv,
                volatility=daily_vol_pct,
                time_horizon_min=time_horizon_min,
                num_slices=1,
                tau_min=time_horizon_min,
                kappa=0.0,
                lambda_risk=lambda_risk or self.default_lambda,
                eta=0.0,
                gamma=0.0,
                epsilon=arrival_price * 0.0003,
                expected_shortfall_dollars=total_abs_shares * arrival_price * 0.0002,
                expected_shortfall_bps=2.0,
                shortfall_variance=0.0,
                utility=total_abs_shares * arrival_price * 0.0002,
                slices=[single_slice],
                is_sliced=False,
                adv_fraction=adv_fraction
            )

        # Convert time horizon to fractional trading days
        T_days = time_horizon_min / self.trading_minutes_per_day
        num_slices = max(2, int(num_slices))
        tau_days = T_days / num_slices
        tau_min = time_horizon_min / num_slices

        # Calibrate coefficients
        sigma, gamma, eta, epsilon = self.calibrate_parameters(arrival_price, adv, daily_vol_pct, tau_days)
        l_risk = lambda_risk if lambda_risk is not None else self.default_lambda

        # Compute urgency / urgency parameter kappa = sqrt(lambda * sigma^2 / eta)
        # Using discrete exact formulation: cosh(kappa * tau) = 1 + (lambda * sigma^2 * tau^2) / (2 * eta)
        arg = (l_risk * (sigma ** 2) * (tau_days ** 2)) / (2.0 * eta)
        if arg > 1e-7:
            # Discrete formulation
            cosh_val = 1.0 + arg
            kappa_tau = math.acosh(cosh_val)
            kappa = kappa_tau / tau_days
        else:
            # Continuous formulation
            kappa = math.sqrt(max(1e-12, (l_risk * (sigma ** 2)) / eta))

        kappa_T = kappa * T_days

        # Compute trajectory positions x_j at discrete steps j = 0, ..., N
        t_j_steps = [j * tau_days for j in range(num_slices + 1)]
        x_positions = []

        for j in range(num_slices + 1):
            if j == num_slices:
                x_positions.append(0.0)
            elif kappa_T < 1e-4:
                # Taylor expansion for small kappa*T -> Linear TWAP
                x_positions.append(total_abs_shares * (1.0 - (j / float(num_slices))))
            else:
                sinh_num = math.sinh(max(0.0, kappa * (T_days - t_j_steps[j])))
                sinh_den = math.sinh(kappa_T)
                x_positions.append(total_abs_shares * (sinh_num / sinh_den))

        # Compute trade sizes n_k = x_{k-1} - x_k
        raw_slices = [max(0.0, x_positions[k - 1] - x_positions[k]) for k in range(1, num_slices + 1)]
        total_raw = sum(raw_slices)
        if total_raw > 0:
            norm_factor = total_abs_shares / total_raw
            int_slices = [int(round(s * norm_factor)) for s in raw_slices]
        else:
            int_slices = [total_abs_shares // num_slices] * num_slices

        # Ensure exact integer sum conservation: sum(int_slices) == total_abs_shares
        diff = total_abs_shares - sum(int_slices)
        if diff != 0:
            # Distribute residual across largest slices
            int_slices[0] += diff

        # Generate slice objects
        slices: List[ExecutionSlice] = []
        rem = total_abs_shares
        for k in range(num_slices):
            qty_k = int_slices[k]
            rem -= qty_k
            minute_offset = round(k * tau_min, 1)
            pct = round((qty_k / total_abs_shares) * 100.0, 2)
            rate_per_min = round(qty_k / max(1.0, tau_min), 2)
            
            # Temporary price impact in bps for this slice: (eta * v_k) / arrival_price * 10000
            v_k = qty_k / max(1e-4, tau_days)
            slice_impact_bps = round(((eta * v_k) / arrival_price) * 10000.0, 2)

            slices.append(ExecutionSlice(
                step=k + 1,
                minute_offset=minute_offset,
                slice_qty=qty_k,
                remaining_qty=max(0, rem),
                pct_of_parent=pct,
                trade_rate_per_min=rate_per_min,
                expected_price_impact_bps=slice_impact_bps
            ))

        # Expected Shortfall E[x] = 0.5 * gamma * X_0^2 + epsilon * sum(|n_k|) + (eta / tau) * sum(n_k^2)
        permanent_term = 0.5 * gamma * (total_abs_shares ** 2)
        spread_term = epsilon * total_abs_shares
        temporary_term = (eta / max(1e-4, tau_days)) * sum(n ** 2 for n in int_slices)
        expected_shortfall_dollars = permanent_term + spread_term + temporary_term

        # Relative shortfall in basis points (bps)
        parent_dollar_val = max(1.0, total_abs_shares * arrival_price)
        expected_shortfall_bps = (expected_shortfall_dollars / parent_dollar_val) * 10000.0

        # Timing Variance V[x] = sigma^2 * sum(tau * x_k^2)
        shortfall_variance = (sigma ** 2) * sum(tau_days * (x ** 2) for x in x_positions[1:])

        # Total Utility Objective = E[x] + lambda * V[x]
        utility = expected_shortfall_dollars + (l_risk * shortfall_variance)

        return AlmgrenChrissTrajectory(
            symbol=symbol,
            side=side,
            total_shares=total_abs_shares,
            arrival_price=arrival_price,
            adv=adv,
            volatility=daily_vol_pct,
            time_horizon_min=time_horizon_min,
            num_slices=num_slices,
            tau_min=round(tau_min, 2),
            kappa=round(kappa, 4),
            lambda_risk=l_risk,
            eta=round(eta, 8),
            gamma=round(gamma, 8),
            epsilon=round(epsilon, 4),
            expected_shortfall_dollars=round(expected_shortfall_dollars, 2),
            expected_shortfall_bps=round(expected_shortfall_bps, 2),
            shortfall_variance=round(shortfall_variance, 2),
            utility=round(utility, 2),
            slices=slices,
            is_sliced=True,
            adv_fraction=round(adv_fraction, 4)
        )

    def calculate_realized_shortfall(
        self,
        side: str,
        arrival_price: float,
        fills: List[Tuple[int, float]] # list of (fill_qty, fill_price)
    ) -> Dict[str, float]:
        """
        Calculates realized Implementation Shortfall vs Arrival Price:
          For Buy : IS = sum(n_k * (P_k - P_0))
          For Sell: IS = sum(n_k * (P_0 - P_k))
        """
        if not fills or arrival_price <= 0:
            return {"shortfall_dollars": 0.0, "shortfall_bps": 0.0, "total_shares": 0, "avg_fill_price": arrival_price}

        side = side.lower()
        total_shares = sum(q for q, p in fills)
        if total_shares <= 0:
            return {"shortfall_dollars": 0.0, "shortfall_bps": 0.0, "total_shares": 0, "avg_fill_price": arrival_price}

        dollar_notional = sum(q * p for q, p in fills)
        avg_fill_price = dollar_notional / total_shares

        if side == "buy":
            shortfall_dollars = sum(q * (p - arrival_price) for q, p in fills)
        else: # sell
            shortfall_dollars = sum(q * (arrival_price - p) for q, p in fills)

        benchmark_val = total_shares * arrival_price
        shortfall_bps = (shortfall_dollars / benchmark_val) * 10000.0

        return {
            "shortfall_dollars": round(shortfall_dollars, 2),
            "shortfall_bps": round(shortfall_bps, 2),
            "total_shares": total_shares,
            "avg_fill_price": round(avg_fill_price, 2),
            "arrival_price": round(arrival_price, 2)
        }
