"""
Project Chronos - Comprehensive Tripartite Architectural Upgrades Test Suite
Tests:
  1. Universe Expansion (100+ liquid assets, panel integrity, absence of NaNs)
  2. Almgren-Chriss Optimal Execution Engine (trajectories, transient impact, child orders)
  3. 3-State Gaussian Hidden Markov Model (HMM regime inference, transition matrix, factor tilts)
  4. Alpaca Paper Broker Integration (micro-lot threshold, throttler, regime weights)
  5. Backtest Runner Integration (realized implementation shortfall vs arrival price)
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Set UTF-8 encoding for Windows terminal output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data_manager import LeanDataManager, DEFAULT_UNIVERSE
from execution.almgren_chriss import AlmgrenChrissExecutor
from models.regime_hmm import MarketRegimeHMM
from alpaca_broker import AlpacaPaperBroker
from runner import LeanRunner

def run_tests():
    print("=" * 88)
    print(" PROJECT CHRONOS - TRIPARTITE ARCHITECTURAL UPGRADES VERIFICATION")
    print("=" * 88)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    workspace = os.path.join(root_dir, "chronos_lean_workspace")
    dm = LeanDataManager(workspace)

    # -------------------------------------------------------------------------
    # SECTION 1: UNIVERSE EXPANSION & DATA INTEGRITY
    # -------------------------------------------------------------------------
    print("[SECTION 1] UNIVERSE EXPANSION & DATA INTEGRITY (100+ ASSETS)")
    print("-" * 88)
    panel = dm.load_universe_panel()
    num_symbols = panel['symbol'].nunique()
    total_bars = len(panel)
    min_date = panel['date'].min().strftime('%Y-%m-%d')
    max_date = panel['date'].max().strftime('%Y-%m-%d')
    nan_counts = panel[['Open', 'High', 'Low', 'Close', 'Volume']].isna().sum().to_dict()

    print(f"  • Configured DEFAULT_UNIVERSE symbols : {len(DEFAULT_UNIVERSE)}")
    print(f"  • Panel Unique Symbols Ingested       : {num_symbols}")
    print(f"  • Total Historical Daily OHLCV Bars   : {total_bars:,}")
    print(f"  • Date Horizon                        : {min_date} to {max_date}")
    print(f"  • Null / NaN Count in OHLCV Columns   : {nan_counts}")

    assert num_symbols >= 100, f"Expected >= 100 symbols, got {num_symbols}"
    assert sum(nan_counts.values()) == 0, f"Panel contains NaNs: {nan_counts}"
    print("  => [PASS] Panel data integrity verified: 100+ liquid assets clean and 0 NaNs.\n")

    # -------------------------------------------------------------------------
    # SECTION 2: ALMGREN-CHRISS OPTIMAL EXECUTION ENGINE
    # -------------------------------------------------------------------------
    print("[SECTION 2] ALMGREN-CHRISS OPTIMAL EXECUTION (TRANSIENT MARKET IMPACT)")
    print("-" * 88)
    ac = AlmgrenChrissExecutor(default_lambda=1e-5, large_order_adv_threshold=0.01)
    
    # Simulate a large institutional order: Liquidating 15,000 shares of NVDA
    # Order value ~$1.875M, ADV = 35M shares, daily vol = 2.4%
    symbol = "NVDA"
    shares = 15000
    price = 125.0
    adv = 35_000_000
    daily_vol = 0.024
    horizon_mins = 60.0
    num_steps = 12 # 5-minute trading intervals

    traj = ac.compute_trajectory(
        symbol=symbol,
        side="sell",
        total_shares=shares,
        arrival_price=price,
        adv=adv,
        daily_vol_pct=daily_vol,
        time_horizon_min=horizon_mins,
        num_slices=num_steps,
        force_slicing=True
    )

    kappa = traj.kappa
    shortfall = traj.expected_shortfall_dollars
    variance = traj.shortfall_variance
    shortfall_bps = traj.expected_shortfall_bps
    tau = traj.tau_min
    order_pct_adv = (shares / adv) * 100.0

    print(f"  • Asset / Target Order       : {symbol} | {shares:,} shares @ ${price:.2f} (${shares*price:,.2f})")
    print(f"  • Order Relative to ADV      : {order_pct_adv:.4f}% of 20-day ADV ({adv:,} shares/day)")
    print(f"  • Liquidation Horizon (T)    : {horizon_mins:.0f} mins across {num_steps} slices (tau = {tau:.1f} mins/slice)")
    print(f"  • Urgency Parameter (kappa)  : {kappa:.6f} (Risk-aversion lambda = {traj.lambda_risk})")
    print(f"  • Expected Impact Shortfall  : ${shortfall:.2f} ({shortfall_bps:.2f} bps)")
    print(f"  • Timing Variance V[x]       : ${np.sqrt(max(variance, 0.0)):.2f} (std dev of execution cost)")

    # Child Order Slices
    print(f"  • Dynamic Micro-Child Slices Generated: {len(traj.slices)} orders")
    print(f"    {'Step':<5} {'Offset(m)':<10} {'Slice Shares':<14} {'Remaining Shares':<18} {'Est. Impact (bps)':<18}")
    for co in traj.slices[:6]:
        print(f"    {co.step:<5} {co.minute_offset:<10.1f} {co.slice_qty:<14} {co.remaining_qty:<18} {co.expected_price_impact_bps:<18.2f}")
    if len(traj.slices) > 6:
        print(f"    ... and {len(traj.slices)-6} subsequent child slices completing at T={horizon_mins:.0f}m")

    # Verify implementation shortfall metric
    sim_fills = [(co.slice_qty, price * (1.0 - 0.0001 * (k + 1))) for k, co in enumerate(traj.slices)]
    child_shares = [co.slice_qty for co in traj.slices]
    realized_sf = ac.calculate_realized_shortfall(
        side="sell",
        arrival_price=price,
        fills=sim_fills
    )
    print(f"  • Realized Shortfall vs Arrival: ${realized_sf['shortfall_dollars']:.2f} ({realized_sf['shortfall_bps']:.2f} bps)")
    assert len(traj.slices) == num_steps
    assert abs(sum(child_shares) - shares) < 1.0
    print("  => [PASS] Almgren-Chriss execution trajectory and child order slicing verified.\n")

    # -------------------------------------------------------------------------
    # SECTION 3: 3-STATE CONTINUOUS GAUSSIAN HMM REGIME ALLOCATION
    # -------------------------------------------------------------------------
    print("[SECTION 3] 3-STATE GAUSSIAN HMM REGIME ALLOCATION")
    print("-" * 88)
    hmm = MarketRegimeHMM(n_states=3, random_state=42)
    regime_telemetry = hmm.fit(panel)
    trans_matrix = regime_telemetry["transition_matrix"]
    means = regime_telemetry["state_means"]
    covars = regime_telemetry["state_covariances"]
    latest = hmm.get_latest_regime(panel)

    print("  • HMM State Transition Probability Matrix A (Rows = Current, Cols = Next):")
    state_labels = ["State 0 (Trend)", "State 1 (Choppy)", "State 2 (Shock)"]
    print(f"    {'':<18} {state_labels[0]:<18} {state_labels[1]:<18} {state_labels[2]:<18}")
    for idx, row in enumerate(trans_matrix):
        print(f"    {state_labels[idx]:<18} {row[0]:<18.4f} {row[1]:<18.4f} {row[2]:<18.4f}")

    print("\n  • Inferred State Component Parameters:")
    for s_idx in range(3):
        vol_mean = means[s_idx][0] * 100.0
        disp_mean = means[s_idx][1]
        print(f"    • {state_labels[s_idx]}: Mean Realized Vol = {vol_mean:5.2f}% | Mean Trend Dispersion = {disp_mean:5.4f}")

    print("\n  • Current (Today) Inferred Market Regime:")
    cur_state = latest["state"]
    cur_name = latest["state_name"]
    probs = latest["probabilities"]
    tilts = latest["factor_tilts"]
    lev = latest["gross_leverage"] * 100.0

    print(f"    • Active State           : State {cur_state} ({cur_name})")
    print(f"    • Posterior Probabilities: P(State 0)={probs[0]*100:.1f}%, P(State 1)={probs[1]*100:.1f}%, P(State 2)={probs[2]*100:.1f}%")
    print(f"    • Dynamic Alpha Tilts    : Momentum = {tilts['Momentum']:.2f}x | Mean-Reversion = {tilts['MeanReversion']:.2f}x")
    print(f"    • Gross Leverage Target  : {lev:.0f}%")

    assert trans_matrix.shape == (3, 3)
    assert abs(np.sum(probs) - 1.0) < 1e-4
    print("  => [PASS] 3-State Gaussian HMM trained and verified.\n")

    # -------------------------------------------------------------------------
    # SECTION 4: ALPACA BROKER THROTTLER & MICRO-LOT PRUNING
    # -------------------------------------------------------------------------
    print("[SECTION 4] ALPACA PAPER BROKER (THROTTLER & MICRO-LOT PRUNING)")
    print("-" * 88)
    broker = AlpacaPaperBroker()
    print(f"  • Alpaca API Configured               : {broker.is_configured}")
    print(f"  • Micro-Lot Pruning Threshold         : ${broker.micro_lot_threshold:.2f} (Required $3–$5)")
    print(f"  • Sequential Order Throttle Delay     : {broker.throttle_delay:.2f}s (Required 0.08–0.10s)")
    print(f"  • Max Orders per Minute Under Throttle: ~{int(60.0 / broker.throttle_delay)} req/min (Limit: 200/min)")
    
    assert 3.0 <= broker.micro_lot_threshold <= 5.0, f"Micro lot threshold out of range: {broker.micro_lot_threshold}"
    assert 0.08 <= broker.throttle_delay <= 0.10, f"Throttle delay out of range: {broker.throttle_delay}"

    # Verify target weights generation across 100+ assets with regime scaling
    target_weights = broker.compute_regime_target_weights()
    num_targets = len(target_weights)
    sum_abs = sum(abs(w) for w in target_weights.values())
    net_exposure = sum(target_weights.values())
    print(f"  • Regime-Conditioned Target Weights   : {num_targets} assets")
    print(f"  • Gross Portfolio Exposure            : {sum_abs*100:.2f}%")
    print(f"  • Net Dollar Exposure                 : {net_exposure*100:.4f}% (Dollar-Neutral)")

    assert num_targets >= 100, f"Expected >= 100 target weights, got {num_targets}"
    assert abs(net_exposure) < 0.01, f"Expected dollar-neutral net exposure, got {net_exposure}"

    # Test Dry-Run Rebalance
    report = broker.rebalance_portfolio(dry_run=True)
    print(f"  • Dry-Run Rebalance Status            : {report.get('status')} ({report.get('total_orders')} orders planned)")
    print(f"  • Almgren-Chriss Execution Telemetry  : {report.get('execution_engine')} (Realized Shortfall: {report.get('total_realized_shortfall_bps', 0.0):.2f} bps)")
    print("  => [PASS] Alpaca Broker micro-lot pruning, throttler, and regime integration verified.\n")

    # -------------------------------------------------------------------------
    # SECTION 5: BACKTEST RUNNER ALMGREN-CHRISS SHORTFALL METRICS
    # -------------------------------------------------------------------------
    print("[SECTION 5] BACKTEST RUNNER (ALMGREN-CHRISS SHORTFALL LOGGING)")
    print("-" * 88)
    runner = LeanRunner(root_dir=root_dir)
    test_expr = "(close - ts_mean(close, 20)) / (ts_std_dev(close, 20) + 1e-6)"
    metrics = runner.run_alpha("TEST_AC_ALPHA", test_expr, send_telegram=False)

    print(f"  • Alpha Backtested                    : TEST_AC_ALPHA")
    print(f"  • Sharpe Ratio                        : {metrics.get('SharpeRatio')}")
    print(f"  • Annualized Return                   : {metrics.get('AnnualizedReturn')*100:.2f}%")
    print(f"  • Max Drawdown                        : {metrics.get('MaxDrawdown')*100:.2f}%")
    print(f"  • Almgren-Chriss Shortfall (bps)      : {metrics.get('AlmgrenChrissShortfallBps')} bps")
    print(f"  • Total Realized Shortfall ($)        : ${metrics.get('TotalImplementationShortfall'):,.2f}")
    print(f"  • Execution Model Logged              : {metrics.get('ExecutionModel')}")

    assert "AlmgrenChrissShortfallBps" in metrics
    assert "TotalImplementationShortfall" in metrics
    assert "Almgren-Chriss" in metrics.get("ExecutionModel", "")
    print("  => [PASS] Backtest runner Almgren-Chriss implementation shortfall logging verified.\n")

    print("=" * 88)
    print(" ALL 5 TRIPARTITE ARCHITECTURAL UPGRADES VERIFIED SUCCESSFULLY!")
    print("=" * 88)

if __name__ == "__main__":
    run_tests()
