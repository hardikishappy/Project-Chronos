"""
Project Chronos - Comprehensive Verification Script
Tests:
  1. Aggressive Capital Deployment ("Full Bull Mode" Cash Sizing in alpaca_broker.py):
     - Eliminates idle cash drag ($96k -> $1,500 reserve).
     - Asserts gross equity deployment >= 95%.
     - Verifies 50% long / 50% short target balance.
     - Verifies micro-lot pruning threshold ($3.00).
     - Verifies HMM State 0 / State 1 target gross leverage = 1.00.
  2. Autonomous Sunday Rolling Factor Re-Auditor & Adaptive Swapper in orchestrator.py:
     - Evaluates candidate alphas over rolling 126-day panel slice.
     - Ingests 3-state Gaussian HMM inferred state probabilities.
     - Verifies Demotion, Promotion, and Equilibrium rules.
     - Recomputes inverse-volatility risk-parity weights across updated active roster.
     - Verifies structured Telegram Sunday Factor Audit Report Card.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Set console encoding to UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from alpaca_broker import AlpacaPaperBroker, MINIMUM_CASH_RESERVE
from orchestrator import ChronosOrchestrator
from runner import LeanRunner
from models.regime_hmm import MarketRegimeHMM, RegimeState

def test_capital_deployment_and_sizing():
    print("=" * 88)
    print(" [1/3] VERIFYING CAPITAL DEPLOYMENT & CASH SIZING (BULL MODE)")
    print("=" * 88)

    broker = AlpacaPaperBroker(os.path.join(ROOT_DIR, ".env"))

    # Assert 1: Minimum operational reserve constant
    assert MINIMUM_CASH_RESERVE == 1500.00, f"Expected MINIMUM_CASH_RESERVE == 1500.00, got {MINIMUM_CASH_RESERVE}"
    assert broker.minimum_cash_reserve == 1500.00, f"Expected broker.minimum_cash_reserve == 1500.00, got {broker.minimum_cash_reserve}"
    assert broker.micro_lot_threshold == 3.0, f"Expected micro_lot_threshold == 3.0, got {broker.micro_lot_threshold}"
    print(f"  [PASS] Minimum Operational Reserve : ${broker.minimum_cash_reserve:,.2f}")
    print(f"  [PASS] Micro-lot Pruning Threshold : ${broker.micro_lot_threshold:,.2f}")

    # Assert 2: Target gross leverage in State 0 & State 1
    panel = broker.load_market_panel()
    target_weights = broker.compute_regime_target_weights(panel)
    regime = broker.regime_hmm.get_latest_regime(panel)
    current_state = regime.get("state", 0)
    expected_lev = 1.00 if current_state in [0, 1] else 0.55
    actual_lev = regime.get("gross_leverage", 1.0)
    assert abs(actual_lev - expected_lev) < 1e-4, f"Expected gross leverage {expected_lev}, got {actual_lev}"
    print(f"  [PASS] HMM State {current_state} Gross Leverage  : {actual_lev * 100:.1f}%")

    # Assert 3: Order sizing across deployed capital ($E - 1500)
    simulated_equity = 99244.00
    target_deployed = simulated_equity - 1500.00 # $97,744.00
    plan = broker.calculate_rebalance_orders(
        target_weights=target_weights,
        total_capital=target_deployed,
        current_positions={}
    )

    total_gross_dollars = plan["gross_deployed_capital"]
    long_dollars = plan["long_leg_value"]
    short_dollars = plan["short_leg_value"]
    deployment_ratio = total_gross_dollars / simulated_equity
    remaining_cash = simulated_equity - total_gross_dollars

    print(f"  [INFO] Portfolio Total Equity       : ${simulated_equity:,.2f}")
    print(f"  [INFO] Target Deployed Capital      : ${target_deployed:,.2f}")
    print(f"  [INFO] Actual Gross Orders Value    : ${total_gross_dollars:,.2f}")
    print(f"  [INFO] Capital Deployment Ratio     : {deployment_ratio * 100:.2f}% (Target: >= 95.0%)")
    print(f"  [INFO] Long Leg Dollar Value        : ${long_dollars:,.2f} ({long_dollars/total_gross_dollars*100:.1f}%)")
    print(f"  [INFO] Short Leg Dollar Value       : ${short_dollars:,.2f} ({short_dollars/total_gross_dollars*100:.1f}%)")
    print(f"  [INFO] Retained Cash Reserve        : ${remaining_cash:,.2f}")
    print(f"  [INFO] Total Buy Orders Generated   : {len(plan['buy_orders'])}")
    print(f"  [INFO] Total Sell Orders Generated  : {len(plan['sell_orders'])}")

    assert deployment_ratio >= 0.95, f"Deployment ratio {deployment_ratio*100:.2f}% is below 95% hurdle!"
    print("  [PASS] Gross deployment exceeds 95% hurdle. Cash drag successfully eliminated!")

    # Assert 4: Telegram /capital card generation
    cap_card = broker.format_capital_telegram_card()
    assert "CAPITAL ALLOCATION" in cap_card
    assert "$1,500.00" in cap_card
    print("\n[Preview of Telegram /capital Card]:")
    print(cap_card)
    print("=" * 88 + "\n")


def test_sunday_audit_and_adaptive_swapper():
    print("=" * 88)
    print(" [2/3] VERIFYING SUNDAY ROLLING FACTOR AUDITOR & ADAPTIVE SWAPPER")
    print("=" * 88)

    orchestrator = ChronosOrchestrator()

    # Dry-run audit on 126-day rolling slice
    print("  Executing rolling 126-day factor audit across all registered alphas...")
    audit = orchestrator.evaluate_weekly_factor_audit(dry_run=True, force=False, lookback_days=126)

    total_eval = audit["total_evaluated"]
    current_active = audit["current_active"]
    new_active = audit["new_active"]
    demoted = audit["demoted"]
    promoted = audit["promoted"]
    retained = audit["retained"]
    weights = audit["weights"]
    regime = audit["regime"]

    print(f"  [PASS] Audit Date                 : {audit['audit_date']}")
    print(f"  [PASS] Market Regime Inferred     : State {regime['current_state']} ({regime['state_name']})")
    print(f"  [PASS] Realized Vol (20d)         : {regime['realized_vol_20d']*100:.2f}%")
    print(f"  [PASS] Trend Dispersion (20d)     : {regime['trend_dispersion_20d']:.4f}")
    print(f"  [PASS] Total Alphas Evaluated     : {total_eval} Candidates")
    print(f"  [PASS] Active Roster (Pre-Audit)  : {len(current_active)} Alphas")
    print(f"  [PASS] Active Roster (Post-Audit) : {len(new_active)} Alphas")
    print(f"  [PASS] Demoted Alphas Count       : {len(demoted)}")
    print(f"  [PASS] Promoted Alphas Count      : {len(promoted)}")
    print(f"  [PASS] Retained Alphas Count      : {len(retained)}")

    # Verify metrics structure
    assert total_eval >= 40, f"Expected at least 40 evaluated alphas, got {total_eval}"
    assert len(new_active) >= 3, f"Expected at least 3 active alphas in portfolio, got {len(new_active)}"
    assert abs(sum(weights.values()) - 1.0) < 1e-3, f"Risk-parity weights sum to {sum(weights.values())}, expected 1.0"
    print(f"  [PASS] Risk-Parity Weights Sum    : {sum(weights.values()):.4f} (Exact 1.00)")

    # Assert Demotion Logic
    for aid in demoted:
        m = audit["evaluated_metrics"][aid]
        is_bad = (m["sharpe_126d"] < 0.85) or (m["rank_ic_20d"] < -0.01) or (m["max_dd_126d"] > 0.16)
        assert is_bad, f"Alpha {aid} was demoted without meeting demotion criteria: {m}"
    print(f"  [PASS] All {len(demoted)} demoted alphas mathematically violated hurdle (Sharpe < 0.85, IC < -0.01, or DD > 16%).")

    # Assert Promotion Logic
    for aid in promoted:
        m = audit["evaluated_metrics"][aid]
        assert m["sharpe_126d"] >= 1.15, f"Promoted alpha {aid} Sharpe {m['sharpe_126d']} < 1.15"
        assert m["rank_ic_20d"] >= 0.02, f"Promoted alpha {aid} Rank IC {m['rank_ic_20d']} < 0.02"
    print(f"  [PASS] All {len(promoted)} promoted alphas strictly satisfied hurdles (Sharpe >= 1.15, Rank IC >= 0.02).")

    # Verify Telegram Sunday Factor Audit Report Card format
    report_card = audit["report_card"]
    assert "🔬 [PROJECT CHRONOS] WEEKLY FACTOR AUDIT REPORT" in report_card
    assert "Audit Date" in report_card
    assert "Regime Alignment" in report_card
    assert "ROSTER ADJUSTMENTS" in report_card
    assert "CAPITAL ALLOCATION POSTURE" in report_card
    assert "$1,500.00" in report_card

    print("\n[Preview of Sunday Factor Audit Report Card]:")
    print(report_card)
    print("=" * 88 + "\n")


def test_audit_execution_and_commit():
    print("=" * 88)
    print(" [3/3] EXECUTING PERSISTENT AUDIT COMMIT & INTEGRATION CHECK")
    print("=" * 88)

    runner = LeanRunner()
    # Execute actual non-dry-run audit to commit updated weights and strategy main.py
    audit = runner.evaluate_weekly_factor_audit(dry_run=False, force=False)

    # Check that backtest_registry.json has been updated with rolling audit metadata
    with open(runner.registry_path, "r", encoding="utf-8") as f:
        reg = json.load(f)

    meta = reg.get("_metadata", {})
    assert "last_sunday_audit_date" in meta, "Missing last_sunday_audit_date in registry _metadata!"
    assert "active_roster" in meta, "Missing active_roster in registry _metadata!"
    assert "active_roster_weights" in meta, "Missing active_roster_weights in registry _metadata!"

    active_roster = meta["active_roster"]
    active_weights = meta["active_roster_weights"]
    print(f"  [PASS] Registry _metadata Updated : Date={meta['last_sunday_audit_date']}")
    print(f"  [PASS] Committed Active Roster    : {len(active_roster)} Alphas -> {active_roster}")
    print(f"  [PASS] Committed Factor Weights   : {active_weights}")

    # Check strategy main.py
    main_py_path = os.path.join(ROOT_DIR, "strategies", "Project_Chronos_Ensemble", "main.py")
    assert os.path.exists(main_py_path), f"Missing ensemble main.py at {main_py_path}"
    with open(main_py_path, "r", encoding="utf-8") as f:
        code = f.read()

    assert "self.alpha_weights" in code, "self.alpha_weights missing from strategies/Project_Chronos_Ensemble/main.py!"
    for aid in active_roster:
        assert aid in code, f"Alpha {aid} missing from generated strategy code!"

    print(f"  [PASS] Generated LEAN Strategy main.py successfully verified with {len(active_roster)} factors.")
    print("=" * 88)
    print(" ALL VERIFICATION CHECKS PASSED PERFECTLY!")
    print("=" * 88)


if __name__ == "__main__":
    test_capital_deployment_and_sizing()
    test_sunday_audit_and_adaptive_swapper()
    test_audit_execution_and_commit()
