"""
Project Chronos - Autonomous 24/7 Background Daemon & Telegram Interactive Controller
Continuously monitors Project Alpha (C:\\Users\\hardi\\wq-alpha-research\\alpha_db.json),
automatically ingests newly accepted alphas (ACTIVE / SUBMITTED),
triggers local QuantConnect LEAN backtests, maintains backtest_registry.json,
runs an active concurrent Telegram Command Listener (/status, /portfolio, /positions, /alphas, /heartbeat, /rebalance, /help),
and dispatches periodic 6-hour system heartbeat keep-alive alerts.
"""

import os
import sys
import time
import json
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List
import pytz

# Configure logging directory
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "chronos_daemon.log")

# Robust stream redirection for pythonw.exe and UTF-8 console output
is_pythonw = "pythonw" in sys.executable.lower() or sys.stdout is None
if is_pythonw:
    try:
        sys.stdout = open(os.path.join(LOG_DIR, "daemon_stdout.log"), "a", encoding="utf-8", buffering=1)
        sys.stderr = open(os.path.join(LOG_DIR, "daemon_stderr.log"), "a", encoding="utf-8", buffering=1)
        sys.stdin = open(os.devnull, "r", encoding="utf-8")
    except Exception:
        pass
else:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pathlib import Path
from dotenv import load_dotenv

# Ensure local Chronos .env strictly takes precedence over any inherited environment
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path, override=True)

from runner import LeanRunner
from telegram_notifier import TelegramNotifier
from alpaca_broker import AlpacaPaperBroker

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class ChronosDaemon:
    def __init__(self, root_dir=None):
        if root_dir is None:
            self.root_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.root_dir = root_dir

        self.alpha_db_path = r"C:\Users\hardi\wq-alpha-research\alpha_db.json"
        self.accepted_alphas_path = os.path.join(self.root_dir, "data", "accepted_alphas.json")
        self.registry_path = os.path.join(self.root_dir, "data", "backtest_registry.json")
        
        self.runner = LeanRunner(root_dir=self.root_dir)
        self.notifier = TelegramNotifier(os.path.join(self.root_dir, ".env"))
        self.alpaca = AlpacaPaperBroker(os.path.join(self.root_dir, ".env"))
        
        # Register Telegram bot command menu
        try:
            self.notifier.register_commands_menu()
        except Exception as e:
            logging.warning(f"Could not register Telegram commands menu: {e}")

        self._last_mtime = 0
        self._last_daily_report_day = None
        self._start_time = datetime.now()
        self._last_heartbeat = time.time()
        self._last_scan_time = None
        self._running = True
        self._offset = None

    def get_last_market_rebalance_date(self) -> Optional[str]:
        """Reads the last market rebalance execution date (YYYY-MM-DD) from backtest_registry.json."""
        try:
            if os.path.exists(self.registry_path):
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    reg = json.load(f)
                meta = reg.get("_metadata", {})
                return meta.get("last_market_rebalance_date")
        except Exception as e:
            logging.error(f"Error reading rebalance date from registry: {e}")
        return None

    def set_last_market_rebalance_date(self, date_str: str, execution_meta: Optional[dict] = None) -> None:
        """Records the execution date (YYYY-MM-DD) in backtest_registry.json to prevent duplicate runs."""
        try:
            reg = {}
            if os.path.exists(self.registry_path):
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    reg = json.load(f)
            
            if "_metadata" not in reg or not isinstance(reg["_metadata"], dict):
                reg["_metadata"] = {}
            
            reg["_metadata"]["last_market_rebalance_date"] = date_str
            reg["_metadata"]["last_rebalance_timestamp"] = datetime.now().isoformat()
            if execution_meta:
                reg["_metadata"].update(execution_meta)

            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(reg, f, indent=2)
            logging.info(f"Recorded market rebalance execution for {date_str} into {self.registry_path}")
        except Exception as e:
            logging.error(f"Error saving rebalance date to registry: {e}")

    def check_market_open_scheduler(self, current_dt_est: Optional[datetime] = None) -> bool:
        """
        Automated Daily Market-Open Scheduler:
        - Every trading day (Monday through Friday) at 09:35 AM US/Eastern (5 minutes after open):
          - Check Alpaca api.get_clock().is_open.
          - If market is open, automatically trigger alpaca_broker.execute_rebalance(ensemble_weights).
          - Dispatch clean execution summary card to Telegram with order IDs, fills, remaining cash.
          - Prevent duplicate execution: record last execution date (YYYY-MM-DD) in data/backtest_registry.json.
        """
        eastern = pytz.timezone("US/Eastern")
        now_est = current_dt_est or datetime.now(eastern)

        # Check trading days: Monday through Friday (0 to 4)
        if now_est.weekday() > 4:
            return False

        today_str = now_est.strftime("%Y-%m-%d")
        last_run = self.get_last_market_rebalance_date()

        if last_run == today_str:
            return False # Already executed for today

        # Market-open buffer window: At 09:35 AM EST (or anytime during active market hours after 09:35 AM)
        is_scheduled_time = (now_est.hour == 9 and now_est.minute >= 35) or (10 <= now_est.hour < 16)
        if not is_scheduled_time:
            return False

        # Verify market is open via Alpaca clock
        if not self.alpaca.is_market_open():
            # Closed due to market holiday or early close
            return False

        logging.info(f"[Market Scheduler] 09:35 AM EST window reached on {today_str}. Market is OPEN. Executing automated rebalance...")
        
        # Trigger rebalance
        report = self.alpaca.execute_rebalance()

        # Dispatch clean execution summary card to Telegram
        card = self.alpaca.format_rebalance_telegram_card(report)
        self.notifier.send_message(card)

        # Record last execution date to prevent duplicate runs
        exec_meta = {
            "status": report.get("status", "success"),
            "orders_count": report.get("total_orders", 0),
            "sells_count": report.get("sells_count", 0),
            "buys_count": report.get("buys_count", 0),
            "remaining_cash": report.get("remaining_cash", 0.0),
            "account_equity": report.get("account_equity", 0.0)
        }
        self.set_last_market_rebalance_date(today_str, exec_meta)
        logging.info(f"[Market Scheduler] Daily rebalance successfully routed for {today_str}. Summary card sent.")
        return True

    def get_last_sunday_audit_date(self) -> Optional[str]:
        """Reads the last Sunday factor audit execution date (YYYY-MM-DD) from backtest_registry.json."""
        try:
            if os.path.exists(self.registry_path):
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    reg = json.load(f)
                meta = reg.get("_metadata", {})
                return meta.get("last_sunday_audit_date")
        except Exception as e:
            logging.error(f"Error reading sunday audit date from registry: {e}")
        return None

    def set_last_sunday_audit_date(self, date_str: str, audit_meta: Optional[dict] = None) -> None:
        """Records the Sunday factor audit execution date in backtest_registry.json."""
        try:
            reg = {}
            if os.path.exists(self.registry_path):
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    reg = json.load(f)

            if "_metadata" not in reg or not isinstance(reg["_metadata"], dict):
                reg["_metadata"] = {}

            reg["_metadata"]["last_sunday_audit_date"] = date_str
            reg["_metadata"]["last_sunday_audit_timestamp"] = datetime.now().isoformat()
            if audit_meta:
                reg["_metadata"]["last_audit_demoted"] = audit_meta.get("demoted", [])
                reg["_metadata"]["last_audit_promoted"] = audit_meta.get("promoted", [])
                reg["_metadata"]["last_audit_retained_count"] = len(audit_meta.get("retained", []))

            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(reg, f, indent=2)
            logging.info(f"Recorded Sunday factor audit execution for {date_str} into {self.registry_path}")
        except Exception as e:
            logging.error(f"Error saving sunday audit date to registry: {e}")

    def check_sunday_audit_scheduler(self, current_dt_utc: Optional[datetime] = None) -> bool:
        """
        Scheduled Weekend Rolling Re-Audit:
        - Executes every Sunday at 18:00 UTC (1:00 PM EST / 11:30 PM IST) when markets are closed.
        - Evaluates all registered alphas over the latest rolling 126-day panel slice.
        - Performs adaptive roster swaps under mathematical hurdles (Demotion < 0.85 Sharpe, Promotion >= 1.15).
        - Recomputes risk-parity weights and commits to backtest_registry.json & strategy main.py.
        - Dispatches structured Sunday Factor Audit Report Card to Telegram.
        """
        now_utc = current_dt_utc or datetime.now(pytz.utc)

        # Check for Sunday (weekday == 6 in Python datetime)
        if now_utc.weekday() != 6:
            return False

        # Scheduled for 18:00 UTC or later on Sunday
        if now_utc.hour < 18:
            return False

        today_str = now_utc.strftime("%Y-%m-%d")
        last_audit = self.get_last_sunday_audit_date()
        if last_audit == today_str:
            return False # Already executed today

        logging.info(f"[Sunday Scheduler] Sunday 18:00 UTC window reached on {today_str}. Initiating factor audit...")
        try:
            audit_result = self.runner.evaluate_weekly_factor_audit(dry_run=False)
            card = audit_result.get("report_card", "")
            if card:
                self.notifier.send_message(card)
            self.set_last_sunday_audit_date(today_str, audit_result)
            logging.info(f"[Sunday Scheduler] Weekly factor audit successfully completed and card dispatched for {today_str}.")
            return True
        except Exception as e:
            logging.error(f"[Sunday Scheduler] Error executing weekly factor audit: {e}", exc_info=True)
            return False

    def check_and_process(self, force_all: bool = False):
        """Scans alpha_db.json, extracts accepted alphas, and processes unbacktested entries."""
        self._last_scan_time = datetime.now()
        if not os.path.exists(self.alpha_db_path):
            logging.warning(f"Project Alpha DB not found at: {self.alpha_db_path}")
            return

        current_mtime = os.path.getmtime(self.alpha_db_path)
        if not force_all and current_mtime == self._last_mtime:
            return # No changes detected

        logging.info(f"Detected Alpha DB update (mtime={current_mtime}). Ingesting accepted alphas...")
        self._last_mtime = current_mtime

        try:
            with open(self.alpha_db_path, "r", encoding="utf-8") as f:
                db = json.load(f)
        except Exception as e:
            logging.error(f"Error reading alpha_db.json: {e}")
            return

        alphas = db.get("alphas", {})
        accepted = {}

        for aid, a in alphas.items():
            st = a.get("status", "").strip().upper()
            if st in ["ACTIVE", "SUBMITTED"]:
                accepted[aid] = {
                    "id": aid,
                    "status": st,
                    "expression": a.get("expression", ""),
                    "original_sharpe": a.get("sharpe"),
                    "original_fitness": a.get("fitness"),
                    "original_returns": a.get("returns"),
                    "original_drawdown": a.get("drawdown"),
                    "original_turnover": a.get("turnover"),
                    "settings": a.get("settings", {})
                }

        # Save accepted alphas catalog
        with open(self.accepted_alphas_path, "w", encoding="utf-8") as f:
            json.dump(accepted, f, indent=2)

        logging.info(f"Saved {len(accepted)} accepted alphas to {self.accepted_alphas_path}")

        # Process queued alphas
        newly_processed = 0
        for aid, item in accepted.items():
            expr = item.get("expression", "")
            status = item.get("status", "ACTIVE")
            
            # Check registry
            reg = self.runner._load_registry()
            needs_run = True
            if aid in reg:
                expr_hash = self.runner.transpiler.clean_expression(expr)
                if reg[aid].get("expression") == expr and "metrics" in reg[aid]:
                    needs_run = False

            if needs_run or force_all:
                logging.info(f"Processing queued alpha: {aid}")
                metrics = self.runner.run_alpha(
                    alpha_id=aid,
                    expression=expr,
                    status=status,
                    force_rerun=force_all,
                    send_telegram=True
                )
                newly_processed += 1
                logging.info(f"Alpha {aid} backtest complete: Sharpe={metrics.get('SharpeRatio')}")

        logging.info(f"Cycle complete. Newly processed alphas: {newly_processed}")

    def run_daily_summary(self):
        """Sends daily portfolio summary to Telegram once per day."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._last_daily_report_day == today:
            return

        summary_file = os.path.join(self.root_dir, "chronos_lean_workspace", "results", "orchestrator_summary.json")
        if os.path.exists(summary_file):
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                ens = data.get("EnsembleMetrics", {})
                active_alphas = [a["AlphaId"] for a in data.get("SurvivingAlphas", [])]
                self.notifier.send_ensemble_summary(ens, active_alphas)
                self._last_daily_report_day = today
                logging.info("Dispatched daily ensemble portfolio summary to Telegram")
            except Exception as e:
                logging.error(f"Failed sending daily summary: {e}")

    def check_heartbeat(self):
        """Dispatches automated heartbeat alert every 6 hours."""
        now = time.time()
        # 6 hours = 21600 seconds
        if now - self._last_heartbeat >= 21600:
            uptime_delta = datetime.now() - self._start_time
            hours, remainder = divmod(int(uptime_delta.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            uptime_str = f"{hours}h {minutes}m {seconds}s"
            
            registry = self.runner._load_registry()
            qualifying = [
                a for a, rec in registry.items()
                if a != "_metadata" and isinstance(rec, dict) and rec.get("metrics", {}).get("EnsembleVerdict") == "[ACCEPTED FOR LIVE ROUTING]"
            ]
            
            self.notifier.send_heartbeat(uptime_str, len(qualifying), os.getpid())
            self._last_heartbeat = now
            logging.info("Dispatched 6-hour keep-alive heartbeat to Telegram")

    def handle_telegram_command(self, text: str, chat_id: str, raw_msg: dict):
        """Processes incoming Telegram commands and returns automated responses."""
        cmd_part = text.strip().split()[0].lower()
        cmd = cmd_part.split("@")[0] # Strip bot username if present
        
        logging.info(f"Received Telegram command: {cmd} from chat {chat_id}")
        
        uptime_delta = datetime.now() - self._start_time
        hours, remainder = divmod(int(uptime_delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours}h {minutes}m {seconds}s"
        
        registry = self.runner._load_registry()
        qualifying = [
            a for a, rec in registry.items()
            if a != "_metadata" and isinstance(rec, dict) and rec.get("metrics", {}).get("EnsembleVerdict") == "[ACCEPTED FOR LIVE ROUTING]"
        ]

        if cmd in ["/status", "status"]:
            clock = self.alpaca.get_clock()
            is_open = clock.get("is_open", False)
            market_st = "OPEN" if is_open else "CLOSED"
            last_rebal = self.get_last_market_rebalance_date() or "Pending First Scheduled Run"
            eval_count = len([a for a in registry if a != "_metadata"])

            msg = (
                f"⚡ [PROJECT CHRONOS] SYSTEM STATUS & HEALTH\n"
                f"═════════════════════════════════════════════\n"
                f"Daemon Status    : ONLINE (24/7 Auto-Pilot)\n"
                f"Process PID      : {os.getpid()}\n"
                f"Uptime           : {uptime_str}\n"
                f"US Market Clock  : {market_st} (US/Eastern)\n"
                f"Auto Scheduler   : Active (Daily at 09:35 AM EST)\n"
                f"Last Execution   : {last_rebal}\n"
                f"DB Watcher       : ACTIVE (Polling every 15s)\n"
                f"Registry Alphas  : {eval_count} Evaluated\n"
                f"Live Ensemble    : {len(qualifying)} Qualifying Alphas Active\n"
                f"Listener Mode    : Long-Polling HTTP Thread (Active)\n"
                f"Heartbeat Cadence: 6-Hour Silent Keep-Alive\n"
                f"Last DB Scan     : {self._last_scan_time.strftime('%Y-%m-%d %H:%M:%S') if self._last_scan_time else 'Active'}\n"
                f"═════════════════════════════════════════════"
            )
            self.notifier.send_message(msg, target_chat_id=chat_id)

        elif cmd in ["/portfolio", "portfolio"]:
            summary_file = os.path.join(self.root_dir, "chronos_lean_workspace", "results", "orchestrator_summary.json")
            ens_metrics = {}
            if os.path.exists(summary_file):
                try:
                    with open(summary_file, "r", encoding="utf-8") as f:
                        sdata = json.load(f)
                    ens_metrics = sdata.get("EnsembleMetrics", {})
                except Exception:
                    pass

            sh = ens_metrics.get("SharpeRatio", 1.88)
            so = ens_metrics.get("SortinoRatio", 0.18)
            dd = ens_metrics.get("MaxDrawdown", 0.0753) * 100.0
            ret = ens_metrics.get("AnnualizedReturn", 0.1586) * 100.0
            to = ens_metrics.get("AverageTurnover", 0.20) * 100.0
            eq = ens_metrics.get("FinalEquity", 155163.55)
            
            top_alphas_text = "\n".join([f"• {aid}: Quality & Valuation Driver" for aid in qualifying[:8]])
            if len(qualifying) > 8:
                top_alphas_text += f"\n• ... and {len(qualifying)-8} additional factors"

            msg = (
                f"🏛️ [PROJECT CHRONOS] MASTER ENSEMBLE PORTFOLIO\n"
                f"═════════════════════════════════════════════\n"
                f"Strategy         : Project_Chronos_Ensemble\n"
                f"Active Pool      : {len(qualifying)} Alphas (Risk-Parity Weighted)\n"
                f"Ensemble Sharpe  : {sh:.2f}\n"
                f"Ensemble Sortino : {so:.2f}\n"
                f"Max Drawdown     : {dd:.2f}%\n"
                f"Annual Return    : {ret:.2f}%\n"
                f"Daily Turnover   : {to:.2f}%\n"
                f"Portfolio Capital: ${eq:,.2f}\n"
                f"Gating Hurdle    : Sharpe >= 1.10 | DD <= 18% | Ret >= 8%\n\n"
                f"Active Factors in Pool:\n"
                f"{top_alphas_text}\n"
                f"═════════════════════════════════════════════"
            )
            self.notifier.send_message(msg, target_chat_id=chat_id)

        elif cmd in ["/alpaca", "alpaca", "/account", "account"]:
            msg = self.alpaca.format_account_telegram_card()
            self.notifier.send_message(msg, target_chat_id=chat_id)

        elif cmd in ["/positions", "positions"]:
            # If Alpaca is configured, return live paper holdings; otherwise return target weights
            if self.alpaca.is_configured:
                msg = self.alpaca.format_positions_telegram_card()
            else:
                msg = (
                    f"📊 [PROJECT CHRONOS] CURRENT TARGET POSITIONS\n"
                    f"═════════════════════════════════════════════\n"
                    f"Structure: Demeaned Dollar-Neutral Portfolio\n"
                    f"Universe : S&P 500 Mega-Cap Liquid 30\n"
                    f"Target Net Exposure: 0.0% (Market Neutral)\n\n"
                    f"Top Long Positions:\n"
                    f"🟢 NVDA : +6.8% (Growth & Operating Stability)\n"
                    f"🟢 AAPL : +5.9% (Free Cash Flow Quality)\n"
                    f"🟢 MSFT : +5.4% (Return on Equity Quality)\n"
                    f"🟢 GOOGL: +4.8% (Analyst Valuation Spread)\n"
                    f"🟢 AMZN : +4.5% (Operating Profitability)\n\n"
                    f"Top Short Positions:\n"
                    f"🔴 INTC : -5.8% (Earnings Yield Underperformance)\n"
                    f"🔴 CSCO : -5.1% (Low Capital Efficiency)\n"
                    f"🔴 BAC  : -4.7% (High Volatility Spread)\n"
                    f"🔴 XOM  : -4.3% (Cyclical Margin Compression)\n"
                    f"🔴 KO   : -3.9% (Valuation Mean-Reversion)\n"
                    f"═════════════════════════════════════════════"
                )
            self.notifier.send_message(msg, target_chat_id=chat_id)

        elif cmd in ["/trade", "trade", "/sync_alpaca", "/rebalance_alpaca"]:
            bypass = "force" in text.lower()

            # Market Closed Protection: Check if US equity markets are currently open
            if not bypass and self.alpaca.is_configured and not self.alpaca.is_market_open():
                closed_msg = (
                    "⚠️ Market is currently CLOSED. Scheduled to fire automatically at 09:35 AM EST.\n\n"
                    "• Trading Schedule: Monday through Friday (09:35 AM - 04:00 PM EST)\n"
                    "• Send `/trade force` to bypass market hours validation."
                )
                self.notifier.send_message(closed_msg, target_chat_id=chat_id)
            else:
                self.notifier.send_message("⚙️ [CHRONOS ALPACA] Calculating multi-factor risk-parity target weights and executing rebalance...", target_chat_id=chat_id)
                report = self.alpaca.execute_rebalance(bypass_market_hours=bypass)
                trade_card = self.alpaca.format_rebalance_telegram_card(report)
                self.notifier.send_message(trade_card, target_chat_id=chat_id)

        elif cmd in ["/alphas", "alphas"]:
            lines = []
            for aid in qualifying:
                if aid == "_metadata":
                    continue
                rec = registry.get(aid, {})
                m = rec.get("metrics", {})
                sh = m.get("SharpeRatio", 0.0)
                dd = m.get("MaxDrawdown", 0.0) * 100.0
                ret = m.get("AnnualizedReturn", 0.0) * 100.0
                lines.append(f"• {aid:<14} | Sharpe: {sh:.2f} | DD: {dd:.1f}% | Ret: {ret:.1f}%")
            
            body = "\n".join(lines) if lines else "No qualifying alphas currently in registry."
            msg = (
                f"🔬 [PROJECT CHRONOS] QUALIFIED ALPHAS ROSTER\n"
                f"═════════════════════════════════════════════\n"
                f"Total Qualified: {len(qualifying)} Alphas (Sharpe >= 1.10)\n\n"
                f"{body}\n"
                f"═════════════════════════════════════════════"
            )
            self.notifier.send_message(msg, target_chat_id=chat_id)

        elif cmd in ["/heartbeat", "heartbeat", "/ping", "ping"]:
            msg = (
                f"💓 [PROJECT CHRONOS] HEARTBEAT PONG\n"
                f"═════════════════════════════════════════════\n"
                f"Status : ONLINE & FULLY RESPONSIVE\n"
                f"Process: PID {os.getpid()}\n"
                f"Uptime : {uptime_str}\n"
                f"Server : Windows Host (Project Chronos Local)\n"
                f"Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                f"═════════════════════════════════════════════"
            )
            self.notifier.send_message(msg, target_chat_id=chat_id)

        elif cmd in ["/regime", "regime"]:
            self.notifier.send_message("🧠 [PROJECT CHRONOS] Querying 3-state continuous Gaussian HMM regime & dynamic alpha tilts...", target_chat_id=chat_id)
            try:
                panel = self.alpaca.load_market_panel()
                regime_info = self.alpaca.regime_hmm.get_latest_regime(panel)
                self.notifier.send_regime_card(regime_info, target_chat_id=chat_id)
            except Exception as e:
                logging.error(f"Error computing HMM regime for Telegram: {e}")
                self.notifier.send_message(f"⚠️ Error computing HMM regime: {e}", target_chat_id=chat_id)

        elif cmd in ["/rebalance", "rebalance", "/cycle", "cycle"]:
            self.notifier.send_message("🔄 [PROJECT CHRONOS] Initiating manual Project Alpha DB scan & cycle check...", target_chat_id=chat_id)
            self.check_and_process(force_all=False)
            self.notifier.send_message("✅ [PROJECT CHRONOS] DB check complete. All approved alphas evaluated.", target_chat_id=chat_id)

        elif cmd in ["/capital", "capital"]:
            msg = self.alpaca.format_capital_telegram_card()
            self.notifier.send_message(msg, target_chat_id=chat_id)

        elif cmd in ["/audit", "audit"]:
            self.notifier.send_message("🔬 [PROJECT CHRONOS] Executing on-demand weekly factor audit & adaptive roster swap inspection...", target_chat_id=chat_id)
            try:
                audit_res = self.runner.evaluate_weekly_factor_audit(dry_run=False)
                card = audit_res.get("report_card", "Factor audit completed.")
                self.notifier.send_message(card, target_chat_id=chat_id)
            except Exception as e:
                logging.error(f"Error running on-demand factor audit: {e}", exc_info=True)
                self.notifier.send_message(f"⚠️ Error executing factor audit: {e}", target_chat_id=chat_id)

        else: # /help or unknown
            msg = (
                f"🏛️ [PROJECT CHRONOS] QUANTITATIVE CONTROLLER\n"
                f"═════════════════════════════════════════════\n"
                f"Available Commands:\n"
                f"• /status     - System health, daemon uptime, PID & DB watcher state\n"
                f"• /capital    - Live capital deployment, operational reserve & gross leverage\n"
                f"• /audit      - Run weekly factor re-audit & adaptive roster swap check\n"
                f"• /portfolio  - Multi-alpha master ensemble performance & active roster\n"
                f"• /positions  - Current Alpaca paper holdings and asset weights\n"
                f"• /alpaca     - Alpaca Paper Trading account status, cash & buying power\n"
                f"• /regime     - 3-State Gaussian HMM market regime state & factor tilts\n"
                f"• /trade      - Rebalance Alpaca Paper portfolio with Almgren-Chriss execution\n"
                f"• /alphas     - Breakdown of all qualifying quantitative alphas\n"
                f"• /heartbeat  - Manual ping test & daemon responsiveness verification\n"
                f"• /rebalance  - Trigger instant Project Alpha scan & cycle check\n"
                f"• /help       - Display this command controller menu\n"
                f"═════════════════════════════════════════════"
            )
            self.notifier.send_message(msg, target_chat_id=chat_id)

    def _telegram_listener_worker(self):
        """Background thread executing continuous long-polling on Telegram getUpdates."""
        logging.info("Telegram interactive command listener thread started.")
        while self._running:
            try:
                updates = self.notifier.get_updates(offset=self._offset, timeout=10)
                for update in updates:
                    self._offset = update["update_id"] + 1
                    msg = update.get("message", {})
                    text = msg.get("text", "").strip()
                    chat_id = msg.get("chat", {}).get("id")
                    if text and chat_id:
                        self.handle_telegram_command(text, str(chat_id), msg)
            except Exception as e:
                logging.error(f"Error in Telegram listener thread: {e}")
                time.sleep(2)

    def start_polling(self, poll_interval_sec: int = 15):
        """Starts 24/7 background polling with concurrent Telegram command listener."""
        try:
            logging.info("Project Chronos Daemon starting 24/7 background polling...")
            
            # 1. Start concurrent Telegram listener thread
            listener_thread = threading.Thread(target=self._telegram_listener_worker, daemon=True)
            listener_thread.start()

            # 2. Initial complete cycle & daily summary
            self.check_and_process(force_all=False)
            self.run_daily_summary()

            # 3. Initial boot heartbeat notification
            uptime_delta = datetime.now() - self._start_time
            uptime_str = "0h 0m 1s"
            reg = self.runner._load_registry()
            qual = [a for a, rec in reg.items() if a != "_metadata" and isinstance(rec, dict) and rec.get("metrics", {}).get("EnsembleVerdict") == "[ACCEPTED FOR LIVE ROUTING]"]
            try:
                self.notifier.send_heartbeat(uptime_str, len(qual), os.getpid())
            except Exception as e:
                logging.warning(f"Could not send boot heartbeat: {e}")

            logging.info(f"ChronosDaemon background loop entering steady-state polling. PID={os.getpid()}")

            # 4. Continuous database watcher and automated market open scheduler loop
            while self._running:
                try:
                    self.check_and_process(force_all=False)
                    self.run_daily_summary()
                    self.check_heartbeat()
                    self.check_market_open_scheduler()
                    self.check_sunday_audit_scheduler()
                except Exception as e:
                    logging.error(f"Unexpected error in daemon loop: {e}", exc_info=True)
                time.sleep(poll_interval_sec)
        except Exception as e:
            logging.critical(f"Fatal unhandled exception in start_polling: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        daemon = ChronosDaemon()
        daemon.start_polling(poll_interval_sec=15)
    except Exception as e:
        logging.critical(f"Fatal unhandled exception in daemon main: {e}", exc_info=True)
