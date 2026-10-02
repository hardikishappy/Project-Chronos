"""
Project Chronos - Telegram Dispatcher
Sends automated quantitative backtest cards and ensemble daily reports
to Telegram using credentials migrated into .env.
Exclusively routed to the Project Chronos V2 Trading Brain bot (@hardik_v2_trading_bot).
"""

import os
import json
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Canonical Project Chronos V2 Trading Brain Credentials
CHRONOS_V2_BOT_TOKEN = "8789634265:AAHxDKF-K0-sQ0DnRzVlrHQZJ9qeBqFt-9Y"
CHRONOS_V2_CHAT_ID = "5551315625"
ALPHA_BOT_PREFIX = "8988607257"


class TelegramNotifier:
    def __init__(self, env_path=None):
        if env_path is None:
            self.env_path = Path(__file__).resolve().parent / ".env"
        else:
            self.env_path = Path(env_path).resolve()
        
        self.bot_token = None
        self.chat_id = None
        
        self._load_env(self.env_path)

    def _load_env(self, env_path: Path):
        # 1. Load from .env with override=True to guarantee local project config takes priority over parent env
        if env_path and env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if "=" in line and not line.startswith("#"):
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            if k == "TELEGRAM_BOT_TOKEN":
                                self.bot_token = v
                            elif k == "TELEGRAM_CHAT_ID":
                                self.chat_id = v
            except Exception as e:
                print(f"[Telegram] Warning reading {env_path}: {e}")

        # 2. Check environment if not explicitly set from .env
        if not self.bot_token:
            self.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not self.chat_id:
            self.chat_id = os.environ.get("TELEGRAM_CHAT_ID")

        # 3. Guard against Alpha bot token contamination (8988607257:... / @Happy_Alpha_bot)
        if self.bot_token and (self.bot_token.startswith(ALPHA_BOT_PREFIX) or "Happy_Alpha_bot" in self.bot_token):
            print("[Telegram] Warning: Detected contaminated Alpha bot token. Enforcing Chronos V2 Trading Brain token.")
            self.bot_token = CHRONOS_V2_BOT_TOKEN

        # 4. Enforce canonical fallback defaults
        self.bot_token = self.bot_token or CHRONOS_V2_BOT_TOKEN
        self.chat_id = self.chat_id or CHRONOS_V2_CHAT_ID

        # Ensure process environment variable strictly reflects the isolated Chronos token
        os.environ["TELEGRAM_BOT_TOKEN"] = self.bot_token
        os.environ["TELEGRAM_CHAT_ID"] = self.chat_id

    def send_message(self, text: str, target_chat_id: str = None, disable_notification: bool = False) -> bool:
        """Sends a text message to Telegram strictly via the V2 Trading Brain bot."""
        cid = target_chat_id or self.chat_id
        if not cid:
            print("[Telegram] Error: Missing TELEGRAM_CHAT_ID")
            return False

        if not self.bot_token:
            print("[Telegram] Error: No bot token available")
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": cid,
                "text": text,
                "disable_notification": "true" if disable_notification else "false"
            }
            data = urllib.parse.urlencode(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data)
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return True
        except Exception as e:
            print(f"[Telegram] Error: Failed dispatching message ({e})")

        return False

    def register_commands_menu(self) -> bool:
        """
        Registers Telegram bot commands menu via setMyCommands API:
          /portfolio - Master 16-Alpha Ensemble stats
          /alpaca    - Live Alpaca account equity, cash & buying power
          /positions - Current open market positions & live PnL
          /trade     - Trigger an immediate on-demand rebalance
          /status    - Background daemon health & uptime
        """
        commands = [
            {"command": "portfolio", "description": "Master Ensemble stats & active roster"},
            {"command": "capital", "description": "Capital deployment metrics & buying power"},
            {"command": "audit", "description": "Run on-demand weekly factor re-audit & roster check"},
            {"command": "alpaca", "description": "Live Alpaca account equity, cash & buying power"},
            {"command": "positions", "description": "Current open market positions & live PnL"},
            {"command": "regime", "description": "3-State Gaussian HMM regime & factor tilts"},
            {"command": "trade", "description": "Trigger an immediate on-demand rebalance"},
            {"command": "status", "description": "Background daemon health & uptime"}
        ]
        if not self.bot_token:
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/setMyCommands"
            headers = {"Content-Type": "application/json"}
            data = json.dumps({"commands": commands}).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    if res_json.get("ok"):
                        print("[Telegram] Successfully registered bot commands menu.")
                        return True
        except Exception as e:
            print(f"[Telegram] Warning: Failed to setMyCommands ({e})")
        return False

    def send_regime_card(self, regime_info: dict, target_chat_id: str = None) -> bool:
        """Sends Gaussian HMM regime telemetry and dynamic factor allocation card to Telegram."""
        state = regime_info.get("state", 0)
        state_name = regime_info.get("state_name", "Unknown")
        probs = regime_info.get("probabilities", [0.33, 0.33, 0.33])
        realized_vol = regime_info.get("realized_vol", 0.0) * 100.0
        dispersion = regime_info.get("trend_dispersion", 0.0)
        tilts = regime_info.get("factor_tilts", {})
        gross_leverage = regime_info.get("gross_leverage", 1.0) * 100.0
        
        state_emojis = {
            0: "🟢 [STATE 0: TRENDING / DIRECTIONAL]",
            1: "🟡 [STATE 1: CHOPPY / MEAN-REVERTING]",
            2: "🔴 [STATE 2: VOLATILE SHOCK / REGIME BREAK]"
        }
        header = state_emojis.get(state, f"State {state}")
        
        msg = (
            f"🧠 [PROJECT CHRONOS] GAUSSIAN HMM MARKET REGIME\n"
            f"═════════════════════════════════════════════\n"
            f"Current Regime   : {header}\n"
            f"Regime Detail    : {state_name}\n\n"
            f"Posterior Probabilities:\n"
            f"• P(State 0 - Trending) : {probs[0]*100:5.1f}%\n"
            f"• P(State 1 - Choppy)   : {probs[1]*100:5.1f}%\n"
            f"• P(State 2 - Shock)    : {probs[2]*100:5.1f}%\n\n"
            f"Panel Market Metrics (20d):\n"
            f"• Realized Ann. Volatility: {realized_vol:5.1f}%\n"
            f"• Cross-Sectional Disp.   : {dispersion:5.3f}\n\n"
            f"Dynamic Alpha Allocations:\n"
            f"• Momentum / Trend Tilt   : {tilts.get('Momentum', 1.0):.2f}x\n"
            f"• Mean-Reversion Tilt     : {tilts.get('MeanReversion', 1.0):.2f}x\n"
            f"• Gross Portfolio Leverage: {gross_leverage:.0f}%\n\n"
            f"Execution Protocol:\n"
            f"• Almgren-Chriss Optimal Execution (Transient Impact)\n"
            f"• Dynamic Child Slicing for Orders > 1% ADV\n"
            f"═════════════════════════════════════════════"
        )
        return self.send_message(msg, target_chat_id=target_chat_id)

    def get_updates(self, offset: int = None, timeout: int = 15) -> list:
        """Fetches incoming updates from Telegram using HTTP long-polling."""
        if not self.bot_token:
            return []

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates?timeout={timeout}"
            if offset is not None:
                url += f"&offset={offset}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout + 5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data.get("ok"):
                        return data.get("result", [])
        except Exception:
            pass
        return []

    def send_heartbeat(self, uptime_str: str, active_alpha_count: int, pid: int) -> bool:
        """Sends periodic 6-hour heartbeat notification card silently (quiet mode)."""
        msg = (
            f"💓 [PROJECT CHRONOS] SYSTEM HEARTBEAT\n"
            f"═════════════════════════════════════════════\n"
            f"Status       : HEALTHY (Autonomous Daemon Active)\n"
            f"Process PID  : {pid}\n"
            f"Uptime       : {uptime_str}\n"
            f"Active Alphas: {active_alpha_count} Qualifying Alphas in Master Ensemble\n"
            f"DB Watcher   : Scanning alpha_db.json every 15s\n"
            f"Command Mode : Active Long-Polling Listener\n"
            f"Cadence      : 6-Hour Silent Keep-Alive Heartbeat\n"
            f"Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"═════════════════════════════════════════════"
        )
        return self.send_message(msg, disable_notification=True)

    def send_alpha_card(self, alpha_id: str, expr: str, sharpe: float, sortino: float, max_dd: float, ann_ret: float, turnover: float, correlation: float, accepted: bool) -> bool:
        """
        Sends standardized Project Chronos Performance Card:
        📈 [PROJECT CHRONOS] ACCEPTED ALPHA BACKTEST COMPLETE
        ═════════════════════════════════════════════════════
        Alpha ID: <alpha_id>
        Expression: <expression>
        Sharpe Ratio: <sharpe> | Sortino: <sortino>
        Max Drawdown: <max_dd>% | Ann. Return: <ann_ret>%
        Turnover: <turnover>%
        Correlation with Active Portfolio: <correlation>
        Ensemble Verdict: [ACCEPTED FOR LIVE ROUTING / REJECTED]
        ═════════════════════════════════════════════════════
        """
        verdict = "[ACCEPTED FOR LIVE ROUTING]" if accepted else "[REJECTED]"
        
        # Clean expression for concise Telegram display
        clean_expr = " ".join(line.strip() for line in expr.splitlines())
        if len(clean_expr) > 180:
            clean_expr = clean_expr[:177] + "..."

        msg = (
            f"📈 [PROJECT CHRONOS] ACCEPTED ALPHA BACKTEST COMPLETE\n"
            f"═════════════════════════════════════════════════════\n"
            f"Alpha ID: {alpha_id}\n"
            f"Expression: {clean_expr}\n"
            f"Sharpe Ratio: {sharpe:.2f} | Sortino: {sortino:.2f}\n"
            f"Max Drawdown: {max_dd:.2f}% | Ann. Return: {ann_ret:.2f}%\n"
            f"Turnover: {turnover:.2f}%\n"
            f"Correlation with Active Portfolio: {correlation:.3f}\n"
            f"Ensemble Verdict: {verdict}\n"
            f"═════════════════════════════════════════════"
        )
        return self.send_message(msg)

    def send_ensemble_summary(self, ensemble_metrics: dict, active_alpha_ids: list) -> bool:
        """Sends daily multi-alpha ensemble strategy summary."""
        ids_str = ", ".join(active_alpha_ids)
        msg = (
            f"🏛️ [PROJECT CHRONOS] ACTIVE ENSEMBLE PORTFOLIO SUMMARY\n"
            f"═════════════════════════════════════════════════════\n"
            f"Strategy: {ensemble_metrics.get('Strategy', 'Project_Chronos_Ensemble')}\n"
            f"Active Alpha Count: {len(active_alpha_ids)} ({ids_str})\n"
            f"Ensemble Sharpe   : {ensemble_metrics.get('SharpeRatio', 0.0):.2f}\n"
            f"Ensemble Sortino  : {ensemble_metrics.get('SortinoRatio', 0.0):.2f}\n"
            f"Max Drawdown      : {ensemble_metrics.get('MaxDrawdown', 0.0)*100:.2f}%\n"
            f"Annual Return     : {ensemble_metrics.get('AnnualizedReturn', 0.0)*100:.2f}%\n"
            f"Daily Turnover    : {ensemble_metrics.get('AverageTurnover', 0.0)*100:.2f}%\n"
            f"Portfolio Capital : ${ensemble_metrics.get('FinalEquity', 100000.0):,.2f}\n"
            f"System State      : Fully Detached Daemon (24/7 Auto-Pilot)\n"
            f"═════════════════════════════════════════════"
        )
        return self.send_message(msg)

if __name__ == "__main__":
    notifier = TelegramNotifier()
    print("Testing TelegramNotifier...")
    print(f"Bot Token prefix: {notifier.bot_token[:10]}...")
    print(f"Chat ID: {notifier.chat_id}")
    ok = notifier.send_message("⚡ [PROJECT CHRONOS] System initialized. Telegram Dispatcher active.")
    print("Sent test message:", ok)
