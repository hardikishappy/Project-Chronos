"""
Project Chronos - Telegram Dispatcher
Sends automated quantitative backtest cards and ensemble daily reports
to Telegram using credentials migrated into .env.
"""

import os
import json
import urllib.request
import urllib.parse
from datetime import datetime

class TelegramNotifier:
    def __init__(self, env_path=None):
        if env_path is None:
            env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        
        self.bot_token = None
        self.chat_id = None
        self.fallback_token = None
        
        self._load_env(env_path)

    def _load_env(self, env_path):
        if os.path.exists(env_path):
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
                        elif k == "FALLBACK_TELEGRAM_BOT_TOKEN":
                            self.fallback_token = v
                            
        # Environment variable overrides
        self.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", self.bot_token)
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", self.chat_id)
        self.fallback_token = os.environ.get("FALLBACK_TELEGRAM_BOT_TOKEN", self.fallback_token)

    def send_message(self, text: str, target_chat_id: str = None, disable_notification: bool = False) -> bool:
        """Sends a text message to Telegram with automatic token fallback."""
        cid = target_chat_id or self.chat_id
        if not cid:
            print("[Telegram] Error: Missing TELEGRAM_CHAT_ID")
            return False

        tokens_to_try = [t for t in [self.bot_token, self.fallback_token] if t]
        if not tokens_to_try:
            print("[Telegram] Error: No bot tokens available")
            return False

        for token in tokens_to_try:
            try:
                url = f"https://api.telegram.org/bot{token}/sendMessage"
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
                print(f"[Telegram] Warning: Failed sending with token ({e}), trying fallback...")

        print("[Telegram] Error: Failed to dispatch message via all tokens")
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
            {"command": "portfolio", "description": "Master 16-Alpha Ensemble stats"},
            {"command": "alpaca", "description": "Live Alpaca account equity, cash & buying power"},
            {"command": "positions", "description": "Current open market positions & live PnL"},
            {"command": "trade", "description": "Trigger an immediate on-demand rebalance"},
            {"command": "status", "description": "Background daemon health & uptime"}
        ]
        tokens_to_try = [t for t in [self.bot_token, self.fallback_token] if t]
        if not tokens_to_try:
            return False

        for token in tokens_to_try:
            try:
                url = f"https://api.telegram.org/bot{token}/setMyCommands"
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

    def get_updates(self, offset: int = None, timeout: int = 15) -> list:
        """Fetches incoming updates from Telegram using HTTP long-polling."""
        tokens_to_try = [t for t in [self.bot_token, self.fallback_token] if t]
        if not tokens_to_try:
            return []

        for token in tokens_to_try:
            try:
                url = f"https://api.telegram.org/bot{token}/getUpdates?timeout={timeout}"
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
            f"═════════════════════════════════════════════════════"
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
            f"═════════════════════════════════════════════════════"
        )
        return self.send_message(msg)

if __name__ == "__main__":
    notifier = TelegramNotifier()
    print("Testing TelegramNotifier...")
    ok = notifier.send_message("⚡ [PROJECT CHRONOS] System initialized. Telegram Dispatcher active.")
    print("Sent test message:", ok)
