"""
Project Chronos - Data Manager & Ingestion Layer
Pulls daily OHLCV bars using yfinance for a liquid universe,
and formats data directly into QuantConnect LEAN's local filesystem layout.
Zero-cost, completely offline.
"""

import os
import sys
import zipfile
import io
import argparse
import pandas as pd
try:
    import yfinance as yf
except Exception:
    yf = None
from datetime import datetime

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "UNH", "XOM",
    "JNJ", "V", "PG", "MA", "HD", "BAC", "ABBV", "KO", "PEP", "MRK",
    "CVX", "COST", "AVGO", "CSCO", "WMT", "MCD", "DIS", "AMD", "NFLX", "INTC"
]

SECTOR_MAP = {
    "AAPL": "Information Technology", "MSFT": "Information Technology", "GOOGL": "Communication Services",
    "AMZN": "Consumer Discretionary", "NVDA": "Information Technology", "META": "Communication Services",
    "TSLA": "Consumer Discretionary", "JPM": "Financials", "UNH": "Healthcare", "XOM": "Energy",
    "JNJ": "Healthcare", "V": "Financials", "PG": "Consumer Staples", "MA": "Financials",
    "HD": "Consumer Discretionary", "BAC": "Financials", "ABBV": "Healthcare", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "MRK": "Healthcare", "CVX": "Energy", "COST": "Consumer Staples",
    "AVGO": "Information Technology", "CSCO": "Information Technology", "WMT": "Consumer Staples",
    "MCD": "Consumer Discretionary", "DIS": "Communication Services", "AMD": "Information Technology",
    "NFLX": "Communication Services", "INTC": "Information Technology"
}

class LeanDataManager:
    def __init__(self, workspace_path=None):
        if workspace_path is None:
            self.workspace_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chronos_lean_workspace")
        else:
            self.workspace_path = workspace_path
            
        self.data_dir = os.path.join(self.workspace_path, "data", "equity", "usa")
        self.daily_dir = os.path.join(self.data_dir, "daily")
        self.map_files_dir = os.path.join(self.data_dir, "map_files")
        self.factor_files_dir = os.path.join(self.data_dir, "factor_files")
        
        for d in [self.daily_dir, self.map_files_dir, self.factor_files_dir]:
            os.makedirs(d, exist_ok=True)

    def fetch_and_ingest(self, symbols=None, start_date="2020-01-01", end_date="2024-01-01"):
        if symbols is None:
            symbols = DEFAULT_UNIVERSE
            
        print(f"[Chronos DataManager] Ingesting {len(symbols)} symbols from {start_date} to {end_date}...")
        results = {}
        
        for sym in symbols:
            try:
                print(f"  -> Downloading {sym}...")
                ticker = yf.Ticker(sym)
                df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
                
                if df.empty or len(df) < 50:
                    print(f"  [!] Warning: Insufficient data for {sym} (rows: {len(df)})")
                    continue
                    
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                self._save_lean_equity_bar(sym, df)
                self._create_lean_auxiliary_files(sym, df)
                results[sym] = df
                print(f"  [OK] {sym}: {len(df)} daily bars stored in LEAN format.")
            except Exception as e:
                print(f"  [X] Failed downloading {sym}: {e}")
                
        # Also store consolidated dataframe for fast local simulation
        self._save_consolidated_cache(results)
        print(f"[Chronos DataManager] Ingestion complete. {len(results)} symbols ready for backtesting.")
        return results

    def _save_lean_equity_bar(self, symbol: str, df: pd.DataFrame):
        """Formats and writes LEAN daily bar zip archive."""
        sym_lower = symbol.lower()
        zip_path = os.path.join(self.daily_dir, f"{sym_lower}.zip")
        csv_filename = f"{sym_lower}.csv"
        
        # LEAN daily bar format: YYYYMMDD 00:00,Open*10000,High*10000,Low*10000,Close*10000,Volume
        lines = []
        for dt, row in df.iterrows():
            dt_str = dt.strftime("%Y%m%d 00:00")
            o = int(round(row['Open'] * 10000))
            h = int(round(row['High'] * 10000))
            l = int(round(row['Low'] * 10000))
            c = int(round(row['Close'] * 10000))
            v = int(round(row['Volume']))
            lines.append(f"{dt_str},{o},{h},{l},{c},{v}")
            
        csv_content = "\n".join(lines) + "\n"
        
        # Write zip archive containing <symbol>.csv
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(csv_filename, csv_content)
            
        # Also save decimal CSV for straightforward inspection and fast numpy loading
        uncompressed_path = os.path.join(self.daily_dir, f"{sym_lower}_raw.csv")
        df.to_csv(uncompressed_path)

    def _create_lean_auxiliary_files(self, symbol: str, df: pd.DataFrame):
        """Generates LEAN map_files and factor_files required by the engine."""
        sym_lower = symbol.lower()
        
        # Map file: YYYYMMDD,symbol,1
        first_date = df.index[0].strftime("%Y%m%d")
        map_path = os.path.join(self.map_files_dir, f"{sym_lower}.csv")
        map_content = f"{first_date},{sym_lower},1\n20501231,{sym_lower},1\n"
        with open(map_path, "w", encoding="utf-8") as f:
            f.write(map_content)
            
        # Factor file: Date,PriceFactor,SplitFactor,ReferencePrice
        factor_path = os.path.join(self.factor_files_dir, f"{sym_lower}.csv")
        factor_content = f"{first_date},1.0,1.0,0.0\n20501231,1.0,1.0,0.0\n"
        with open(factor_path, "w", encoding="utf-8") as f:
            f.write(factor_content)

    def _save_consolidated_cache(self, results: dict):
        """Saves a multi-index consolidated panel for rapid vectorized simulation."""
        if not results:
            return
        records = []
        for sym, df in results.items():
            sub = df.copy()
            sub['symbol'] = sym
            sub['sector'] = SECTOR_MAP.get(sym, 'Technology')
            records.append(sub)
        panel = pd.concat(records)
        panel.reset_index(names='date', inplace=True)
        # Ensure date is tz-naive datetime
        panel['date'] = pd.to_datetime(panel['date']).dt.tz_localize(None)
        cache_file = os.path.join(self.workspace_path, "data", "consolidated_panel.csv")
        panel.to_csv(cache_file, index=False)
        print(f"  [OK] Consolidated universe panel saved to: {cache_file}")

    def load_universe_panel(self):
        cache_file = os.path.join(self.workspace_path, "data", "consolidated_panel.csv")
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Consolidated data not found at {cache_file}. Run fetch_and_ingest() first.")
        df = pd.read_csv(cache_file, parse_dates=['date'])
        return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chronos Free Data Ingestion Layer")
    parser.add_argument("--symbols", type=str, default="", help="Comma separated tickers (e.g. AAPL,MSFT)")
    parser.add_argument("--start", type=str, default="2021-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-01-01", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else DEFAULT_UNIVERSE
    dm = LeanDataManager()
    dm.fetch_and_ingest(symbols=syms, start_date=args.start, end_date=args.end)
