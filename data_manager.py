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
    # Information Technology (24)
    "AAPL", "MSFT", "NVDA", "AVGO", "CSCO", "AMD", "INTC", "CRM", "ORCL", "ADBE",
    "QCOM", "TXN", "AMAT", "IBM", "NOW", "INTU", "MU", "LRCX", "ADI", "PANW",
    "KLAC", "SNPS", "CDNS", "CRWD",
    # Communication Services (9)
    "GOOGL", "META", "DIS", "NFLX", "CMCSA", "VZ", "T", "CHTR", "TMUS",
    # Consumer Discretionary (12)
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "BKNG", "TJX", "TGT", "F", "GM",
    # Consumer Staples (11)
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "KMB", "STZ",
    # Financials (15)
    "JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "BLK", "C", "AXP",
    "SCHW", "CB", "MMC", "PGR", "USB",
    # Healthcare (17)
    "UNH", "JNJ", "ABBV", "MRK", "LLY", "PFE", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "CVS", "GILD", "ISRG", "CI", "REGN", "VRTX",
    # Energy (9)
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY",
    # Industrials (10)
    "CAT", "GE", "UNP", "HON", "RTX", "BA", "DE", "LMT", "UPS", "ADP",
    # Materials (5)
    "LIN", "APD", "SHW", "FCX", "NEM",
    # Utilities (4)
    "NEE", "SO", "DUK", "SRE",
    # Real Estate (4)
    "PLD", "AMT", "EQIX", "CCI"
]

SECTOR_MAP = {
    # Information Technology
    "AAPL": "Information Technology", "MSFT": "Information Technology", "NVDA": "Information Technology",
    "AVGO": "Information Technology", "CSCO": "Information Technology", "AMD": "Information Technology",
    "INTC": "Information Technology", "CRM": "Information Technology", "ORCL": "Information Technology",
    "ADBE": "Information Technology", "QCOM": "Information Technology", "TXN": "Information Technology",
    "AMAT": "Information Technology", "IBM": "Information Technology", "NOW": "Information Technology",
    "INTU": "Information Technology", "MU": "Information Technology", "LRCX": "Information Technology",
    "ADI": "Information Technology", "PANW": "Information Technology", "KLAC": "Information Technology",
    "SNPS": "Information Technology", "CDNS": "Information Technology", "CRWD": "Information Technology",
    # Communication Services
    "GOOGL": "Communication Services", "META": "Communication Services", "DIS": "Communication Services",
    "NFLX": "Communication Services", "CMCSA": "Communication Services", "VZ": "Communication Services",
    "T": "Communication Services", "CHTR": "Communication Services", "TMUS": "Communication Services",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary", "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "LOW": "Consumer Discretionary", "BKNG": "Consumer Discretionary", "TJX": "Consumer Discretionary",
    "TGT": "Consumer Discretionary", "F": "Consumer Discretionary", "GM": "Consumer Discretionary",
    # Consumer Staples
    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "COST": "Consumer Staples", "WMT": "Consumer Staples", "PM": "Consumer Staples",
    "MO": "Consumer Staples", "MDLZ": "Consumer Staples", "CL": "Consumer Staples",
    "KMB": "Consumer Staples", "STZ": "Consumer Staples",
    # Financials
    "JPM": "Financials", "V": "Financials", "MA": "Financials", "BAC": "Financials",
    "WFC": "Financials", "MS": "Financials", "GS": "Financials", "BLK": "Financials",
    "C": "Financials", "AXP": "Financials", "SCHW": "Financials", "CB": "Financials",
    "MMC": "Financials", "PGR": "Financials", "USB": "Financials",
    # Healthcare
    "UNH": "Healthcare", "JNJ": "Healthcare", "ABBV": "Healthcare", "MRK": "Healthcare",
    "LLY": "Healthcare", "PFE": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare",
    "DHR": "Healthcare", "BMY": "Healthcare", "AMGN": "Healthcare", "CVS": "Healthcare",
    "GILD": "Healthcare", "ISRG": "Healthcare", "CI": "Healthcare", "REGN": "Healthcare",
    "VRTX": "Healthcare",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "EOG": "Energy", "MPC": "Energy", "PSX": "Energy", "VLO": "Energy", "OXY": "Energy",
    # Industrials
    "CAT": "Industrials", "GE": "Industrials", "UNP": "Industrials", "HON": "Industrials",
    "RTX": "Industrials", "BA": "Industrials", "DE": "Industrials", "LMT": "Industrials",
    "UPS": "Industrials", "ADP": "Industrials",
    # Materials
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials", "FCX": "Materials", "NEM": "Materials",
    # Utilities
    "NEE": "Utilities", "SO": "Utilities", "DUK": "Utilities", "SRE": "Utilities",
    # Real Estate
    "PLD": "Real Estate", "AMT": "Real Estate", "EQIX": "Real Estate", "CCI": "Real Estate"
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

    def fetch_and_ingest(self, symbols=None, start_date="2021-01-01", end_date="2024-01-01"):
        if symbols is None:
            symbols = DEFAULT_UNIVERSE
            
        print(f"[Chronos DataManager] Ingesting {len(symbols)} symbols from {start_date} to {end_date}...")
        results = {}
        
        # 1. Batch download with yf.download for high throughput
        batch_df = None
        try:
            print(f"  -> Batch downloading {len(symbols)} tickers via multithreaded yfinance...")
            batch_df = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                threads=True
            )
        except Exception as e:
            print(f"  [!] Batch download failed ({e}), falling back to iterative retrieval...")

        # 2. Process each symbol
        for sym in symbols:
            df = None
            try:
                if batch_df is not None and not batch_df.empty:
                    if isinstance(batch_df.columns, pd.MultiIndex):
                        cols = batch_df.columns
                        # Level 0 is metric, Level 1 is ticker OR vice versa
                        if sym in cols.levels[1]:
                            sub_dict = {}
                            for col_name in ['Open', 'High', 'Low', 'Close', 'Volume']:
                                if (col_name, sym) in cols:
                                    sub_dict[col_name] = batch_df[(col_name, sym)]
                            if sub_dict:
                                df = pd.DataFrame(sub_dict).dropna()
                        elif sym in cols.levels[0]:
                            df = batch_df[sym][['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

                if df is None or df.empty or len(df) < 50:
                    # Fallback to single-ticker history
                    ticker = yf.Ticker(sym)
                    df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
                    if not df.empty:
                        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

                if df is None or df.empty or len(df) < 50:
                    print(f"  [!] Warning: Insufficient data for {sym} (rows: {len(df) if df is not None else 0})")
                    continue

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
        
        # 1. Primary workspace location
        cache_file = os.path.join(self.workspace_path, "data", "consolidated_panel.csv")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        panel.to_csv(cache_file, index=False)
        print(f"  [OK] Consolidated universe panel saved to: {cache_file}")

        # 2. Synchronize to root data/ directories for fast access by all modules
        root_dir = os.path.dirname(os.path.abspath(self.workspace_path))
        data_dir = os.path.join(root_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        root_cache = os.path.join(data_dir, "consolidated_panel.csv")
        universe_cache = os.path.join(data_dir, "universe_panel.csv")
        panel.to_csv(root_cache, index=False)
        panel.to_csv(universe_cache, index=False)
        print(f"  [OK] Root cache mirrors updated: {root_cache}, {universe_cache}")

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
