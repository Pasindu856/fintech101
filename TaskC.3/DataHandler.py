# File: DataHandler.py
# Author: Pasindu Pahasara Balasooriya Lekamlage
# Student ID: 104348348
# COS30018 – Intelligent Systems
# Project: Option C – Task C.2 (Data Processing 1)
#
# Features implemented for Task C.2:
#   • Load multi-feature OHLCV dataset from Yahoo Finance
#   • Do NOT keep Volume, Dividends, Stock Splits (per instructions)
#   • Handle NaN values with DROP ONLY
#   • Split dataset into Train/Test by Ratio, Date, or Random
#   • Save/load dataset automatically as CSV (no user prompt)
#   • Optional MinMax scaling on price columns
#   • Clear instructions to user for input

# -------------------- IMPORTS --------------------
from __future__ import annotations             # Ensures forward references work in type hints
import os                                     # Used to handle file and directory paths
from dataclasses import dataclass             # For creating configuration with default values
from typing import Dict, Optional, Tuple      # For type hinting dictionaries and tuples

import numpy as np                            # For numerical operations and random split
import pandas as pd                           # For handling tabular data (DataFrames)
import yfinance as yf                         # For downloading stock data from Yahoo Finance
from sklearn.preprocessing import MinMaxScaler # For feature scaling (0–1 range)

# -------------------- CONFIG CLASS --------------------
@dataclass
class ProcessorConfig:
    interval: str = "1d"              # Data interval ("1d" daily, "1wk" weekly, etc.)
    auto_adjust: bool = False         # Do not auto-adjust prices (we keep Adj Close separately)
    split_method: str = "ratio"       # Default split method is ratio
    test_size: float = 0.2            # Test set size for ratio split (20%)
    split_date: Optional[str] = None  # Date for date-based split
    random_state: int = 42            # Seed for reproducible random splits
    random_low: float = 0.15          # Minimum random test ratio
    random_high: float = 0.30         # Maximum random test ratio
    scale: bool = False               # Default: do not scale features
    feature_cols: Optional[list[str]] = None  # If None, automatically choose OHLC/adjclose
    cache_dir: Optional[str] = "cache"        # Directory for cached CSV file
    cache_name: Optional[str] = None          # File name for cached CSV

# -------------------- MAIN PROCESSOR CLASS --------------------
class StockDataProcessor:
    """
    Handles entire pipeline: download -> clean -> cache -> split -> scale
    """

    def __init__(self, ticker: str, start: str, end: str, **kwargs):
        self.ticker = ticker                         # Save ticker symbol (e.g., "CBA.AX")
        self.start_date = start                      # Save start date string
        self.end_date = end                          # Save end date string
        self.config = ProcessorConfig(**kwargs)      # Load config with defaults (overridable)

        self.raw_data: Optional[pd.DataFrame] = None # Placeholder for full cleaned dataset
        self.train_set: pd.DataFrame = pd.DataFrame()# Placeholder for training set
        self.test_set: pd.DataFrame = pd.DataFrame() # Placeholder for testing set
        self.scalers: Dict[str, MinMaxScaler] = {}   # Placeholder for fitted scalers

    # ---------- CACHE PATH ----------
    def _cache_path(self) -> Optional[str]:
        if not self.config.cache_dir:                # If caching disabled, return None
            return None
        os.makedirs(self.config.cache_dir, exist_ok=True) # Ensure cache directory exists
        fname = (self.config.cache_name              # If custom file name provided, use it
                 or f"{self.ticker}_{self.start_date}_{self.end_date}_{self.config.interval}.csv")
        return os.path.join(self.config.cache_dir, fname) # Build full file path

    # ---------- LOAD FROM CACHE ----------
    def _load_from_cache(self) -> Optional[pd.DataFrame]:
        path = self._cache_path()                    # Get cache path
        if path and os.path.exists(path):            # If file exists
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True) # Load CSV with Date index
                if not df.empty:                     # If file not empty, return dataframe
                    return df
            except Exception:                        # If error occurs, ignore and re-download
                return None
        return None                                  # No cache found

    # ---------- SAVE TO CACHE ----------
    def _save_to_cache(self, df: pd.DataFrame) -> None:
        path = self._cache_path()                    # Get cache path
        if path:                                     # If caching enabled
            df.to_csv(path)                          # Save dataframe to CSV

    # ---------- DOWNLOAD DATA ----------
    def _fetch_data(self) -> pd.DataFrame:
        cached = self._load_from_cache()             # Try to load from CSV cache
        if cached is not None:                       # If cache found, return cached data
            return cached
        try:
            t = yf.Ticker(self.ticker)               # Create Yahoo Finance Ticker object
            raw = t.history(                         # Download history
                start=self.start_date,
                end=self.end_date,
                interval=self.config.interval,
                auto_adjust=self.config.auto_adjust,
                actions=False,                       # Do NOT include Dividends or Stock Splits
            )
            if raw.empty:                            # If no data, raise error
                raise ValueError(f"No data for {self.ticker} between {self.start_date} and {self.end_date}")
            keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in raw.columns] # Keep OHLC/Adj Close
            df = raw[keep].copy()                    # Copy only required columns (drop Volume)
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to download data: {e}")

    # ---------- CLEAN DATA ----------
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()                               # Copy to avoid modifying original
        df.columns = [c.lower().replace(" ", "") for c in df.columns] # Standardise column names
        if not isinstance(df.index, pd.DatetimeIndex): # Ensure index is datetime
            df.index = pd.to_datetime(df.index, errors="coerce")
        if "adjclose" not in df.columns and "close" in df.columns: # Ensure adjclose column exists
            df["adjclose"] = df["close"]
        for col in ("open", "high", "low", "close", "adjclose"):   # Convert columns to numeric
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.sort_index(inplace=True)                 # Sort chronologically
        df = df.dropna()                            # Drop rows with NaNs (per requirement)
        return df

    # ---------- SPLIT METHODS ----------
    def _split_by_ratio(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ts = float(self.config.test_size)           # Read test size ratio
        n = len(df)                                 # Total rows
        split_idx = int((1.0 - ts) * n)             # Index for train/test boundary
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    def _split_by_date(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cutoff = pd.to_datetime(self.config.split_date) # Convert split_date to datetime
        train_df = df.loc[df.index < cutoff].copy() # Rows before cutoff = train
        test_df = df.loc[df.index >= cutoff].copy() # Rows after cutoff = test
        return train_df, test_df

    def _split_randomly(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(self.config.random_state) # Reproducible random generator
        test_ratio = float(rng.uniform(self.config.random_low, self.config.random_high)) # Pick random test ratio
        mask = rng.random(len(df)) > test_ratio    # Random mask True=train, False=test
        train_df = df[mask].copy().sort_index()    # Train set
        test_df = df[~mask].copy().sort_index()    # Test set
        return train_df, test_df

    def _split_data(self) -> None:
        m = self.config.split_method.lower()        # Read split method
        if m == "ratio":
            self.train_set, self.test_set = self._split_by_ratio(self.raw_data)
        elif m == "date":
            self.train_set, self.test_set = self._split_by_date(self.raw_data)
        elif m == "random":
            self.train_set, self.test_set = self._split_randomly(self.raw_data)
        else:
            raise ValueError("split_method must be: ratio | date | random")

    # ---------- SCALING ----------
    def _scale_data(self) -> Dict[str, MinMaxScaler]:
        if not self.config.scale:                   # If scaling disabled, return empty dict
            return {}
        candidates = ["open", "high", "low", "close", "adjclose"] # Only these columns allowed
        cols = self.config.feature_cols or [c for c in candidates if c in self.train_set.columns]
        self.train_set = self.train_set.copy()      # Work on copies to avoid warnings
        self.test_set = self.test_set.copy()
        scalers: Dict[str, MinMaxScaler] = {}
        for col in cols:                            # For each feature column
            scaler = MinMaxScaler()                 # Create MinMax scaler
            self.train_set.loc[:, col] = scaler.fit_transform(self.train_set[[col]]) # Fit+transform train
            if not self.test_set.empty and col in self.test_set:
                self.test_set.loc[:, col] = scaler.transform(self.test_set[[col]])   # Transform test
            scalers[col] = scaler                   # Save scaler for later use
        return scalers

    # ---------- PIPELINE ----------
    def run_pipeline(self) -> Dict[str, object]:
        print(f"[C.2] Processing {self.ticker} from {self.start_date} to {self.end_date} …")
        fetched = self._fetch_data()                # Download or load cache
        self.raw_data = self._clean_data(fetched)   # Clean data
        self._save_to_cache(self.raw_data)          # Save cleaned dataset to CSV automatically
        self._split_data()                          # Split into train/test
        self.scalers = self._scale_data()           # Apply scaling if enabled
        return {"df": self.raw_data, "train_df": self.train_set, "test_df": self.test_set, "scalers": self.scalers}

# -------------------- CLI --------------------
def _validate_date(s: str) -> bool:
    try:
        pd.to_datetime(s, format="%Y-%m-%d")       # Check if string can be parsed as date
        return True
    except Exception:
        return False

def get_user_input() -> dict:
    print("Welcome to the Stock Data Processor! 📊")
    ticker_symbol = "CBA.AX"                       # Ticker fixed as per assignment
    print(f"Ticker: {ticker_symbol}")

    start = input("Start date (YYYY-MM-DD): ").strip() # Ask user for start date
    if not _validate_date(start):                  # Validate format
        raise ValueError("Invalid start date format")
    end = input("End date (YYYY-MM-DD): ").strip() # Ask user for end date
    if not _validate_date(end):
        raise ValueError("Invalid end date format")

    print("\nChoose split method:")                # Prompt for split method
    print("1 -> Ratio (e.g., 80/20)")
    print("2 -> Date  (before/after cutoff)")
    print("3 -> Random (demo only)")
    choice = input("Enter 1, 2, or 3: ").strip()

    split_method = ""
    split_kwargs = {}
    if choice == "1":
        split_method = "ratio"
        ratio_str = input("Test ratio (e.g., 0.2): ").strip()
        split_kwargs["test_size"] = float(ratio_str) if ratio_str else 0.2
    elif choice == "2":
        split_method = "date"
        sd = input("Cutoff date for TEST (YYYY-MM-DD): ").strip()
        if not _validate_date(sd):
            raise ValueError("Invalid split date format")
        split_kwargs["split_date"] = sd
    elif choice == "3":
        split_method = "random"
        print("Random split will assign ~15-30% to test")
    else:
        raise ValueError("Invalid choice. Enter 1, 2, or 3")

    scale = input("Apply MinMax scaling? (y/N): ").strip().lower() == "y" # Ask if scaling needed
    feature_cols = None
    if scale:
        cols = input("Columns to scale (comma-separated, blank=all): ").strip()
        if cols:
            feature_cols = [c.strip().lower() for c in cols.split(",")]

    return {
        "ticker": ticker_symbol,
        "start": start,
        "end": end,
        "split_method": split_method,
        "scale": scale,
        "feature_cols": feature_cols,
        **split_kwargs,
    }

# -------------------- ENTRYPOINT --------------------
if __name__ == "__main__":
    try:
        cfg = get_user_input()                      # Collect user inputs
        proc = StockDataProcessor(**cfg)            # Create processor with config
        out = proc.run_pipeline()                   # Run pipeline

        train_df = out["train_df"]                  # Extract train DataFrame
        test_df = out["test_df"]                    # Extract test DataFrame

        print("\n--- Processing Complete ---")
        print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
        if not train_df.empty:
            print("Train period:", train_df.index.min().date(), "→", train_df.index.max().date())
            print("\nTrain head:\n", train_df.head(5))
        if not test_df.empty:
            print("Test period:", test_df.index.min().date(), "→", test_df.index.max().date())
            print("\nTest head:\n", test_df.head(5))
        if cfg.get("scale"):
            print("\nScaled columns:", list(out["scalers"].keys()))

        cp = proc._cache_path()                     # Show cache file path
        if cp:
            print(f"\nCached CSV saved at: {cp}")

    except Exception as e:
        print(f"\nError: {e}")                      # Print any error nicely