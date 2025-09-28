# File: DataHandler.py
# Author: Pasindu Pahasara Balasooriya Lekamlage
# Student ID: 104348348
# COS30018 – Intelligent Systems
# Project: Option C – Task C.2 (Data Processing 1)
#
# Features (updated):
#   • Load multi-feature OHLCV dataset from Yahoo Finance (keeps Volume)
#   • Exclude Dividends and Stock Splits
#   • Handle NaN values with DROP ONLY
#   • Split dataset into Train/Test by Ratio, Date, or Random
#   • Save/load dataset automatically as CSV (no user prompt)
#   • Optional MinMax scaling on prices and volume
#   • Defensive feature selection (skip missing columns gracefully)

from __future__ import annotations            # Allow forward type hints in older Python
import os                                     # File paths and directory handling
from dataclasses import dataclass             # Simple config container with defaults
from typing import Dict, Optional, Tuple, List # Type hints for readability

import numpy as np                            # Random generator for random split
import pandas as pd                           # DataFrame operations and CSV I/O
import yfinance as yf                         # Download market data from Yahoo Finance
from sklearn.preprocessing import MinMaxScaler # Normalise columns into 0–1 range


# -------------------- CONFIG CLASS --------------------
@dataclass
class ProcessorConfig:
    interval: str = "1d"                      # Candle interval (e.g., daily)
    auto_adjust: bool = False                 # Keep unadjusted OHLC; keep Adj Close separately
    split_method: str = "ratio"               # Default split strategy
    test_size: float = 0.2                    # Test ratio for ratio split
    split_date: Optional[str] = None          # Cutoff date for date split
    random_state: int = 42                    # Seed for reproducible random split
    random_low: float = 0.15                  # Lower bound for random test ratio
    random_high: float = 0.30                 # Upper bound for random test ratio
    scale: bool = False                       # Enable/disable MinMax scaling
    feature_cols: Optional[List[str]] = None  # Which columns to scale; None = scale all available OHLCV
    cache_dir: Optional[str] = "cache"        # Folder to store cached CSV
    cache_name: Optional[str] = None          # Filename for cache; None = auto-generate


# -------------------- MAIN PROCESSOR CLASS --------------------
class StockDataProcessor:
    """
    Pipeline: download -> clean -> cache -> split -> scale
    """

    def __init__(self, ticker: str, start: str, end: str, **kwargs):
        self.ticker = ticker                              # Store ticker symbol
        self.start_date = start                           # Store start date (YYYY-MM-DD)
        self.end_date = end                               # Store end date   (YYYY-MM-DD)
        self.config = ProcessorConfig(**kwargs)           # Build config from defaults + kwargs

        self.raw_data: Optional[pd.DataFrame] = None      # Full cleaned dataset (before split)
        self.train_set: pd.DataFrame = pd.DataFrame()     # Training subset
        self.test_set: pd.DataFrame = pd.DataFrame()      # Test subset
        self.scalers: Dict[str, MinMaxScaler] = {}        # Fitted scalers by column name

    # ---------- HELPERS ----------
    def _cache_path(self) -> Optional[str]:
        if not self.config.cache_dir:                     # If cache is disabled, return None
            return None
        os.makedirs(self.config.cache_dir, exist_ok=True) # Ensure cache directory exists
        fname = (                                         # Choose filename
            self.config.cache_name                        # Use custom name if provided
            or f"{self.ticker}_{self.start_date}_{self.end_date}_{self.config.interval}.csv"
        )
        return os.path.join(self.config.cache_dir, fname) # Full path to CSV

    @staticmethod
    def _prune_features(desired: Optional[List[str]], available: List[str]) -> List[str]:
        """Return only features that exist in 'available'; log missing ones."""
        if not desired:                                   # If user did not specify, use all available
            return available
        chosen = [c for c in desired if c in available]   # Keep only columns that exist
        missing = [c for c in desired if c not in available] # List the ones we cannot find
        if missing:
            print(f"[C.2] Skipping missing feature(s): {missing}")  # Inform the user
        if not chosen:
            # If nothing remains, fail early so the caller can fix their list
            raise ValueError("None of the requested feature_cols are available in the dataset.")
        return chosen

    # ---------- LOAD/SAVE CACHE ----------
    def _load_from_cache(self) -> Optional[pd.DataFrame]:
        path = self._cache_path()                          # Where the cache should be
        if path and os.path.exists(path):                  # If the file exists
            try:
                # Read CSV with index parsed as dates
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                if not df.empty:                           # Only accept non-empty cache
                    return df
            except Exception:
                # If cache is corrupted or unreadable, ignore and re-download
                return None
        return None                                        # No usable cache

    def _save_to_cache(self, df: pd.DataFrame) -> None:
        path = self._cache_path()                          # Get path
        if path:                                           # If caching is enabled
            df.to_csv(path)                                # Save cleaned data to disk

    # ---------- DOWNLOAD DATA ----------
    def _fetch_data(self) -> pd.DataFrame:
        cached = self._load_from_cache()                   # Try cache first
        if cached is not None:
            return cached
        try:
            t = yf.Ticker(self.ticker)                     # Create Yahoo Finance handle
            raw = t.history(                               # Download candles
                start=self.start_date,
                end=self.end_date,
                interval=self.config.interval,
                auto_adjust=self.config.auto_adjust,       # Keep Adj Close separate
                actions=False,                             # Do not include dividends/splits rows
            )
            if raw.empty:                                  # If no data returned, abort
                raise ValueError(
                    f"No data for {self.ticker} between {self.start_date} and {self.end_date}"
                )
            # Keep standard OHLC, Adj Close, and Volume if present
            keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in raw.columns]
            df = raw[keep].copy()                          # Work on a copy of just those columns
            return df
        except Exception as e:
            # Wrap any error so callers see a clean message
            raise RuntimeError(f"Failed to download data: {e}")

    # ---------- CLEAN DATA ----------
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()                                     # Avoid in-place edits on input
        # Standardise column names: lowercase, remove spaces (e.g., "Adj Close" -> "adjclose")
        df.columns = [c.lower().replace(" ", "") for c in df.columns]
        # Ensure the index is a proper DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")

        # If "adjclose" is missing but we have "close", create adjclose from close
        if "adjclose" not in df.columns and "close" in df.columns:
            df["adjclose"] = df["close"]

        # Convert numeric columns; ignore any that are not present
        for col in ("open", "high", "low", "close", "adjclose", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.sort_index(inplace=True)                        # Keep chronological order
        df = df.dropna()                                   # Drop any row with NaNs (assignment rule)
        return df                                          # Return cleaned DataFrame

    # ---------- SPLIT METHODS ----------
    def _split_by_ratio(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ts = float(self.config.test_size)                  # Read test ratio
        n = len(df)                                        # Total rows
        split_idx = int((1.0 - ts) * n)                    # Index where test starts
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()  # Train, Test

    def _split_by_date(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cutoff = pd.to_datetime(self.config.split_date)    # Parse cutoff string
        train_df = df.loc[df.index < cutoff].copy()        # Rows before cutoff
        test_df = df.loc[df.index >= cutoff].copy()        # Rows on/after cutoff
        return train_df, test_df

    def _split_randomly(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(self.config.random_state)    # Reproducible RNG
        # Draw a random test ratio within the configured range
        test_ratio = float(rng.uniform(self.config.random_low, self.config.random_high))
        # Mask True = goes to train, False = goes to test
        mask = rng.random(len(df)) > test_ratio
        train_df = df[mask].copy().sort_index()             # Keep chronological order after filtering
        test_df = df[~mask].copy().sort_index()
        return train_df, test_df

    def _split_data(self) -> None:
        m = self.config.split_method.lower()                # Which split method to use
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
        if not self.config.scale:                           # Skip if scaling disabled
            return {}
        # Candidate columns to scale (only those present will be used)
        candidates = ["open", "high", "low", "close", "adjclose", "volume"]
        available = [c for c in candidates if c in self.train_set.columns]
        # If user passed a list, prune it to the available set; else use all available
        cols = self._prune_features(self.config.feature_cols, available)

        self.train_set = self.train_set.copy()              # Avoid SettingWithCopy warnings
        self.test_set = self.test_set.copy()
        scalers: Dict[str, MinMaxScaler] = {}               # Collect fitted scalers

        for col in cols:
            scaler = MinMaxScaler()                         # 0–1 scaling per column
            self.train_set.loc[:, col] = scaler.fit_transform(self.train_set[[col]])
            if not self.test_set.empty and col in self.test_set:
                self.test_set.loc[:, col] = scaler.transform(self.test_set[[col]])
            scalers[col] = scaler                           # Store for inverse-transform later
        return scalers

    # ---------- PIPELINE ----------
    def run_pipeline(self) -> Dict[str, object]:
        print(f"[C.2] Processing {self.ticker} from {self.start_date} to {self.end_date}")  # Status
        fetched = self._fetch_data()                        # Download or load cached raw data
        self.raw_data = self._clean_data(fetched)           # Clean and standardise columns
        self._save_to_cache(self.raw_data)                  # Cache the cleaned dataset
        self._split_data()                                  # Split into training and test sets
        self.scalers = self._scale_data()                   # Optionally scale columns
        # Return all artifacts for downstream tasks (C.4/C.5)
        return {
            "df": self.raw_data,
            "train_df": self.train_set,
            "test_df": self.test_set,
            "scalers": self.scalers,
        }


# -------------------- CLI --------------------
def _validate_date(s: str) -> bool:
    try:
        pd.to_datetime(s, format="%Y-%m-%d")               # Strict date parsing
        return True
    except Exception:
        return False


def get_user_input() -> dict:
    print("Welcome to the Stock Data Processor!")          # Simple prompt banner
    ticker_symbol = "CBA.AX"                               # Ticker fixed by assignment
    print(f"Ticker: {ticker_symbol}")

    start = input("Start date (YYYY-MM-DD): ").strip()     # Ask for start date
    if not _validate_date(start):                          # Validate the format
        raise ValueError("Invalid start date format")
    end = input("End date (YYYY-MM-DD): ").strip()         # Ask for end date
    if not _validate_date(end):
        raise ValueError("Invalid end date format")

    print("\nChoose split method:")                        # Present split options
    print("1 -> Ratio (e.g., 80/20)")
    print("2 -> Date  (before/after cutoff)")
    print("3 -> Random (demo only)")
    choice = input("Enter 1, 2, or 3: ").strip()           # Read choice

    split_method = ""                                      # Will hold chosen method
    split_kwargs = {}                                      # Extra parameters per method
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
        print("Random split will assign about 15–30% to test")
    else:
        raise ValueError("Invalid choice. Enter 1, 2, or 3")

    scale = input("Apply MinMax scaling? (y/N): ").strip().lower() == "y"  # Enable scaling?
    feature_cols = None                                                    # Optional list to scale
    if scale:
        cols = input("Columns to scale (comma-separated, blank=all OHLCV): ").strip()
        if cols:
            feature_cols = [c.strip().lower() for c in cols.split(",")]     # Normalise names

    # Package user choices into a kwargs dict for StockDataProcessor
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
        cfg = get_user_input()                             # Ask user for all settings
        proc = StockDataProcessor(**cfg)                   # Build processor with those settings
        out = proc.run_pipeline()                          # Execute the full pipeline

        train_df = out["train_df"]                         # Unpack outputs
        test_df = out["test_df"]

        print("\n--- Processing Complete ---")             # Summary information
        print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
        if not train_df.empty:
            print("Train period:", train_df.index.min().date(), "->", train_df.index.max().date())
            print("\nTrain head:\n", train_df.head(5))
        if not test_df.empty:
            print("Test period:", test_df.index.min().date(), "->", test_df.index.max().date())
            print("\nTest head:\n", test_df.head(5))
        if cfg.get("scale"):
            print("\nScaled columns:", list(out["scalers"].keys()))

        cp = proc._cache_path()                            # Show where cache lives
        if cp:
            print(f"\nCached CSV saved at: {cp}")

        print("\nAvailable columns:", train_df.columns.tolist())  # Final check of columns

    except Exception as e:
        print(f"\nError: {e}")                             # Friendly error message