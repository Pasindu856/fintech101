# File: taskc3.py
# COS30018 – Option C – Task C.3
# This script will create:
#   • Candlestick chart where each candle can represent n trading days
#   • Boxplot of closing prices grouped into n-day blocks
# Data comes from Task C.2 DataHandler

import numpy as np                # for grouping rows into blocks
import pandas as pd               # for handling DataFrame (tables of stock data)
import matplotlib.pyplot as plt   # for making boxplots
import mplfinance as mpf          # for making candlestick charts

# -------- Helper functions --------

def _prep(df: pd.DataFrame) -> pd.DataFrame:
    # make a copy of the input DataFrame so we don’t change the original
    df = df.copy()
    # ensure index is DateTime (required for plotting with mplfinance)
    df.index = pd.to_datetime(df.index, errors="coerce")
    # rename columns to Title-case so mplfinance recognises them
    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adjclose": "Adj Close",
        "volume": "Volume"
    })
    # define which columns must be there
    need = {"Open", "High", "Low", "Close"}
    # drop rows that don’t have a valid date
    df = df.loc[~df.index.isna()]
    # drop rows where price columns have NaN
    df = df.dropna(subset=list(need))
    # sort by date to keep order
    return df.sort_index()

def _group_n_days(df: pd.DataFrame, n: int) -> pd.DataFrame:
    # if n < 1 it is invalid
    if n < 1:
        raise ValueError("n must be >= 1")
    # if not enough rows to form a block raise error
    if len(df) < n:
        raise ValueError(f"Need at least {n} rows")
    # if n == 1 just return original data
    if n == 1:
        return df.copy()

    # create groups of size n using integer division
    k = np.arange(len(df)) // n
    # group the DataFrame by these block numbers
    g = df.groupby(k)

    # aggregate OHLC values: first Open, max High, min Low, last Close
    out = pd.DataFrame({
        "Open":  g["Open"].first(),
        "High":  g["High"].max(),
        "Low":   g["Low"].min(),
        "Close": g["Close"].last(),
    })
    # if Volume column exists, sum it across the block
    if "Volume" in df.columns:
        out["Volume"] = g["Volume"].sum()

    # use the last date of each block as index label
    last_dates = g.apply(lambda x: x.index[-1])
    out.index = pd.DatetimeIndex(last_dates.values, name=df.index.name)
    # drop rows with NaT index (just in case)
    out = out.loc[~out.index.isna()]
    return out

# -------- Plot functions --------

def candlestick_plot(ohlc_df: pd.DataFrame, n: int, *, ticker: str, style: str = "charles"):
    # prepare data (rename columns, sort, drop NaNs)
    data = _group_n_days(_prep(ohlc_df), n)
    # plot candlestick chart with mplfinance
    mpf.plot(
        data,                              # aggregated OHLC data
        type="candle",                     # candlestick type
        volume=("Volume" in data.columns), # add volume panel only if volume column exists
        style=style,                       # chart style (e.g. yahoo, charles, binance)
        title=f"{ticker} – {n}-Day Candlestick Chart", # chart title
        figsize=(12, 6),                   # figure size
        ylabel="Trading Price"             # label for y-axis
    )

def boxplot_blocks(df: pd.DataFrame, n: int, *, column: str = "Close", ticker: str = ""):
    # select only the chosen column (e.g. Close price) and clean it
    s = _prep(df)[column].dropna()
    # check enough rows exist for at least one block
    if len(s) < n:
        raise ValueError(f"Need at least {n} rows")

    # create groups of size n
    k = np.arange(len(s)) // n
    blocks, labels = [], []
    # loop over each group
    for _, g in s.groupby(k):
        if g.empty:
            continue
        # add the close prices for this block to blocks list
        blocks.append(g.values)
        # get the last date of this block
        t = g.index[-1]
        # if valid date, format it as string for x-axis label
        labels.append("" if pd.isna(t) else t.strftime("%Y-%m-%d"))

    # create figure for boxplot
    plt.figure(figsize=(12, 6))
    # draw boxplot, show mean marker
    plt.boxplot(blocks, showmeans=True)
    ax = plt.gca()
    # reduce label clutter: show every step-th label
    step = max(1, len(labels)//12)
    ax.set_xticklabels([lab if i % step == 0 else "" for i, lab in enumerate(labels)], rotation=45, ha="right")
    # set title of plot
    plt.title(f"{ticker} – {column} Distribution in {n}-Day Blocks")
    # set label for y-axis
    plt.ylabel(f"{column} Price")
    # add horizontal gridlines
    plt.grid(True, axis="y", alpha=0.3)
    # adjust layout to avoid cut-off labels
    plt.tight_layout()
    # show the plot
    plt.show()

# -------- Main section --------

if __name__ == "__main__":
    # set ticker symbol and date range
    ticker = "CBA.AX"
    start  = "2018-01-01"
    end    = "2021-12-31"

    # try Task C.2 function style first
    try:
        from DataHandler import Loading_and_processing
        # call Task C.2 function to get train, test, full dataset
        train_df, test_df, full_df, _ = Loading_and_processing(
            ticker=ticker, start=start, end=end,
            split_method="ratio", test_size=0.2,
            scale=True, feature_cols=["open", "high", "low", "close", "volume"]
        )
    except Exception:
        # if function not found, use class style
        from DataHandler import StockDataProcessor
        # create StockDataProcessor object with required parameters
        proc = StockDataProcessor(
            ticker=ticker, start=start, end=end,
            split_method="ratio", test_size=0.2, scale=True,
            feature_cols=["open", "high", "low", "close", "adjclose"]
        )
        # run pipeline to get data dictionary
        result = proc.run_pipeline()
        # take the full DataFrame
        full_df = result["df"]

    # ask user for block size
    n = int(input("Enter trading days per block: ").strip())

    # create boxplot
    boxplot_blocks(full_df, n, column="Close", ticker=ticker)
    # create candlestick chart
    candlestick_plot(full_df, n, ticker=ticker, style="charles")