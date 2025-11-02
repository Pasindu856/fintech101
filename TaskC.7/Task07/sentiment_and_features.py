import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# settings
OUT_DIR = "Taskc7"
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_ID = "yiyanghkust/finbert-tone"
BATCH_SIZE = 32
MAX_LEN = 128


# function to load data
def load_data():
    # Load the historical DJIA price data
    prices = pd.read_csv(f"{OUT_DIR}/upload_DJIA_table.csv")
    prices.columns = prices.columns.str.lower()
    prices.rename(columns={"adj close": "adjclose"}, inplace=True)
    prices["date"] = pd.to_datetime(prices["date"])
    # Clean and sort by date to make sure it’s in time order
    prices = prices.dropna().sort_values("date").reset_index(drop=True)

    # Load the Reddit financial news dataset
    news = pd.read_csv(f"{OUT_DIR}/RedditNews.csv")
    news.columns = news.columns.str.lower()
    news["date"] = pd.to_datetime(news["date"])
    # Remove rows with missing news text and sort by date
    news = news.dropna(subset=["news"]).sort_values("date").reset_index(drop=True)
    # Remove stray characters like b' and quotes left over from CSV formatting
    news["news"] = news["news"].astype(str).str.replace(r"^b['\"]|['\"]$", "", regex=True)

    return prices, news


# Finbert sentiment
def add_finbert(news_df):
    # Use GPU acceleration if available (MPS on Mac, CPU otherwise)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device.type == "mps" else torch.float32

    # Load the FinBERT model and tokenizer from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, torch_dtype=dtype
    ).to(device)
    model.eval()

    scores = []
    # Process news headlines in small batches for faster GPU inference
    for i in tqdm(range(0, len(news_df), BATCH_SIZE), desc="FinBERT scoring"):
        batch = news_df["news"].iloc[i:i + BATCH_SIZE].tolist()
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(device)

        # Disable gradient computation for efficiency
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            # FinBERT output: [negative, neutral, positive]
            # sentiment score = positive - negative
            batch_scores = probs[:, 2] - probs[:, 0]
        scores.extend(batch_scores)

    # Store sentiment results in a new column
    news_df["finbert_mean"] = scores
    return news_df


# vader sentiment
def add_vader(news_df):
    # Initialize the VADER sentiment analyzer (rule-based)
    vader = SentimentIntensityAnalyzer()
    tqdm.pandas(desc="VADER scoring")

    # Apply VADER sentiment scoring for each news headline
    # Note: compound score is usually the main indicator; here we label as "Vader_score"
    news_df["vader_mean"] = news_df["news"].progress_apply(
        lambda x: vader.polarity_scores(x)["Vader_score"]
    )
    return news_df


# Aggregrate daily sentiment
def merge_sentiment_with_prices(prices, news):
    # Average the daily sentiment values from all headlines for each date
    daily_sent = (
        news.groupby("date")[["vader_mean", "finbert_mean"]]
        .mean()
        .reset_index()
    )
    # Merge the daily average sentiment with the stock price data
    merged = pd.merge(prices, daily_sent, on="date", how="inner")
    return merged


# Adding features
def add_features(df):
    # Sort by date to make sure indicators are calculated correctly
    df = df.sort_values("date").reset_index(drop=True)

    # Basic technical indicators
    df["ret_1d"] = df["adjclose"].pct_change(1)                    # 1-day return
    df["mom_5"] = df["adjclose"] / df["adjclose"].shift(5) - 1     # 5-day momentum
    df["vol_5"] = df["ret_1d"].rolling(5).std()                    # 5-day volatility
    df["ma_5"] = df["adjclose"].rolling(5).mean()                  # 5-day moving average
    df["ma_10"] = df["adjclose"].rolling(10).mean()                # 10-day moving average

    # RSI (Relative Strength Index) 14-day calculation
    win = 14
    delta = df["adjclose"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(win).mean()
    avg_loss = pd.Series(loss).rolling(win).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # Target variable setup
    # ret_next_1d = percentage change for next day
    df["ret_next_1d"] = df["adjclose"].shift(-1) / df["adjclose"] - 1
    # Binary target: 1 if price goes up next day, 0 if down
    df["target"] = (df["adjclose"].shift(-1) > df["adjclose"]).astype(int)

    # Drop rows with missing values from rolling calculations
    df = df.dropna().reset_index(drop=True)
    return df



if __name__ == "__main__":
    # Load stock price and news headline data
    prices, news = load_data()

    # Generate VADER and FinBERT sentiment scores
    news = add_vader(news)
    news = add_finbert(news)

    #  Merge daily average sentiment scores with stock data
    combined = merge_sentiment_with_prices(prices, news)

    #  Add technical indicators and target labels
    dataset = add_features(combined)

    #  Save final cleaned dataset for modeling
    output_path = f"{OUT_DIR}/final_dataset.csv"
    dataset.to_csv(output_path, index=False)
    print(f"\nSaved final dataset -> {output_path}")

    # Display basic stats for verification
    print(f"Total rows: {len(dataset)} | Up days: {dataset['target'].sum()} | Down days: {len(dataset) - dataset['target'].sum()}")
