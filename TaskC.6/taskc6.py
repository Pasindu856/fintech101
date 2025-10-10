# COS30018 – Task C.5 
# Multivariate + Multistep stock forecasting with per-day plots and CSV exports.
# I wrote comments for each line so I  can understand clearly.


# ----- Imports -----
import os                     # To set random seed reproducibility for Python hash
import random                 # To fix Python's own random behaviour
import json                   # To save lists/dicts into CSV-friendly text
import numpy as np            # For arrays and math operations
import pandas as pd           # For handling DataFrames and saving CSVs
import matplotlib.pyplot as plt  # For plotting graphs
import tensorflow as tf       # TensorFlow for building deep learning models

# Keras model building (we use Sequential for simplicity)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping

# My custom data processor (from Task C.2)
from DataHandler import StockDataProcessor


# -----------------------------
# Function to create sequences
# -----------------------------
def make_sequences_multivar(df, feature_cols, target_col, lookback=60, lookahead=5):
    """
    Build sequences from a multivariate dataframe.
    X shape = (samples, lookback, features)
    y shape = (samples, lookahead)
    """

    # Remove invalid feature columns if not in dataframe
    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        raise ValueError("No valid feature columns for X.")

    # Make sure target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    # Sort by time, drop missing values
    s = df.sort_index().dropna()

    # Convert features into numpy array (N, F)
    Xmat = s[feature_cols].astype("float32").values
    # Convert target into numpy array (N,)
    tgt = s[target_col].astype("float32").values

    N, F = Xmat.shape
    m = N - lookback - lookahead + 1  # number of samples we can make
    if m <= 0:
        raise ValueError("Not enough rows for chosen lookback/lookahead.")

    # Use simple Python loops to build samples
    X_list, y_list = [], []
    for i in range(m):
        # Past lookback rows as input
        X_list.append(Xmat[i : i + lookback, :])
        # Next lookahead rows of target as output
        y_list.append(tgt[i + lookback : i + lookback + lookahead])

    # Convert lists into numpy arrays
    X = np.stack(X_list).astype("float32")  # (m, lookback, F)
    y = np.stack(y_list).astype("float32")  # (m, lookahead)
    return X, y


# -----------------------------
# Function to split train/val
# -----------------------------
def split_train_val(X, y, val_ratio=0.15):
    n = len(X)
    if n == 0:
        raise ValueError("Empty sequence set.")
    n_val = max(1, int(n * val_ratio))  # at least 1 validation sample
    return (X[:-n_val], y[:-n_val]), (X[-n_val:], y[-n_val:])


# -----------------------------
# Function to build the model
# -----------------------------
def build_model(rnn_kind, num_layers, layer_size, lookback, input_dim, output_dim, dropout=0.0):
    """
    rnn_kind = "LSTM", "GRU", or "RNN"
    num_layers = how many recurrent layers
    layer_size = number of units per layer
    lookback = number of timesteps in each input
    input_dim = number of features
    output_dim = how many steps ahead we predict
    """
    rnn_kind = rnn_kind.strip().upper()
    if rnn_kind not in {"LSTM", "GRU", "RNN"}:
        raise ValueError("rnn_kind must be one of: LSTM, GRU, RNN")

    # Pick the right layer type
    RNN = {"LSTM": LSTM, "GRU": GRU, "RNN": SimpleRNN}[rnn_kind]

    # Build model with Sequential for simplicity
    model = Sequential(name=f"{rnn_kind}_stack")

    # First layer requires input shape
    if num_layers == 1:
        model.add(RNN(layer_size, input_shape=(lookback, input_dim)))
    else:
        # First layer returns sequences if we stack more
        model.add(RNN(layer_size, return_sequences=True, input_shape=(lookback, input_dim)))
        # Middle layers (if > 2)
        for _ in range(num_layers - 2):
            model.add(RNN(layer_size, return_sequences=True))
        # Last recurrent layer
        model.add(RNN(layer_size))

    # Optional dropout layer
    if dropout and dropout > 0:
        model.add(Dropout(dropout))

    # Final dense layer outputs lookahead predictions
    model.add(Dense(output_dim, name="output"))

    # Compile model with Adam optimizer and MSE loss
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="mse",
                  metrics=["mae"])
    return model


# -----------------------------
# Helper metrics
# -----------------------------
def _rmse(arr):
    arr = np.asarray(arr, dtype=float)
    return float(np.sqrt(np.mean(np.square(arr))))

def _mape(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    denom = np.clip(np.abs(a), 1e-8, None)
    return float(np.mean(np.abs((a - p) / denom)) * 100.0)


# -----------------------------
# Main Program
# -----------------------------
if __name__ == "__main__":
    # Fix random seeds for reproducibility
    SEED = 42
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Load data using my DataHandler
    TICKER = "CBA.AX"
    proc = StockDataProcessor(
        ticker=TICKER,
        start="2015-01-01",
        end="2021-01-01",
        split_method="ratio",
        test_size=0.2,
        scale=True,
        feature_cols=["open", "high", "low", "close", "adjclose", "volume"],
    )
    out = proc.run_pipeline()
    train_df, test_df = out["train_df"], out["test_df"]

    # Defensive feature selection
    desired = ["open", "high", "low", "close", "adjclose", "volume"]
    feature_cols = [c for c in desired if c in train_df.columns]
    if not feature_cols:
        feature_cols = ["adjclose"] if "adjclose" in train_df else ["close"]
    target_col = "adjclose" if "adjclose" in train_df else "close"
    print(f"[C.5] Using features: {feature_cols}")
    print(f"[C.5] Using target: {target_col}")

    # Fixed lookback
    LOOKBACK = 60

    # Ask user for settings
    print("Model type? 1=LSTM  2=GRU  3=RNN")
    t = int(input("Enter choice [1]: ").strip() or "1")
    rnn_kind = "LSTM" if t == 1 else ("GRU" if t == 2 else "RNN")

    num_layers = int(input("Number of layers [1]: ").strip() or "1")
    layer_size = int(input("Units per layer [64]: ").strip() or "64")
    lookahead = int(input("Predict how many days ahead (k)? [5]: ").strip() or "5")
    epochs = int(input("Epochs [120]: ").strip() or "120")
    batch_size = int(input("Batch size [32]: ").strip() or "32")

    # Build sequences
    X_train_all, y_train_all = make_sequences_multivar(train_df, feature_cols, target_col, LOOKBACK, lookahead)
    X_test, y_test = make_sequences_multivar(test_df, feature_cols, target_col, LOOKBACK, lookahead)

    # Train/val split
    (Xtr, ytr), (Xval, yval) = split_train_val(X_train_all, y_train_all, val_ratio=0.15)

    # Build model
    model = build_model(rnn_kind, num_layers, layer_size, LOOKBACK, len(feature_cols), lookahead, dropout=0.15)

    # Early stopping to avoid overfitting
    es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1)

    # Train model
    history = model.fit(
        Xtr, ytr,
        validation_data=(Xval, yval),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[es],
        shuffle=False
    )

    # Evaluate
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MSE: {test_loss:.6f} | Test MAE: {test_mae:.6f}")

    # Predict
    y_hat = model.predict(X_test, verbose=0)

    # Per-day MAE (scaled values)
    per_day_mae_scaled = np.mean(np.abs(y_test - y_hat), axis=0)
    print("Per-day MAE (scaled):", per_day_mae_scaled)

    # ----- Save predictions and metrics to CSVs -----
    target_start = LOOKBACK + lookahead - 1
    target_index = test_df.index[target_start : target_start + y_test.shape[0]]

    rows = []
    for d in range(lookahead):
        a_sc, p_sc = y_test[:, d], y_hat[:, d]
        for j in range(len(a_sc)):
            rows.append({
                "timestamp": target_index[j],
                "sample_index": j,
                "day_ahead": d + 1,
                "actual_scaled": float(a_sc[j]),
                "pred_scaled": float(p_sc[j]),
                "abs_error_scaled": float(abs(a_sc[j] - p_sc[j])),
                "ticker": TICKER,
                "lookback": LOOKBACK,
                "lookahead": lookahead,
                "model": rnn_kind,
                "layers": num_layers,
                "units": layer_size,
            })
    df_long = pd.DataFrame(rows)

    # Inverse scale if possible
    target_scaler = out.get("scalers", {}).get(target_col, None)
    if target_scaler is not None:
        df_long["actual"] = target_scaler.inverse_transform(df_long[["actual_scaled"]]).ravel()
        df_long["pred"] = target_scaler.inverse_transform(df_long[["pred_scaled"]]).ravel()
        df_long["abs_error"] = (df_long["actual"] - df_long["pred"]).abs()

    # Save long CSV
    all_days_csv = f"pred_vs_actual_all_days_{rnn_kind}.csv"
    df_long.to_csv(all_days_csv, index=False)

    # Save per-day CSVs
    for d in range(1, lookahead + 1):
        df_long[df_long["day_ahead"] == d].to_csv(f"pred_vs_actual_day{d}_{rnn_kind}.csv", index=False)

    # Metrics CSV
    metrics = []
    for d in range(1, lookahead + 1):
        sub = df_long[df_long["day_ahead"] == d]
        rec = {
            "day_ahead": d,
            "MAE_scaled": float(np.mean(np.abs(sub["actual_scaled"] - sub["pred_scaled"]))),
            "RMSE_scaled": _rmse(sub["actual_scaled"] - sub["pred_scaled"]),
        }
        if "actual" in sub.columns:
            rec.update({
                "MAE": float(np.mean(np.abs(sub["actual"] - sub["pred"]))),
                "RMSE": _rmse(sub["actual"] - sub["pred"]),
                "MAPE_percent": _mape(sub["actual"], sub["pred"]),
            })
        metrics.append(rec)
    pd.DataFrame(metrics).to_csv(f"metrics_per_day_{rnn_kind}.csv", index=False)

    # ----- Plots -----
    # One plot per horizon
    for d in range(lookahead):
        plt.figure()
        plt.plot(y_test[:, d], label=f"Actual Day {d+1}")
        plt.plot(y_hat[:, d], label=f"Predicted Day {d+1}")
        plt.legend()
        plt.title(f"{rnn_kind} Day {d+1} Ahead")
        plt.savefig(f"pred_day{d+1}_{rnn_kind}.png")
        plt.close()

    # Loss curve
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title(f"{rnn_kind} Loss Curve")
    plt.savefig(f"loss_{rnn_kind}.png")
    plt.close()

    # Per-day MAE bar
    plt.figure()
    plt.bar(np.arange(1, lookahead+1), per_day_mae_scaled)
    plt.title(f"{rnn_kind} Per-day MAE (scaled)")
    plt.savefig(f"mae_per_day_{rnn_kind}.png")
    plt.close()