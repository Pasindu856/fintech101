# COS30018 – Task C.6 (final)
# ARIMA + SARIMA (rolling multistep, univariate) + DL (multivariate, multistep)
# Runs both ARIMA and SARIMA automatically, builds two ensembles with one alpha,
# and draws a single combined plot per horizon. 

import os                                   # For environment settings (seed hashing)
import random                               # For Python-level reproducible randomness
import warnings                             # To hide harmless scaler warnings
import numpy as np                          # For arrays and numeric operations
import pandas as pd                         # For DataFrame operations and CSV export
import matplotlib.pyplot as plt             # For plotting results
import tensorflow as tf                     # For deep learning (Keras)

# Keras model parts
from tensorflow.keras import Sequential     # Simple sequential model container
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN  # Layers used
from tensorflow.keras.callbacks import EarlyStopping                       # Early stopping

# Statsmodels for ARIMA/SARIMA
from statsmodels.tsa.arima.model import ARIMA      # ARIMA implementation
from statsmodels.tsa.statespace.sarimax import SARIMAX  # SARIMA implementation

# My data pipeline from Task C.2
from DataHandler import StockDataProcessor

# Hide a known scikit-learn warning about feature names when scaling arrays
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Set default matplotlib styles to improve readability of plots
plt.rcParams.update({
    "figure.figsize": (12, 5),   # Wider plots
    "lines.linewidth": 2.0,      # Thicker lines
    "axes.grid": True,           # Show grid
    "grid.alpha": 0.25,          # Light grid
    "legend.frameon": False      # No legend box
})


# -----------------------------
# Metrics helpers
# -----------------------------
def _rmse(arr):
    arr = np.asarray(arr, dtype=float)        # Ensure numeric array
    return float(np.sqrt(np.mean(np.square(arr))))  # RMSE formula


def _mape(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)    # Convert to float arrays
    denom = np.clip(np.abs(a), 1e-8, None)               # Avoid divide-by-zero
    return float(np.mean(np.abs((a - p) / denom)) * 100.0)  # MAPE in percent


# -----------------------------
# Multivariate sequence builder (same style as Task C.5)
# -----------------------------
def make_sequences_multivar(df, feature_cols, target_col, lookback=60, lookahead=5):
    feature_cols = [c for c in feature_cols if c in df.columns]  # Keep only columns that exist
    if not feature_cols:
        raise ValueError("No valid feature columns for X.")      # Stop if none remain
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")  # Target must exist

    s = df.sort_index().dropna()               # Sort chronologically and drop missing rows
    Xmat = s[feature_cols].astype("float32").values  # Features as float32 matrix (N, F)
    tgt  = s[target_col].astype("float32").values    # Target as float32 vector (N,)

    N, F = Xmat.shape                              # N rows, F features
    m = N - lookback - lookahead + 1               # Number of samples possible
    if m <= 0:
        raise ValueError("Not enough rows for chosen lookback/lookahead.")  # Safety check

    X_list, y_list = [], []                        # Buffers for samples
    for i in range(m):                             # Slide a window across time
        X_list.append(Xmat[i:i+lookback, :])       # Past lookback rows go to X
        y_list.append(tgt[i+lookback:i+lookback+lookahead])  # Next k rows go to y

    X = np.stack(X_list).astype("float32")         # Shape (m, lookback, F)
    Y = np.stack(y_list).astype("float32")         # Shape (m, lookahead)
    return X, Y                                    # Return sequences


# -----------------------------
# Simple time split: train/val by ratio (no shuffle)
# -----------------------------
def split_train_val(X, y, val_ratio=0.15):
    n = len(X)                                     # Total samples
    if n == 0:
        raise ValueError("Empty sequence set.")    # Safety check
    n_val = max(1, int(n * val_ratio))             # At least one validation sample
    return (X[:-n_val], y[:-n_val]), (X[-n_val:], y[-n_val:])  # Time-ordered split


# -----------------------------
# RNN model builder (LSTM/GRU/SimpleRNN)
# -----------------------------
def build_rnn(rnn_kind, num_layers, layer_size, lookback, input_dim, output_dim, dropout=0.0):
    rnn_kind = rnn_kind.strip().upper()                   # Normalise kind string
    if rnn_kind not in {"LSTM", "GRU", "RNN"}:            # Validate
        raise ValueError("rnn_kind must be one of: LSTM, GRU, RNN")
    RNN = {"LSTM": LSTM, "GRU": GRU, "RNN": SimpleRNN}[rnn_kind]  # Select class

    model = Sequential(name=f"{rnn_kind}_stack")          # Start sequential model
    if num_layers == 1:                                   # Single layer case
        model.add(RNN(layer_size, input_shape=(lookback, input_dim)))
    else:
        model.add(RNN(layer_size, return_sequences=True, input_shape=(lookback, input_dim)))  # First layer
        for _ in range(num_layers - 2):                   # Optional middle layers
            model.add(RNN(layer_size, return_sequences=True))
        model.add(RNN(layer_size))                        # Last recurrent layer (no sequences)

    if dropout and dropout > 0:
        model.add(Dropout(dropout))                       # Optional dropout

    model.add(Dense(output_dim, name="output"))           # Final dense to k steps
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),  # Adam optimizer
                  loss="mse", metrics=["mae"])            # MSE loss, MAE metric
    return model                                          # Return compiled model


# -----------------------------
# Rolling multistep ARIMA (rewritten, aligned with DL windows)
# -----------------------------
def roll_forecast_arima(train_vals: np.ndarray,
                        test_vals: np.ndarray,
                        lookback: int,
                        horizon: int,
                        order=(5, 1, 0)) -> np.ndarray:
    """
    Create a forecast matrix with shape (N, horizon), where each row corresponds
    to the DL forecast origin on the test set and contains the next k predictions.
    """
    tr = np.asarray(train_vals, dtype=float).reshape(-1)   # Train series as float vector
    te = np.asarray(test_vals, dtype=float).reshape(-1)    # Test series as float vector
    N  = len(te) - lookback - horizon + 1                  # Number of forecast origins
    if N <= 0:
        raise ValueError("Not enough test data for given lookback + horizon.")  # Safety check

    out  = np.empty((N, horizon), dtype=float)             # Output matrix
    hist = np.r_[tr, te[:lookback]].astype(float)          # Initial history = train + first lookback of test

    for i in range(N):                                     # Loop over each forecast origin
        try:
            fitted = ARIMA(hist, order=order).fit(method_kwargs={"warn_convergence": False})  # Fit ARIMA
            fc     = fitted.forecast(steps=horizon)        # Forecast next k steps
            out[i] = np.asarray(fc, dtype=float)           # Save predictions
        except Exception:
            out[i] = np.full(horizon, hist[-1], dtype=float)  # Fallback to last value if failure
        hist = np.r_[hist, te[lookback + i]]               # Reveal the next true test value

    return out                                             # Return ARIMA forecast matrix


# -----------------------------
# Rolling multistep SARIMA (rewritten, aligned with DL windows)
# -----------------------------
def roll_forecast_sarima(train_vals: np.ndarray,
                         test_vals: np.ndarray,
                         lookback: int,
                         horizon: int,
                         order=(1, 1, 1),
                         seasonal_order=(0, 1, 1, 5)) -> np.ndarray:
    """
    Same alignment as ARIMA but with SARIMA to capture seasonality (e.g., weekly).
    """
    tr = np.asarray(train_vals, dtype=float).reshape(-1)   # Train series as float vector
    te = np.asarray(test_vals, dtype=float).reshape(-1)    # Test series as float vector
    N  = len(te) - lookback - horizon + 1                  # Number of forecast origins
    if N <= 0:
        raise ValueError("Not enough test data for given lookback + horizon.")  # Safety check

    out  = np.empty((N, horizon), dtype=float)             # Output matrix
    hist = np.r_[tr, te[:lookback]].astype(float)          # Initial history

    for i in range(N):                                     # Loop over each forecast origin
        try:
            mod = SARIMAX(hist, order=order, seasonal_order=seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False)  # Build SARIMA
            fit = mod.fit(disp=False)                      # Fit quietly
            fc  = fit.forecast(steps=horizon)              # Forecast next k steps
            out[i] = np.asarray(fc, dtype=float)           # Save predictions
        except Exception:
            out[i] = np.full(horizon, hist[-1], dtype=float)  # Fallback on failure
        hist = np.r_[hist, te[lookback + i]]               # Reveal next test observation

    return out                                             # Return SARIMA forecast matrix


# -----------------------------
# Save long CSV + metrics CSV (compatible with Task C.5)
# -----------------------------
def save_outputs(tag, ticker, lookback, lookahead, target_index, y_true_scaled, y_pred_scaled,
                 scaler=None, model_name=""):
    rows = []                                              # Buffer for long-form rows
    for d in range(lookahead):                             # For each forecast day
        a_sc, p_sc = y_true_scaled[:, d], y_pred_scaled[:, d]  # True and predicted (scaled)
        for j in range(len(a_sc)):                         # Each sample index
            rows.append({
                "timestamp": target_index[j],              # Time index for the sample
                "sample_index": j,                         # Row id within test windows
                "day_ahead": d + 1,                        # Horizon (1..k)
                "actual_scaled": float(a_sc[j]),           # Scaled actual
                "pred_scaled": float(p_sc[j]),             # Scaled prediction
                "abs_error_scaled": float(abs(a_sc[j] - p_sc[j])),  # Scaled absolute error
                "ticker": ticker,                          # Ticker symbol
                "lookback": lookback,                      # Lookback used
                "lookahead": lookahead,                    # k-step horizon
                "model": model_name                        # Model name tag
            })
    df_long = pd.DataFrame(rows)                           # Create long-form DataFrame

    if scaler is not None:                                 # If scaler available
        df_long["actual"] = scaler.inverse_transform(df_long[["actual_scaled"]]).ravel()  # Back to price scale
        df_long["pred"]   = scaler.inverse_transform(df_long[["pred_scaled"]]).ravel()    # Back to price scale
        df_long["abs_error"] = (df_long["actual"] - df_long["pred"]).abs()                # Absolute error (price)

    long_csv = f"{tag}_pred_vs_actual_all_days_{model_name}.csv"  # File name for long CSV
    df_long.to_csv(long_csv, index=False)                # Save long CSV without index

    metrics = []                                         # Buffer for per-day metrics
    for d in range(1, lookahead + 1):                    # For each horizon day
        sub = df_long[df_long["day_ahead"] == d]         # Filter rows of that horizon
        rec = {
            "day_ahead": d,                              # Day ahead number
            "MAE_scaled": float(np.mean(np.abs(sub["actual_scaled"] - sub["pred_scaled"]))),   # MAE scaled
            "RMSE_scaled": _rmse(sub["actual_scaled"] - sub["pred_scaled"]),                   # RMSE scaled
        }
        if "actual" in sub.columns:                      # If inverse scaled columns exist
            rec.update({
                "MAE": float(np.mean(np.abs(sub["actual"] - sub["pred"]))),                   # MAE original
                "RMSE": _rmse(sub["actual"] - sub["pred"]),                                   # RMSE original
                "MAPE_percent": _mape(sub["actual"], sub["pred"]),                            # MAPE original
            })
        metrics.append(rec)                              # Append record
    metrics_csv = f"{tag}_metrics_per_day_{model_name}.csv"  # File name for metrics CSV
    pd.DataFrame(metrics).to_csv(metrics_csv, index=False)   # Save metrics CSV

    return df_long, long_csv, metrics_csv                # Return DataFrames and paths


# -----------------------------
# Combined plot per horizon (clean style)
# -----------------------------
def plot_combined_per_horizon(day_idx, y_true, y_dl, y_arima, y_sarima, y_ens_a, y_ens_s):
    plt.figure()                                         # New figure for this horizon
    plt.plot(y_true[:, day_idx],   label=f"Actual D{day_idx+1}", color="black")          # Actual series
    plt.plot(y_dl[:, day_idx],     label=f"DL D{day_idx+1}",     color="#1f77b4")        # DL series
    plt.plot(y_arima[:, day_idx],  label=f"ARIMA D{day_idx+1}",  color="#ff7f0e", linestyle="--")  # ARIMA
    plt.plot(y_sarima[:, day_idx], label=f"SARIMA D{day_idx+1}", color="#2ca02c", linestyle="-.")  # SARIMA
    plt.plot(y_ens_a[:, day_idx],  label=f"ENS(ARIMA+DL) D{day_idx+1}",  color="#d62728", alpha=0.9)   # Ens A
    plt.plot(y_ens_s[:, day_idx],  label=f"ENS(SARIMA+DL) D{day_idx+1}", color="#9467bd", alpha=0.9)   # Ens S

    plt.title(f"C.6 Combined – Day {day_idx+1} Ahead")   # Title with horizon number
    plt.xlabel("Test sample index")                       # X label is index in test windows
    plt.ylabel("Price")                                   # Y label is price scale
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)  # Move legend outside
    plt.tight_layout()                                    # Fit elements nicely
    plt.savefig(f"C6_combined_day{day_idx+1}.png", dpi=150)  # Save image for report
    plt.close()                                           # Close figure to free memory


# -----------------------------
# Main script
# -----------------------------
if __name__ == "__main__":
    SEED = 42                                            # Fixed seed for reproducibility
    os.environ["PYTHONHASHSEED"] = str(SEED)             # Make hash seed deterministic
    random.seed(SEED)                                    # Seed Python RNG
    np.random.seed(SEED)                                 # Seed NumPy RNG
    tf.random.set_seed(SEED)                             # Seed TensorFlow RNG

    TICKER = "CBA.AX"                                    # Chosen ticker for assignment
    proc = StockDataProcessor(                           # Build data processor
        ticker=TICKER,
        start="2015-01-01",
        end="2021-01-01",
        split_method="ratio",
        test_size=0.2,
        scale=True,
        feature_cols=["open", "high", "low", "close", "adjclose", "volume"],  # Multivariate features
    )
    out = proc.run_pipeline()                            # Run Task C.2 pipeline
    train_df, test_df = out["train_df"], out["test_df"] # Get train and test sets

    desired = ["open", "high", "low", "close", "adjclose", "volume"]          # Preferred features
    feature_cols = [c for c in desired if c in train_df.columns]              # Keep those present
    target_col = "adjclose" if "adjclose" in train_df else "close"            # Choose target col
    print(f"[C.6] Features for DL: {feature_cols}")                           # Log features
    print(f"[C.6] Target: {target_col}")                                      # Log target

    LOOKBACK = 60                                        # Lookback window size

    print("RNN? 1=LSTM  2=GRU  3=RNN")                   # Ask user for RNN type
    t = int(input("Enter choice [1]: ").strip() or "1")  # Read choice with default
    rnn_kind = "LSTM" if t == 1 else ("GRU" if t == 2 else "RNN")  # Map to name
    num_layers = int(input("Number of layers [1]: ").strip() or "1")      # Layers
    layer_size = int(input("Units per layer [64]: ").strip() or "64")     # Units
    lookahead = int(input("Predict how many days ahead (k)? [5]: ").strip() or "5")  # Horizon k
    epochs = int(input("Epochs [120]: ").strip() or "120")                # Training epochs
    batch_size = int(input("Batch size [32]: ").strip() or "32")          # Batch size

    X_train_all, y_train_all = make_sequences_multivar(   # Build sequences for train set
        train_df, feature_cols, target_col, LOOKBACK, lookahead
    )
    X_test, y_test = make_sequences_multivar(             # Build sequences for test set
        test_df, feature_cols, target_col, LOOKBACK, lookahead
    )
    (Xtr, ytr), (Xval, yval) = split_train_val(           # Split train into train/val
        X_train_all, y_train_all, val_ratio=0.15
    )

    model = build_rnn(                                    # Create RNN model
        rnn_kind, num_layers, layer_size, LOOKBACK, len(feature_cols), lookahead, dropout=0.15
    )
    es = EarlyStopping(                                   # Early stopping configuration
        monitor="val_loss", patience=20, restore_best_weights=True, verbose=1
    )
    history = model.fit(                                  # Train the model
        Xtr, ytr,
        validation_data=(Xval, yval),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[es],
        shuffle=False                                     # Keep time order
    )
    y_hat_dl_scaled = model.predict(X_test, verbose=0)    # DL predictions (scaled)

    scaler = out.get("scalers", {}).get(target_col, None) # Get scaler for target if available

    def inv_series(df, col, scaler):                      # Helper to inverse-scale 1D series
        s = df[col].astype("float32").values.reshape(-1, 1)   # Make column into 2D array
        if scaler is not None:                               # If scaler exists
            return scaler.inverse_transform(s).ravel()       # Inverse scale to price
        return s.ravel()                                     # Else return as 1D float

    train_orig = inv_series(train_df, target_col, scaler)    # Train target on original scale
    test_orig  = inv_series(test_df,  target_col, scaler)    # Test target on original scale

    target_start = LOOKBACK + lookahead - 1                  # First index of target in test set
    M = y_test.shape[0]                                      # Number of sliding samples in test
    target_index = test_df.index[target_start: target_start + M]  # Aligned timestamps

    ar_order   = (5, 1, 0)                                   # ARIMA order used
    sar_order  = (1, 1, 1)                                   # SARIMA AR order
    seas_order = (0, 1, 1, 5)                                # SARIMA seasonal order (weekly)

    y_hat_arima_orig  = roll_forecast_arima(                 # Run ARIMA forecasts
        train_orig, test_orig, LOOKBACK, lookahead, order=ar_order
    )
    y_hat_sarima_orig = roll_forecast_sarima(                # Run SARIMA forecasts
        train_orig, test_orig, LOOKBACK, lookahead, order=sar_order, seasonal_order=seas_order
    )

    if scaler is not None:                                   # Convert DL predictions to original scale
        y_true_orig   = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel().reshape(M, lookahead)
        y_hat_dl_orig = scaler.inverse_transform(y_hat_dl_scaled.reshape(-1, 1)).ravel().reshape(M, lookahead)
    else:
        y_true_orig   = y_test.copy()                        # If no scaler, nothing to convert
        y_hat_dl_orig = y_hat_dl_scaled.copy()

    print("Ensemble weight α (stat model) in [0..1]?  default 0.5")   # Ask for alpha
    try:
        alpha = float(input("Enter α [0.5]: ").strip() or "0.5")      # Read alpha with default
        alpha = min(max(alpha, 0.0), 1.0)                             # Clamp to [0, 1]
    except Exception:
        alpha = 0.5                                                   # Fallback default

    y_hat_ens_arima_orig  = alpha * y_hat_arima_orig  + (1.0 - alpha) * y_hat_dl_orig  # ARIMA+DL
    y_hat_ens_sarima_orig = alpha * y_hat_sarima_orig + (1.0 - alpha) * y_hat_dl_orig  # SARIMA+DL

    def fwd_scale(arr2d, scaler):                          # Forward-scale to match CSV style
        if scaler is None:
            return arr2d.astype("float32")                 # If no scaler, cast only
        flat = arr2d.reshape(-1, 1)                        # Flatten to (N*H, 1)
        sc   = scaler.transform(flat).ravel()              # Apply scaler
        return sc.reshape(arr2d.shape).astype("float32")   # Back to (N, H)

    y_test_scaled           = y_test                        # Already scaled from sequence builder
    y_hat_arima_scaled      = fwd_scale(y_hat_arima_orig,     scaler)  # ARIMA scaled
    y_hat_sarima_scaled     = fwd_scale(y_hat_sarima_orig,    scaler)  # SARIMA scaled
    y_hat_dl_scaled_csv     = y_hat_dl_scaled                 # DL already scaled
    y_hat_ens_arima_scaled  = fwd_scale(y_hat_ens_arima_orig,  scaler) # Ensemble ARIMA scaled
    y_hat_ens_sarima_scaled = fwd_scale(y_hat_ens_sarima_orig, scaler) # Ensemble SARIMA scaled

    tag = "C6"                                              # File tag for Task C.6
    _ = save_outputs(tag, TICKER, LOOKBACK, lookahead, target_index, y_test_scaled, y_hat_arima_scaled,
                     scaler, f"ARIMA{ar_order}")            # Save ARIMA CSVs and metrics
    _ = save_outputs(tag, TICKER, LOOKBACK, lookahead, target_index, y_test_scaled, y_hat_sarima_scaled,
                     scaler, f"SARIMA{sar_order}x{seas_order}")  # Save SARIMA CSVs and metrics
    _ = save_outputs(tag, TICKER, LOOKBACK, lookahead, target_index, y_test_scaled, y_hat_dl_scaled_csv,
                     scaler, rnn_kind)                      # Save DL CSVs and metrics
    _ = save_outputs(tag, TICKER, LOOKBACK, lookahead, target_index, y_test_scaled, y_hat_ens_arima_scaled,
                     scaler, f"ENS(ARIMA+{rnn_kind},α={alpha:.2f})")  # Save ensemble A
    _ = save_outputs(tag, TICKER, LOOKBACK, lookahead, target_index, y_test_scaled, y_hat_ens_sarima_scaled,
                     scaler, f"ENS(SARIMA+{rnn_kind},α={alpha:.2f})") # Save ensemble S

    for d in range(lookahead):                               # One combined plot per horizon
        plot_combined_per_horizon(
            d,
            y_true=y_true_orig,
            y_dl=y_hat_dl_orig,
            y_arima=y_hat_arima_orig,
            y_sarima=y_hat_sarima_orig,
            y_ens_a=y_hat_ens_arima_orig,
            y_ens_s=y_hat_ens_sarima_orig
        )

    plt.figure(figsize=(10, 4))                              # New figure for loss curve
    plt.plot(history.history["loss"], label="Train Loss")    # Training loss
    plt.plot(history.history["val_loss"], label="Val Loss")  # Validation loss
    plt.title(f"{rnn_kind} Loss Curve (C.6)")                # Title
    plt.xlabel("Epoch")                                      # X label
    plt.ylabel("MSE")                                        # Y label
    plt.legend()                                             # Legend
    plt.tight_layout()                                       # Fit nicely
    plt.savefig(f"C6_loss_{rnn_kind}.png", dpi=150)          # Save figure
    plt.close()                                              # Close to free memory

    print("\nFinished Task C.6. CSVs and combined plots are saved.")   # Final message