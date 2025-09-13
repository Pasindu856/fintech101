# File: taskc4_alt.py
# COS30018 – Task C.4
# My deep learning code for stock forecasting
# Includes reproducibility, early stopping, plots, and results logging.

# -----------------------------
# Imports
# -----------------------------
import os      # used to set environment variables for reproducibility
import random  # used to fix Python random behaviour
import numpy as np   # numerical arrays, maths, sliding windows
import pandas as pd  # handle tabular data and save results into CSV
import matplotlib.pyplot as plt  # plotting graphs (predictions, losses)
import tensorflow as tf  # main deep learning framework

# Import Keras layers, models, and callbacks
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, GRU, SimpleRNN  # building blocks
from tensorflow.keras.models import Model  # functional API model class
from tensorflow.keras.callbacks import EarlyStopping  # to stop training early

# Import my Task C.2 data pipeline class
from DataHandler import StockDataProcessor


# -----------------------------
# Function: make_sequences
# -----------------------------
def make_sequences(series, lookback=60, lookahead=1):
    """
    Create supervised sequences from a 1D series.
    Each X = lookback past values, y = future value after lookahead.
    """
    s = series.copy().sort_index().dropna()  # make a clean, sorted copy of the data
    v = s.values.astype(float).reshape(-1)   # turn it into a 1D float array
    N = v.shape[0]                           # total number of points
    m = N - lookback - lookahead + 1         # number of usable training samples
    if m <= 0:                               # stop if not enough rows
        raise ValueError("Not enough rows for the chosen lookback/lookahead.")

    sw = np.lib.stride_tricks.sliding_window_view(v, lookback)  # generate rolling windows
    X = sw[:m]  # only keep the first m windows (aligned with y below)
    y = v[lookback + lookahead - 1 : lookback + lookahead - 1 + m]  # future values as targets

    X = X.astype("float32")[..., None]  # cast to float32 and add feature dimension
    y = y.astype("float32")             # cast targets to float32
    return X, y  # return features and labels


# -----------------------------
# Function: split_train_val
# -----------------------------
def split_train_val(X, y, val_ratio=0.15):
    """
    Split the data into training and validation sets while keeping time order.
    """
    n = len(X)                                # total number of samples
    if n == 0:                                # safety check for empty input
        raise ValueError("Empty sequence set.")
    n_val = max(1, int(n * val_ratio))        # number of validation samples (at least 1)
    return (X[:-n_val], y[:-n_val]), (X[-n_val:], y[-n_val:])  # return split parts


# -----------------------------
# Function: build_model
# -----------------------------
def build_model(layer_name: str, num_layers: int, layer_size: int, lookback: int,
                dropout: float = 0.0) -> tf.keras.Model:
    """
    Build and compile a model using the chosen RNN type (LSTM, GRU, or RNN).
    """
    name = layer_name.strip().upper()  # clean up the string and convert to uppercase
    layer_map = {"LSTM": LSTM, "GRU": GRU, "RNN": SimpleRNN}  # map name to class
    if name not in layer_map:          # check validity of input
        raise ValueError("layer_name must be one of: LSTM, GRU, RNN")
    RNN = layer_map[name]              # choose the layer type

    x_in = Input(shape=(lookback, 1), name="input")  # define the input layer
    x = x_in                                        # pass input to x
    for i in range(num_layers):                     # loop through number of layers
        return_seq = (i < num_layers - 1)           # middle layers must return sequences
        x = RNN(layer_size, return_sequences=return_seq, name=f"{name}_{i+1}")(x)  # add layer

    if dropout and dropout > 0:                     # optional dropout
        x = Dropout(dropout, name="post_stack_dropout")(x)

    y_out = Dense(1, name="output")(x)  # final dense layer with one output

    model = Model(inputs=x_in, outputs=y_out, name=f"{name}_stack")  # create model object
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # Adam optimiser
                  loss="mse", metrics=["mae"])  # use MSE as loss and MAE for tracking
    return model  # return ready-to-train model


# -----------------------------
# Main program
# -----------------------------
if __name__ == "__main__":
    # 0) Reproducibility
    SEED = 42                                     # fixed seed number
    os.environ["PYTHONHASHSEED"] = str(SEED)      # fix hash seed for Python
    random.seed(SEED)                             # fix Python random
    np.random.seed(SEED)                          # fix NumPy random
    tf.random.set_seed(SEED)                      # fix TensorFlow random

    # 1) Load stock data from Task C.2 pipeline
    proc = StockDataProcessor(
        ticker="CBA.AX",                          # stock ticker symbol
        start="2016-01-01",                       # start date
        end="2021-01-01",                         # end date
        split_method="ratio", test_size=0.2,      # split by ratio into train/test
        scale=True, feature_cols=["adjclose"]     # use only adjusted close as feature
    )
    out = proc.run_pipeline()                     # run pipeline and get results
    train_df, test_df = out["train_df"], out["test_df"]  # extract train/test sets

    # 2) Make sequences for train and test
    LOOKBACK = 60
    X_train_all, y_train_all = make_sequences(train_df["adjclose"], lookback=LOOKBACK, lookahead=1)
    X_test,       y_test     = make_sequences(test_df["adjclose"],  lookback=LOOKBACK, lookahead=1)

    # 3) Split training into train/validation
    (Xtr, ytr), (Xval, yval) = split_train_val(X_train_all, y_train_all, val_ratio=0.15)

    # 4) Ask user for configuration
    print("Model type? 1=LSTM  2=GRU  3=RNN")  # let user choose type
    try:
        t = int(input("Enter choice [1]: ").strip() or "1")  # default is 1
    except Exception:
        t = 1
    layer_name = "LSTM" if t == 1 else ("GRU" if t == 2 else "RNN")  # pick option

    try:
        num_layers = int(input("Number of layers [1]: ").strip() or "1")  # default 1
    except Exception:
        num_layers = 1

    try:
        layer_size = int(input("Units per layer [32]: ").strip() or "32")  # default 32
    except Exception:
        layer_size = 32

    try:
        epochs = int(input("Epochs [100]: ").strip() or "100")  # default 100
    except Exception:
        epochs = 100

    try:
        batch_size = int(input("Batch size [32]: ").strip() or "32")  # default 32
    except Exception:
        batch_size = 32

    # 5) Build model and train
    model = build_model(layer_name, num_layers, layer_size, LOOKBACK, dropout=0.15)

    es = EarlyStopping(                   # configure EarlyStopping
        monitor="val_loss",               # monitor validation loss
        patience=20,                      # stop after 20 no-improve epochs
        restore_best_weights=True,        # restore best weights automatically
        verbose=1
    )

    history = model.fit(                  # train model
        Xtr, ytr,
        validation_data=(Xval, yval),     # pass validation data
        epochs=epochs, batch_size=batch_size,
        verbose=1,
        callbacks=[es],                   # add EarlyStopping
        shuffle=False                     # keep chronological order
    )

    # 6) Evaluate on test set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)  # run evaluation
    print(f"Test MSE: {test_loss:.6f} | Test MAE: {test_mae:.6f}")   # show results

    # 7) Predictions and plots
    y_hat = model.predict(X_test, verbose=0).ravel()  # make predictions

    # Plot predicted vs actual
    plt.figure()
    plt.plot(y_test, label="Actual")                  # plot true values
    plt.plot(y_hat, label="Predicted")                # plot predicted values
    plt.title(f"{layer_name}: Actual vs Predicted")   # title
    plt.xlabel("Test sample")                         # x-axis label
    plt.ylabel("Scaled Price")                        # y-axis label
    plt.legend()
    plt.tight_layout()
    pred_png = f"pred_{layer_name}.png"               # file name for plot
    plt.savefig(pred_png, dpi=160)                    # save to file
    plt.close()
    print(f"Predictions plot saved as {pred_png}")    # confirm save

    # Plot training and validation loss
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")     # training loss curve
    plt.plot(history.history["val_loss"], label="Val Loss")   # validation loss curve
    plt.xlabel("Epoch")                                      # x-axis label
    plt.ylabel("MSE Loss")                                   # y-axis label
    plt.title(f"{layer_name} Loss Curve")                    # title
    plt.legend()
    plt.tight_layout()
    loss_png = f"loss_{layer_name}.png"                      # file name for plot
    plt.savefig(loss_png, dpi=160)                           # save to file
    plt.close()
    print(f"Loss curve saved as {loss_png}")                 # confirm save

    # 8) Save experiment results
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1  # epoch with lowest val_loss
    summary = pd.DataFrame([{                                   # build one row DataFrame
        "kind": layer_name,
        "layers": num_layers,
        "units": layer_size,
        "epochs_requested": epochs,
        "epochs_ran": len(history.history["loss"]),
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "val_best": float(np.min(history.history["val_loss"])),
        "test_mse": float(test_loss),
        "test_mae": float(test_mae),
        "loss_png": loss_png,
        "pred_png": pred_png
    }])

    csv_path = "results_c4.csv"                                # file path
    write_header = not pd.io.common.file_exists(csv_path)      # add header if file is new
    summary.to_csv(csv_path, mode="a", header=write_header, index=False)  # append to CSV
    print(f"Appended results to {csv_path}")                   # confirm save