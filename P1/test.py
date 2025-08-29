import numpy as np

import matplotlib.pyplot as plt

from stock_prediction import create_model, load_data
from parameters import *


def plot_graph(test_df):
    """
    This function plots true close price along with predicted close price
    with blue and red colors respectively
    """
    plt.plot(test_df[f'true_adjclose_{LOOKUP_STEP}'], c='b')
    plt.plot(test_df[f'adjclose_{LOOKUP_STEP}'], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()
def get_final_df(model, data):
    """
    Build a dataframe with actual vs predicted prices for the test set.
    - Flattens MultiIndex columns (uses level 0)
    - Normalizes names to lowercase
    - Ensures 'adjclose' exists (fallbacks)
    - Forces numeric dtypes
    - Vectorized profit computation
    """
    import numpy as np
    import pandas as pd

    def _series_from_aliases(df, aliases):
        """Return the first matching column as a Series, even if duplicates exist."""
        cols = [str(c).lower() for c in df.columns]
        for alias in aliases:
            if alias in cols:
                idxs = [i for i, c in enumerate(cols) if c == alias]
                s = df.iloc[:, idxs[0]]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                return s.squeeze()
        raise KeyError(f"None of aliases {aliases} found in columns: {list(df.columns)}")

    X_test = data["X_test"]
    y_test = data["y_test"]

    # Predict
    y_pred = model.predict(X_test, verbose=0)

    # Invert scaling if needed
    if SCALE:
        y_test = np.squeeze(
            data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0))
        )
        y_pred = np.squeeze(
            data["column_scaler"]["adjclose"].inverse_transform(y_pred)
        )
    else:
        y_test = np.squeeze(y_test)
        y_pred = np.squeeze(y_pred)

    # Work on a copy
    test_df = data["test_df"].copy()

    # --- FLATTEN MultiIndex columns (take level 0) ---
    if isinstance(test_df.columns, pd.MultiIndex):
        test_df.columns = [str(t[0]) for t in test_df.columns]

    # --- Normalize to lowercase, drop dups ---
    test_df.columns = [str(c).lower() for c in test_df.columns]
    test_df = test_df.loc[:, ~test_df.columns.duplicated(keep="first")]

    # Ensure base 'adjclose' exists
    if "adjclose" not in test_df.columns:
        if "adj close" in test_df.columns:
            test_df["adjclose"] = test_df["adj close"]
        elif "close" in test_df.columns:
            test_df["adjclose"] = test_df["close"]

    # Attach predictions/ground truth
    test_df[f"adjclose_{LOOKUP_STEP}"] = np.asarray(y_pred, dtype=float)
    test_df[f"true_adjclose_{LOOKUP_STEP}"] = np.asarray(y_test, dtype=float)

    # Robust Series selection → numeric float
    adj_s  = _series_from_aliases(test_df, ["adjclose", "adj close", "close"])
    pred_s = _series_from_aliases(test_df, [f"adjclose_{LOOKUP_STEP}"])
    true_s = _series_from_aliases(test_df, [f"true_adjclose_{LOOKUP_STEP}"])

    adj_s  = pd.to_numeric(adj_s, errors="coerce").astype(float)
    pred_s = pd.to_numeric(pred_s, errors="coerce").astype(float)
    true_s = pd.to_numeric(true_s, errors="coerce").astype(float)

    # Write back cleaned series
    test_df["adjclose"] = adj_s
    test_df[f"adjclose_{LOOKUP_STEP}"] = pred_s
    test_df[f"true_adjclose_{LOOKUP_STEP}"] = true_s

    # Sort by time
    test_df.sort_index(inplace=True)

    # Vectorized profits
    cur  = test_df["adjclose"].to_numpy()
    pred = test_df[f"adjclose_{LOOKUP_STEP}"].to_numpy()
    true = test_df[f"true_adjclose_{LOOKUP_STEP}"].to_numpy()

    test_df["buy_profit"]  = np.where(pred > cur,  true - cur,  0.0)
    test_df["sell_profit"] = np.where(pred < cur,  cur - true,  0.0)

    return test_df







def predict(model, data):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][-N_STEPS:]
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    if SCALE:
        predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]
    return predicted_price


# load the data
data = load_data(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE,
                shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                feature_columns=FEATURE_COLUMNS)

# construct the model
model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

# load optimal model weights from results folder
import glob, os

weights_files = glob.glob("results/*.weights.h5")
if not weights_files:
    raise FileNotFoundError("No weights found in results/")
model_path = max(weights_files, key=os.path.getmtime)  # pick most recent
print("Loading:", model_path)
model.load_weights(model_path)

# evaluate the model
loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
# calculate the mean absolute error (inverse scaling)
if SCALE:
    mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
else:
    mean_absolute_error = mae

# get the final dataframe for the testing set
final_df = get_final_df(model, data)
# predict the future price
future_price = predict(model, data)
# we calculate the accuracy by counting the number of positive profits
accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buy_profit'] > 0])) / len(final_df)
# calculating total buy & sell profit
total_buy_profit  = final_df["buy_profit"].sum()
total_sell_profit = final_df["sell_profit"].sum()
# total profit by adding sell & buy together
total_profit = total_buy_profit + total_sell_profit
# dividing total profit by number of testing samples (number of trades)
profit_per_trade = total_profit / len(final_df)
# printing metrics
print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")
print(f"{LOSS} loss:", loss)
print("Mean Absolute Error:", mean_absolute_error)
print("Accuracy score:", accuracy_score)
print("Total buy profit:", total_buy_profit)
print("Total sell profit:", total_sell_profit)
print("Total profit:", total_profit)
print("Profit per trade:", profit_per_trade)
# plot true/pred prices graph
plot_graph(final_df)
print(final_df.tail(10))
# save the final dataframe to csv-results folder
csv_results_folder = "csv-results"
if not os.path.isdir(csv_results_folder):
    os.mkdir(csv_results_folder)
csv_filename = os.path.join(csv_results_folder, model_name + ".csv")
final_df.to_csv(csv_filename)
import matplotlib.pyplot as plt

# Plot Actual vs Predicted (LOOKUP_STEP-ahead)