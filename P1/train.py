from stock_prediction import create_model, load_data
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
import os, random, numpy as np, tensorflow as tf
import pandas as pd
from parameters import *

# --- reproducibility ---
SEED = 314
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# --- folders ---
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# --- data ---
data = load_data(
    ticker, N_STEPS,
    scale=SCALE, split_by_date=SPLIT_BY_DATE,
    shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
    feature_columns=FEATURE_COLUMNS
)

# save raw dataframe snapshot for the report
data["df"].to_csv(ticker_data_filename)

# --- model ---
model = create_model(
    N_STEPS, len(FEATURE_COLUMNS),
    loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL
)

# --- callbacks ---
checkpointer = ModelCheckpoint(
    os.path.join("results", model_name + ".weights.h5"),
    save_weights_only=True, save_best_only=True, monitor="val_loss", verbose=1
)
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# --- train ---
history = model.fit(
    data["X_train"], data["y_train"],
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(data["X_test"], data["y_test"]),
    callbacks=[checkpointer, tensorboard, early, reduce],
    verbose=1
)

# --- save training history for C.1 report ---
hist_df = pd.DataFrame(history.history)
hist_path = os.path.join("results", model_name + "_history.csv")
hist_df.to_csv(hist_path, index=False)
print("Saved training history to:", hist_path)