import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt


# specify dataset location and folder for results
DATA_PATH = "Taskc7/final_dataset.csv"
OUT_DIR = "evidence"
os.makedirs(OUT_DIR, exist_ok=True)  

# read data and sort by date so training uses older data first
data = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

# define sets of input features for testing
price_features = ["ret_1d", "ma_5", "ma_10", "mom_5", "vol_5", "RSI_14"]  # technical indicators
sentiment_features = ["vader_mean"]  # we can use vader_mean or finbert mean to train the model .

# dictionary with two feature combinations
feature_sets = {
    "PriceOnly": price_features,
    "PricePlusSentiment": price_features + sentiment_features,
}

# choose which feature set to use for this model run
config_name =  "PricePlusSentiment"
X = data[feature_sets[config_name]].copy()  # independent variables
y = data["target"].astype(int)              # target variable (0 = down, 1 = up)

# split data into train and test based on time order, not random
split_ratio = 0.8
split_point = int(len(X) * split_ratio)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

# standardize data so each feature has mean 0 and std 1
# this is important for logistic regression since it’s sensitive to feature scales
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# build logistic regression model
# use 'balanced' class weight so model handles uneven target distribution better
# increase max_iter to ensure convergence
model = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")
model.fit(X_train_scaled, y_train)  # fit the model to the training data

# generate predictions on the test set
y_pred = model.predict(X_test_scaled)

# evaluate the model using several metrics
accuracy = accuracy_score(y_test, y_pred)       # overall correctness
precision = precision_score(y_test, y_pred, zero_division=0)  # how many predicted ups were real ups
recall = recall_score(y_test, y_pred, zero_division=0)        # how many real ups were found
f1 = f1_score(y_test, y_pred, zero_division=0)                # balance of precision and recall
conf_mat = confusion_matrix(y_test, y_pred)                   # counts of TP, TN, FP, FN

# compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted Down (0)", "Predicted Up (1)"],
            yticklabels=["Actual Down (0)", "Actual Up (1)"])
plt.title(f"Confusion Matrix - {config_name}")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.show()



# show key results
print("\n---------------- Logistic Regression ----------------")
print(f"Feature Set : {config_name}")
print(f"Accuracy    : {accuracy:.3f}")
print(f"Precision   : {precision:.3f}")
print(f"Recall      : {recall:.3f}")
print(f"F1 Score    : {f1:.3f}")
print("\nConfusion Matrix [TN FP; FN TP]:")
print(conf_mat)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3, zero_division=0))  # detailed per-class results

# save metrics in a small summary table
metrics = pd.DataFrame([{
    "FeatureSet": config_name,
    "Model": "LogisticRegression",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1
}])
metrics.to_csv(f"{OUT_DIR}/lr_results_{config_name}.csv", index=False)

# extract and save feature coefficients to see which inputs mattered most
coef_df = pd.Series(model.coef_[0], index=feature_sets[config_name]).sort_values(ascending=False)
coef_df.to_csv(f"{OUT_DIR}/lr_coefficients_{config_name}.csv", header=["coefficient"])

# confirm successful completion
print("\nSaved metrics and coefficients to 'evidence/'")
print("----------------------------------------------------")


