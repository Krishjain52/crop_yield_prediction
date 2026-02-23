import pandas as pd
import numpy as np
import joblib
import os
import random
import matplotlib.pyplot as plt
random.seed(42)
np.random.seed(42)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# Create models folder
os.makedirs("models", exist_ok=True)


# Load Dataset
df = pd.read_csv("data/yield_df.csv")

# Drop unwanted index column
if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)

df = df.dropna()


# Feature Engineering
df["rain_temp_ratio"] = (
    df["average_rain_fall_mm_per_year"] /
    (df["avg_temp"] + 1)
)


# One-hot Encoding
df = pd.get_dummies(df, drop_first=True)


# Define Target
TARGET = "hg/ha_yield"

X = df.drop(TARGET, axis=1)
y = df[TARGET]


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train Random Forest

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest Results")
print("MAE :", rf_mae)
print("RMSE:", rf_rmse)
print("R2  :", rf_r2)


# Train XGBoost

xgb = XGBRegressor(random_state=42)
xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2 = r2_score(y_test, xgb_pred)

print("\nXGBoost Results")
print("MAE :", xgb_mae)
print("RMSE:", xgb_rmse)
print("R2  :", xgb_r2)


# Select Best Model

if xgb_r2 > rf_r2:
    best_model = xgb
    best_name = "XGBoost"
    best_mae = xgb_mae
    best_rmse = xgb_rmse
    best_r2 = xgb_r2
else:
    best_model = rf
    best_name = "Random Forest"
    best_mae = rf_mae
    best_rmse = rf_rmse
    best_r2 = rf_r2

print(f"\nBest Model Selected: {best_name}")

importances = best_model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns)

feat_imp.sort_values(ascending=False).head(10).plot(kind="barh")
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.savefig("models/feature_importance.png")
plt.close()
# Save Model + Features + Metrics

joblib.dump(best_model, "models/model.pkl")
joblib.dump(X.columns.tolist(), "models/features.pkl")
joblib.dump(best_name, "models/model_name.pkl")

metrics = {
    "mae": best_mae,
    "rmse": best_rmse,
    "r2": best_r2
}

joblib.dump(metrics, "models/metrics.pkl")

print("\nModel saved successfully!")