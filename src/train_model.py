import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "forecast_features.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)
df["week"] = pd.to_datetime(df["week"])

# Use the final 6 weeks as the test period
last_week = df["week"].max()
test_start = last_week - pd.Timedelta(weeks=5)

train = df[df["week"] < test_start].copy()
test = df[df["week"] >= test_start].copy()

features = [
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",
    "rolling_mean_4",
    "rolling_mean_8",
    "month",
    "week_of_year"
]

X_train = train[features]
y_train = train["weekly_sales"]

X_test = test[features]
y_test = test["weekly_sales"]

print("Training rows:", len(train))
print("Testing rows:", len(test))

model = HistGradientBoostingRegressor(
    max_iter=150,
    learning_rate=0.08,
    max_leaf_nodes=31,
    random_state=42
)

print("Training model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
predictions = predictions.clip(min=0)

# WAPE
wape = (
    abs(y_test - predictions).sum()
    / abs(y_test).sum()
)

mae = mean_absolute_error(y_test, predictions)

print(f"ML Model WAPE: {wape * 100:.2f}%")
print(f"ML Model MAE: {mae:.2f}")

joblib.dump(model, MODEL_DIR / "demand_forecast_model.pkl")

results = test[["id", "week", "weekly_sales"]].copy()
results["forecast"] = predictions
results.to_csv(
    BASE_DIR / "data" / "processed" / "forecast_results.csv",
    index=False
)

print("Model saved.")
print("Forecast results saved.")