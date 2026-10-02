import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "weekly_sales.csv"

df = pd.read_csv(DATA_FILE)
df["week"] = pd.to_datetime(df["week"])

df = df.sort_values(["id", "week"])

# Previous 4-week demand as the seasonal-naive forecast
df["baseline_forecast"] = df.groupby("id")["weekly_sales"].shift(4)

# Keep only rows where a baseline forecast exists
evaluation = df.dropna(subset=["baseline_forecast"]).copy()

# WAPE
wape = (
    (evaluation["weekly_sales"] - evaluation["baseline_forecast"]).abs().sum()
    / evaluation["weekly_sales"].abs().sum()
)

print("Baseline rows:", len(evaluation))
print(f"Seasonal-naive WAPE: {wape:.4f}")
print(f"Seasonal-naive WAPE (%): {wape * 100:.2f}%")