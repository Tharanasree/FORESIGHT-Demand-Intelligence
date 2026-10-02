import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "weekly_sales.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "forecast_features.csv"

df = pd.read_csv(DATA_FILE)
df["week"] = pd.to_datetime(df["week"])

df = df.sort_values(["id", "week"])

# Lag features
for lag in [1, 2, 4, 8]:
    df[f"lag_{lag}"] = df.groupby("id")["weekly_sales"].shift(lag)

# Rolling demand features
df["rolling_mean_4"] = (
    df.groupby("id")["weekly_sales"]
      .transform(lambda x: x.shift(1).rolling(4).mean())
)

df["rolling_mean_8"] = (
    df.groupby("id")["weekly_sales"]
      .transform(lambda x: x.shift(1).rolling(8).mean())
)

# Calendar features
df["month"] = df["week"].dt.month
df["week_of_year"] = df["week"].dt.isocalendar().week.astype(int)

# Remove rows without enough history
df = df.dropna()

df.to_csv(OUTPUT_FILE, index=False)

print("Feature dataset shape:", df.shape)
print("Saved:", OUTPUT_FILE)    