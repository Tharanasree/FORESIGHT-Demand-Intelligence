import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

input_file = PROCESSED_DIR / "sales_long.csv"
output_file = PROCESSED_DIR / "weekly_sales.csv"

print("Loading sales data...")

df = pd.read_csv(
    input_file,
    usecols=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "date", "sales"]
)

df["date"] = pd.to_datetime(df["date"])

print("Aggregating weekly sales...")

df["week"] = df["date"].dt.to_period("W").dt.start_time

weekly = (
    df.groupby(
        ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "week"],
        as_index=False
    )["sales"]
    .sum()
    .rename(columns={"sales": "weekly_sales"})
)

weekly.to_csv(output_file, index=False)

print("Weekly data shape:", weekly.shape)
print("Saved:", output_file)