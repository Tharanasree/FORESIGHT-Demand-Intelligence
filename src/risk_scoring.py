import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = DATA_DIR / "forecast_results.csv"
OUTPUT_FILE = DATA_DIR / "risk_results.csv"

df = pd.read_csv(INPUT_FILE)
df["week"] = pd.to_datetime(df["week"])

# Use recent observed demand to estimate inventory assumptions.
# These are illustrative assumptions, not actual inventory data.
df = df.sort_values(["id", "week"]).reset_index(drop=True)

recent_demand = (
    df.groupby("id")["weekly_sales"]
      .transform(lambda x: x.rolling(4, min_periods=1).mean())
)

df["assumed_on_hand"] = (recent_demand * 2).round().clip(lower=0)
df["assumed_on_order"] = 0
df["assumed_lead_time_weeks"] = 2

df["projected_lead_time_demand"] = (
    df["forecast"].clip(lower=0) * df["assumed_lead_time_weeks"]
)

df["available_inventory"] = (
    df["assumed_on_hand"] + df["assumed_on_order"]
)

df["stockout_risk"] = (
    df["available_inventory"] < df["projected_lead_time_demand"]
)

df["overstock_risk"] = (
    df["available_inventory"] > recent_demand * 4
)

df["risk_status"] = "Normal"
df.loc[df["overstock_risk"], "risk_status"] = "Overstock Risk"
df.loc[df["stockout_risk"], "risk_status"] = "Stockout Risk"

df.to_csv(OUTPUT_FILE, index=False)

print("Risk results saved:", OUTPUT_FILE)
print("\nRisk summary:")
print(df["risk_status"].value_counts())