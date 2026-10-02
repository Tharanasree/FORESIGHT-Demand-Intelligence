import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

files = list(DATA_DIR.glob("*.csv"))

if not files:
    print("No CSV files found in data/raw/")
else:
    for file_path in files:
        print("\n" + "=" * 60)
        print("DATASET:", file_path.name)
        print("=" * 60)

        df = pd.read_csv(file_path, nrows=5)

        print("\nColumns:")
        print(df.columns.tolist())

        print("\nFirst 5 rows:")
        print(df.head())

        print("\nShape of sample:")
        print(df.shape)

        print("\nMissing values in sample:")
        print(df.isnull().sum())