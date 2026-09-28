import pandas as pd
import numpy as np
import pickle
from pathlib import Path

RAW_FILE = "data/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
OUTPUT_FILE = "models/live_feature_medians.pkl"

print("=" * 50)
print("SAVING LIVE FEATURE MEDIANS")
print("=" * 50)

df = pd.read_csv(RAW_FILE, low_memory=False)
df.columns = df.columns.str.strip()

# Remove accidental repeated header rows
df = df[df["Label"].astype(str).str.strip().str.lower() != "label"].copy()

# Timestamp
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="coerce",
    dayfirst=True
)

df = df.dropna(subset=["Timestamp"])

# Numeric conversion — exactly like preprocess_states.py
numeric_columns = [
    column for column in df.columns
    if column not in ["Timestamp", "Label"]
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Replace inf with NaN
df[numeric_columns] = df[numeric_columns].replace(
    [np.inf, -np.inf],
    np.nan
)

# Calculate medians from the complete cleaned dataset
medians = df[numeric_columns].median()

# Save
Path("models").mkdir(exist_ok=True)

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(medians.to_dict(), f)

print(f"Features saved: {len(medians)}")
print(f"Output: {OUTPUT_FILE}")

print("\nRelevant medians:")
print(f"Flow Byts/s : {medians['Flow Byts/s']}")
print(f"Flow Pkts/s : {medians['Flow Pkts/s']}")

print("\nNaN counts before imputation:")
print(f"Flow Byts/s : {df['Flow Byts/s'].isna().sum()}")
print(f"Flow Pkts/s : {df['Flow Pkts/s'].isna().sum()}")

print("\nDONE")