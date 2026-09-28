import pandas as pd
import numpy as np
import os

INPUT_FILE = "data/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
OUTPUT_FILE = "data/processed/network_states.csv"

# Time window for one network state
WINDOW_SIZE = "1min"

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"Original shape: {df.shape}")

# --------------------------------------------------
# 1. CLEAN COLUMN NAMES
# --------------------------------------------------

df.columns = df.columns.str.strip()

# Remove accidental repeated header rows
df = df[df["Label"] != "Label"].copy()

# --------------------------------------------------
# 2. PARSE TIMESTAMP
# --------------------------------------------------

print("Parsing timestamps...")

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    dayfirst=True,
    errors="coerce"
)

# Remove invalid timestamps
df = df.dropna(subset=["Timestamp"])

# --------------------------------------------------
# 3. CLEAN NUMERIC VALUES
# --------------------------------------------------

# Columns we should NOT convert to numeric
exclude_cols = ["Timestamp", "Label"]

numeric_cols = [
    col for col in df.columns
    if col not in exclude_cols
]

# Convert everything else to numeric
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Replace infinity with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Fill missing numeric values
df[numeric_cols] = df[numeric_cols].fillna(
    df[numeric_cols].median()
)

# --------------------------------------------------
# 4. SORT CHRONOLOGICALLY
# --------------------------------------------------

print("Sorting chronologically...")

df = df.sort_values("Timestamp").reset_index(drop=True)

# --------------------------------------------------
# 5. CREATE TIME WINDOWS
# --------------------------------------------------

print("Creating 1-minute network states...")

df["TimeWindow"] = df["Timestamp"].dt.floor(WINDOW_SIZE)

# --------------------------------------------------
# 6. AGGREGATE FEATURES
# --------------------------------------------------

# Mean value of every feature inside each minute
feature_states = df.groupby("TimeWindow")[numeric_cols].mean()

# --------------------------------------------------
# 7. ADD NETWORK ACTIVITY FEATURES
# --------------------------------------------------

# Number of flows in this network state
feature_states["Flow Count"] = (
    df.groupby("TimeWindow").size()
)

# --------------------------------------------------
# 8. CALCULATE INFILTRATION RISK
# --------------------------------------------------

df["IsInfiltration"] = (
    df["Label"]
    .astype(str)
    .str.strip()
    .str.lower()
    == "infilteration"
).astype(int)

feature_states["Infiltration Ratio"] = (
    df.groupby("TimeWindow")["IsInfiltration"].mean()
)

# Binary label for the state
feature_states["IsInfiltrationState"] = (
    feature_states["Infiltration Ratio"] > 0
).astype(int)

# --------------------------------------------------
# 9. SAVE NETWORK STATES
# --------------------------------------------------

feature_states = feature_states.reset_index()

os.makedirs("data/processed", exist_ok=True)

feature_states.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n==============================")
print("PREPROCESSING COMPLETE")
print("==============================")

print(f"\nNetwork states created: {len(feature_states)}")
print(f"Features per state: {len(feature_states.columns)}")

print("\nFirst 10 states:")
print(feature_states.head(10).to_string())

print("\nState label distribution:")
print(
    feature_states["IsInfiltrationState"]
    .value_counts()
)

print(f"\nSaved to: {OUTPUT_FILE}")