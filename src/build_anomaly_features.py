import pandas as pd
import numpy as np

INPUT_FILE = "data/processed/world_model_scores.csv"
OUTPUT_FILE = "data/processed/anomaly_features.csv"

df = pd.read_csv(INPUT_FILE)

df["TimeWindow"] = pd.to_datetime(df["TimeWindow"])
df = df.sort_values("TimeWindow").reset_index(drop=True)

score = df["Prediction_MSE"]

# --------------------------------------------------
# Rolling anomaly features
# --------------------------------------------------

df["MSE_Rolling_3"] = (
    score.rolling(3, min_periods=1).mean()
)

df["MSE_Rolling_5"] = (
    score.rolling(5, min_periods=1).mean()
)

df["MSE_Rolling_10"] = (
    score.rolling(10, min_periods=1).mean()
)

# Maximum recent anomaly
df["MSE_Max_5"] = (
    score.rolling(5, min_periods=1).max()
)

df["MSE_Max_10"] = (
    score.rolling(10, min_periods=1).max()
)

# --------------------------------------------------
# Change / momentum
# --------------------------------------------------

df["MSE_Change_1"] = (
    score.diff()
)

df["MSE_Change_3"] = (
    score - score.shift(3)
)

# Ratio relative to recent baseline
rolling_baseline = (
    score.rolling(10, min_periods=3).median()
)

df["MSE_Ratio_10"] = (
    score / (rolling_baseline + 1e-6)
)

# --------------------------------------------------
# Threshold from pre-attack baseline
# --------------------------------------------------

baseline = df[
    (df["TimeWindow"] >= pd.Timestamp("2018-03-01 01:10:00")) &
    (df["TimeWindow"] < pd.Timestamp("2018-03-01 02:00:00"))
]["Prediction_MSE"]

threshold_95 = baseline.quantile(0.95)
threshold_99 = baseline.quantile(0.99)

df["Above_95"] = (
    score > threshold_95
).astype(int)

df["Above_99"] = (
    score > threshold_99
).astype(int)

# Number of threshold crossings in recent windows
df["Above95_Count_5"] = (
    df["Above_95"]
    .rolling(5, min_periods=1)
    .sum()
)

df["Above95_Count_10"] = (
    df["Above_95"]
    .rolling(10, min_periods=1)
    .sum()
)

# --------------------------------------------------
# Save
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Saved:", OUTPUT_FILE)

print("\nThresholds:")
print(f"95th percentile: {threshold_95:.4f}")
print(f"99th percentile: {threshold_99:.4f}")

print("\nFeature columns:")
for col in df.columns:
    print(" -", col)