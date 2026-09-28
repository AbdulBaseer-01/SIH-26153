import pandas as pd

ANOMALY_FILE = "data/processed/anomaly_features.csv"
TARGET_FILE = "data/processed/forecast_targets.csv"
OUTPUT_FILE = "data/processed/forecast_dataset.csv"


# ---------------------------------------------------------
# Load
# ---------------------------------------------------------

anomaly = pd.read_csv(ANOMALY_FILE)
targets = pd.read_csv(TARGET_FILE)

anomaly["TimeWindow"] = pd.to_datetime(
    anomaly["TimeWindow"]
)

targets["TimeWindow"] = pd.to_datetime(
    targets["TimeWindow"]
)


print(f"Anomaly rows: {len(anomaly)}")
print(f"Target rows:  {len(targets)}")


# ---------------------------------------------------------
# Select forecasting features
# ---------------------------------------------------------

feature_columns = [
    "TimeWindow",
    "Prediction_MSE",
    "Prediction_MAE",
    "MSE_Rolling_3",
    "MSE_Rolling_5",
    "MSE_Rolling_10",
    "MSE_Max_5",
    "MSE_Max_10",
    "MSE_Change_1",
    "MSE_Change_3",
    "MSE_Ratio_10",
    "Above_95",
    "Above_99",
    "Above95_Count_5",
    "Above95_Count_10",
    "IsInfiltrationState",
]

features = anomaly[feature_columns].copy()


# ---------------------------------------------------------
# Select targets
# ---------------------------------------------------------

target_columns = [
    "TimeWindow",
    "Infiltration_Next_5m",
    "Infiltration_Next_10m",
    "Infiltration_Next_15m",
    "Attack_Start_Next_5m",
    "Attack_Start_Next_10m",
    "Attack_Start_Next_15m",
]

target_data = targets[target_columns].copy()


# ---------------------------------------------------------
# Merge by timestamp
# ---------------------------------------------------------

df = features.merge(
    target_data,
    on="TimeWindow",
    how="inner"
)

df = df.sort_values(
    "TimeWindow"
).reset_index(drop=True)


# ---------------------------------------------------------
# Verify merge
# ---------------------------------------------------------

print(f"\nMerged rows: {len(df)}")

print(
    f"Time range: "
    f"{df['TimeWindow'].min()} -> "
    f"{df['TimeWindow'].max()}"
)


# ---------------------------------------------------------
# Check target counts
# ---------------------------------------------------------

print("\nAttack-start target counts:")

for horizon in [5, 10, 15]:

    col = f"Attack_Start_Next_{horizon}m"

    print(f"\n{col}")

    print(
        df[col]
        .value_counts(dropna=False)
        .sort_index()
    )


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)