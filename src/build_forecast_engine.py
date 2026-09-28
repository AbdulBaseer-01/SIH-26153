import pandas as pd
import numpy as np


INPUT = "data/processed/anomaly_features.csv"
OUTPUT = "data/processed/forecast_engine_results.csv"


# =========================================================
# LOAD
# =========================================================

df = pd.read_csv(INPUT)

df["TimeWindow"] = pd.to_datetime(df["TimeWindow"])

df = (
    df.sort_values("TimeWindow")
      .reset_index(drop=True)
)


print(f"Loaded {len(df)} anomaly states")


# =========================================================
# BASELINE
# =========================================================

baseline = df[
    (df["TimeWindow"] >= "2018-03-01 01:10:00") &
    (df["TimeWindow"] <= "2018-03-01 01:59:00")
]["Prediction_MSE"].dropna()

THRESHOLD_95 = baseline.quantile(0.95)
THRESHOLD_99 = baseline.quantile(0.99)

print("\nBaseline:")
print(f"Samples: {len(baseline)}")
print(f"95th percentile: {THRESHOLD_95:.4f}")
print(f"99th percentile: {THRESHOLD_99:.4f}")


# =========================================================
# NORMALIZATION HELPERS
# =========================================================

def threshold_ratio(value, threshold):
    if pd.isna(value):
        return 0.0

    if threshold <= 0:
        return 0.0

    return value / threshold


def clamp(value, low=0, high=10):
    return max(low, min(high, value))


# =========================================================
# 5-MINUTE FORECAST
#
# Focus:
# - current anomaly
# - immediate change
# - recent 3/5 minute persistence
# =========================================================

def score_5m(row):

    score = 0.0

    mse = row["Prediction_MSE"]
    rolling_3 = row["MSE_Rolling_3"]
    rolling_5 = row["MSE_Rolling_5"]
    change_1 = row["MSE_Change_1"]
    count_5 = row["Above95_Count_5"]

    # Current anomaly
    if mse > THRESHOLD_99:
        score += 4
    elif mse > THRESHOLD_95:
        score += 2

    # Recent rolling behaviour
    if rolling_3 > THRESHOLD_99:
        score += 2.5
    elif rolling_3 > THRESHOLD_95:
        score += 1.5

    if rolling_5 > THRESHOLD_99:
        score += 2
    elif rolling_5 > THRESHOLD_95:
        score += 1

    # Sudden increase
    if change_1 > THRESHOLD_99:
        score += 2
    elif change_1 > THRESHOLD_95:
        score += 1

    # Recent persistence
    if count_5 >= 3:
        score += 2
    elif count_5 >= 2:
        score += 1

    return clamp(score)


# =========================================================
# 10-MINUTE FORECAST
#
# Focus:
# - persistence
# - rolling behaviour
# - anomaly ratio
# =========================================================

def score_10m(row):

    score = 0.0

    rolling_5 = row["MSE_Rolling_5"]
    rolling_10 = row["MSE_Rolling_10"]
    max_10 = row["MSE_Max_10"]
    ratio = row["MSE_Ratio_10"]
    count_10 = row["Above95_Count_10"]

    # Rolling 5
    if rolling_5 > THRESHOLD_99:
        score += 2
    elif rolling_5 > THRESHOLD_95:
        score += 1

    # Rolling 10
    if rolling_10 > THRESHOLD_99:
        score += 3
    elif rolling_10 > THRESHOLD_95:
        score += 2

    # Recent maximum
    if max_10 > THRESHOLD_99:
        score += 2
    elif max_10 > THRESHOLD_95:
        score += 1

    # Relative increase
    if ratio > 5:
        score += 3
    elif ratio > 3:
        score += 2
    elif ratio > 2:
        score += 1

    # Persistence
    if count_10 >= 5:
        score += 2
    elif count_10 >= 3:
        score += 1

    return clamp(score)


# =========================================================
# 15-MINUTE FORECAST
#
# Focus:
# - sustained anomaly
# - longer temporal behaviour
# - trend rather than single spikes
# =========================================================

def score_15m(row):

    score = 0.0

    rolling_10 = row["MSE_Rolling_10"]
    max_10 = row["MSE_Max_10"]
    ratio = row["MSE_Ratio_10"]
    count_10 = row["Above95_Count_10"]
    change_3 = row["MSE_Change_3"]

    # Long rolling behaviour
    if rolling_10 > THRESHOLD_99:
        score += 3
    elif rolling_10 > THRESHOLD_95:
        score += 2

    # Maximum anomaly over longer window
    if max_10 > THRESHOLD_99:
        score += 2
    elif max_10 > THRESHOLD_95:
        score += 1

    # Relative anomaly level
    if ratio > 5:
        score += 3
    elif ratio > 3:
        score += 2
    elif ratio > 2:
        score += 1

    # Sustained threshold crossings
    if count_10 >= 5:
        score += 3
    elif count_10 >= 3:
        score += 2
    elif count_10 >= 2:
        score += 1

    # Longer-term increase
    if change_3 > THRESHOLD_99:
        score += 2
    elif change_3 > THRESHOLD_95:
        score += 1

    return clamp(score)


# =========================================================
# STATE
# =========================================================

def state_from_score(score):

    if score >= 7:
        return "HIGH"

    if score >= 4:
        return "MEDIUM"

    return "LOW"


# =========================================================
# CALCULATE HORIZONS
# =========================================================

df["Risk_5m"] = df.apply(score_5m, axis=1)
df["Risk_10m"] = df.apply(score_10m, axis=1)
df["Risk_15m"] = df.apply(score_15m, axis=1)

df["Forecast_5m"] = df["Risk_5m"].apply(state_from_score)
df["Forecast_10m"] = df["Risk_10m"].apply(state_from_score)
df["Forecast_15m"] = df["Risk_15m"].apply(state_from_score)


# =========================================================
# OVERALL STATE
#
# Highest horizon risk becomes the dashboard state.
# =========================================================

df["Forecast_Score"] = df[
    ["Risk_5m", "Risk_10m", "Risk_15m"]
].max(axis=1)

df["Forecast_State"] = df["Forecast_Score"].apply(
    state_from_score
)


# =========================================================
# REASON GENERATOR
# =========================================================

def generate_reason(row):

    reasons = []

    if row["Risk_5m"] >= 7:
        reasons.append("strong immediate anomaly")

    elif row["Risk_5m"] >= 4:
        reasons.append("elevated short-term anomaly")

    if row["Risk_10m"] >= 7:
        reasons.append("persistent anomaly over the recent window")

    elif row["Risk_10m"] >= 4:
        reasons.append("developing anomaly pattern")

    if row["Risk_15m"] >= 7:
        reasons.append("sustained longer-term anomaly")

    ratio = row["MSE_Ratio_10"]

    if ratio > 5:
        reasons.append("error exceeds recent baseline by >5x")

    elif ratio > 3:
        reasons.append("error exceeds recent baseline by >3x")

    if row["Above95_Count_5"] >= 3:
        reasons.append("3+ elevated observations in 5 minutes")

    if not reasons:
        reasons.append("no strong temporal anomaly pattern")

    return "; ".join(reasons)


df["Forecast_Reason"] = df.apply(
    generate_reason,
    axis=1
)


# =========================================================
# GROUND TRUTH
# =========================================================

df["Ground_Truth"] = np.where(
    df["IsInfiltrationState"] == 1,
    "ATTACK",
    "NORMAL"
)


# =========================================================
# SAVE
# =========================================================

output_columns = [
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

    "Risk_5m",
    "Risk_10m",
    "Risk_15m",

    "Forecast_5m",
    "Forecast_10m",
    "Forecast_15m",

    "Forecast_Score",
    "Forecast_State",

    "Forecast_Reason",

    "IsInfiltrationState",
    "Ground_Truth",
]

result = df[output_columns]

result.to_csv(
    OUTPUT,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("OVERALL FORECAST STATE")
print("=" * 60)

print(
    result["Forecast_State"]
    .value_counts()
)


print("\n" + "=" * 60)
print("HORIZON DISTRIBUTION")
print("=" * 60)

for column in [
    "Forecast_5m",
    "Forecast_10m",
    "Forecast_15m"
]:

    print(f"\n{column}")
    print(
        result[column]
        .value_counts()
    )


# =========================================================
# ATTACK 2 PRE-WINDOW
# =========================================================

print("\n" + "=" * 60)
print("ATTACK 2 PRE-ATTACK WINDOW")
print("=" * 60)

attack2 = result[
    (result["TimeWindow"] >= "2018-03-01 09:40:00") &
    (result["TimeWindow"] <= "2018-03-01 09:56:00")
]

print(
    attack2[
        [
            "TimeWindow",
            "Prediction_MSE",
            "Risk_5m",
            "Risk_10m",
            "Risk_15m",
            "Forecast_5m",
            "Forecast_10m",
            "Forecast_15m",
        ]
    ].to_string(index=False)
)


# =========================================================
# ATTACK 1 PRE-WINDOW
# =========================================================

print("\n" + "=" * 60)
print("ATTACK 1 PRE-ATTACK WINDOW")
print("=" * 60)

attack1 = result[
    (result["TimeWindow"] >= "2018-03-01 01:40:00") &
    (result["TimeWindow"] <= "2018-03-01 01:59:00")
]

print(
    attack1[
        [
            "TimeWindow",
            "Prediction_MSE",
            "Risk_5m",
            "Risk_10m",
            "Risk_15m",
            "Forecast_5m",
            "Forecast_10m",
            "Forecast_15m",
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(OUTPUT)