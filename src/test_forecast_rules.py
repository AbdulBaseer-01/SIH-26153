import pandas as pd
import numpy as np


# ============================================================
# LOAD DATA
# ============================================================

INPUT_FILE = "data/processed/anomaly_features.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=["TimeWindow"])

print("=" * 70)
print("NETWORK ATTACK FORECAST RULE EVALUATION")
print("=" * 70)

# ------------------------------------------------------------
# Baseline thresholds
# Use the known pre-attack normal period: 01:10 - 01:59
# ------------------------------------------------------------

baseline = df[
    (df["TimeWindow"] >= "2018-03-01 01:10:00") &
    (df["TimeWindow"] <= "2018-03-01 01:59:00")
]

threshold_95 = baseline["Prediction_MSE"].quantile(0.95)
threshold_99 = baseline["Prediction_MSE"].quantile(0.99)

print(f"\n95th percentile threshold: {threshold_95:.4f}")
print(f"99th percentile threshold: {threshold_99:.4f}")
print(f"Baseline samples: {len(baseline)}")


# ============================================================
# FORECAST RULE
# ============================================================

def classify(row):

    mse = row["Prediction_MSE"]
    ratio = row["MSE_Ratio_10"]
    count_5 = row["Above95_Count_5"]
    count_10 = row["Above95_Count_10"]
    rolling_5 = row["MSE_Rolling_5"]
    change_3 = row["MSE_Change_3"]

    # --------------------------------------------------------
    # ELEVATED
    # Strong current anomaly OR repeated anomalies
    # --------------------------------------------------------

    if mse > threshold_99 and ratio > 3:
        return "ELEVATED"

    if count_5 >= 2 or count_10 >= 3:
        return "ELEVATED"

    # --------------------------------------------------------
    # WATCH
    # Growing anomaly but not strong enough for elevated
    # --------------------------------------------------------

    if (
        mse > threshold_95
        and change_3 > 0
        and ratio > 2
    ):
        return "WATCH"

    if rolling_5 > threshold_95:
        return "WATCH"

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    return "NORMAL"


df["ForecastState"] = df.apply(classify, axis=1)


# ============================================================
# HELPER
# ============================================================

def show_period(title, start, end):

    period = df[
        (df["TimeWindow"] >= start) &
        (df["TimeWindow"] < end)
    ].copy()

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    if len(period) == 0:
        print("No data.")
        return

    print(
        period[
            [
                "TimeWindow",
                "IsInfiltrationState",
                "Prediction_MSE",
                "MSE_Ratio_10",
                "Above95_Count_5",
                "Above95_Count_10",
                "ForecastState",
            ]
        ].to_string(index=False)
    )


# ============================================================
# PRE-ATTACK WINDOWS
# ============================================================

show_period(
    "ATTACK 1 — PRE-ATTACK WINDOW",
    "2018-03-01 01:30:00",
    "2018-03-01 02:00:00",
)

show_period(
    "ATTACK 2 — PRE-ATTACK WINDOW",
    "2018-03-01 09:27:00",
    "2018-03-01 09:57:00",
)


# ============================================================
# ATTACK WINDOWS
# ============================================================

show_period(
    "ATTACK 1 — ATTACK PERIOD",
    "2018-03-01 02:00:00",
    "2018-03-01 03:37:00",
)

show_period(
    "ATTACK 2 — ATTACK PERIOD",
    "2018-03-01 09:57:00",
    "2018-03-01 10:55:00",
)


# ============================================================
# OVERALL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("OVERALL FORECAST STATE COUNTS")
print("=" * 70)

print(
    df["ForecastState"]
    .value_counts()
    .to_string()
)


# ============================================================
# GROUND TRUTH COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FORECAST STATE vs GROUND TRUTH")
print("=" * 70)

confusion = pd.crosstab(
    df["ForecastState"],
    df["IsInfiltrationState"],
    rownames=["ForecastState"],
    colnames=["Ground Truth"],
)

print(confusion)


# ============================================================
# STATE COUNTS BY GROUND TRUTH
# ============================================================

print("\n" + "=" * 70)
print("FORECAST STATES WITHIN NORMAL / ATTACK PERIODS")
print("=" * 70)

for label, name in [(0, "NORMAL"), (1, "INFILTRATION")]:

    subset = df[df["IsInfiltrationState"] == label]

    print(f"\n{name}: {len(subset)} samples")

    print(
        subset["ForecastState"]
        .value_counts()
        .to_string()
    )


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE = "data/processed/forecast_rule_results.csv"

df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print(f"Saved: {OUTPUT_FILE}")
print("=" * 70)