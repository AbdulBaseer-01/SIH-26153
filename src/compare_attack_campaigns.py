import pandas as pd
import numpy as np

states = pd.read_csv("data/processed/network_states.csv")

# Remove time and label columns
feature_cols = [
    col for col in states.columns
    if col not in [
        "TimeWindow",
        "IsInfiltrationState"
    ]
]

# Define periods based on state indices
attack1 = states.iloc[60:157].copy()
attack2 = states.iloc[387:445].copy()

normal1 = states.iloc[0:50].copy()
normal2 = states.iloc[200:250].copy()

# Keep only numeric columns
numeric_cols = states.select_dtypes(include=[np.number]).columns.tolist()

# Remove target-related columns
exclude = [
    "IsInfiltrationState",
    "Infiltration Ratio"
]

numeric_cols = [c for c in numeric_cols if c not in exclude]

print("\n==============================")
print("CAMPAIGN COMPARISON")
print("==============================")

comparison = pd.DataFrame({
    "Attack_1_Mean": attack1[numeric_cols].mean(),
    "Attack_2_Mean": attack2[numeric_cols].mean(),
    "Normal_Mean": pd.concat([normal1, normal2])[numeric_cols].mean()
})

comparison["Attack_Difference_%"] = (
    abs(comparison["Attack_1_Mean"] - comparison["Attack_2_Mean"])
    /
    (abs(comparison["Attack_1_Mean"]) + 1e-9)
) * 100

comparison = comparison.sort_values(
    "Attack_Difference_%",
    ascending=False
)

print("\n===== TOP 20 FEATURES WHERE ATTACK 1 AND ATTACK 2 DIFFER MOST =====\n")

print(
    comparison[
        [
            "Attack_1_Mean",
            "Attack_2_Mean",
            "Normal_Mean",
            "Attack_Difference_%"
        ]
    ].head(20)
)

print("\n==============================")
print("FLOW COUNTS")
print("==============================")

print("\nAttack Campaign 1:")
print(attack1["Flow Count"].describe())

print("\nAttack Campaign 2:")
print(attack2["Flow Count"].describe())

print("\n==============================")
print("INFILTRATION RATIOS")
print("==============================")

print("\nAttack Campaign 1:")
print(attack1["Infiltration Ratio"].describe())

print("\nAttack Campaign 2:")
print(attack2["Infiltration Ratio"].describe())