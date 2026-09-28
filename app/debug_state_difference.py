import numpy as np
import pandas as pd


RAW_FILE = "data/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
STATE_FILE = "data/processed/network_states.csv"


print("Loading raw data...")

raw = pd.read_csv(
    RAW_FILE,
    low_memory=False
)

raw.columns = raw.columns.str.strip()

# Same exclusions as preprocess_states.py
exclude_cols = [
    "Timestamp",
    "Label"
]

numeric_cols = [
    col
    for col in raw.columns
    if col not in exclude_cols
]

# EXACT SAME NUMERIC CLEANING
for col in numeric_cols:
    raw[col] = pd.to_numeric(
        raw[col],
        errors="coerce"
    )

raw = raw.replace(
    [np.inf, -np.inf],
    np.nan
)

raw[numeric_cols] = raw[numeric_cols].fillna(
    raw[numeric_cols].median()
)

# Timestamp
raw["Timestamp"] = pd.to_datetime(
    raw["Timestamp"],
    dayfirst=True,
    errors="coerce"
)

raw = raw.dropna(
    subset=["Timestamp"]
)

# EXACT SAME CHRONOLOGICAL SORT
raw = raw.sort_values(
    "Timestamp"
).reset_index(
    drop=True
)

# EXACT SAME 1-MINUTE WINDOW
raw["TimeWindow"] = raw[
    "Timestamp"
].dt.floor("1min")


print("Loading original states...")

states = pd.read_csv(
    STATE_FILE
)

states["TimeWindow"] = pd.to_datetime(
    states["TimeWindow"]
)


# --------------------------------------------------
# RECREATE ORIGINAL AGGREGATION
# --------------------------------------------------

raw_means = (
    raw.groupby("TimeWindow")[
        numeric_cols
    ]
    .mean()
)

raw_means["Flow Count"] = (
    raw.groupby("TimeWindow")
    .size()
)

raw_means = raw_means.reset_index()


# --------------------------------------------------
# COMPARE
# --------------------------------------------------

state_columns = [
    column
    for column in states.columns
    if column not in [
        "TimeWindow",
        "Infiltration Ratio",
        "IsInfiltrationState"
    ]
]

original = states[
    ["TimeWindow"] + state_columns
]

recreated = raw_means[
    ["TimeWindow"] + state_columns
]


merged = original.merge(
    recreated,
    on="TimeWindow",
    suffixes=(
        "_original",
        "_recreated"
    )
)


print()
print("==============================")
print("ORIGINAL PREPROCESSING DEBUG")
print("==============================")

print(
    "Original states:",
    len(original)
)

print(
    "Recreated states:",
    len(recreated)
)

print(
    "Matched states:",
    len(merged)
)


# --------------------------------------------------
# FEATURE DIFFERENCES
# --------------------------------------------------

differences = []

for column in state_columns:

    original_values = pd.to_numeric(
        merged[
            f"{column}_original"
        ],
        errors="coerce"
    )

    recreated_values = pd.to_numeric(
        merged[
            f"{column}_recreated"
        ],
        errors="coerce"
    )

    diff = (
        original_values
        - recreated_values
    ).abs()

    diff = diff.replace(
        [np.inf, -np.inf],
        np.nan
    )

    differences.append(
        (
            column,
            diff.max(),
            diff.mean()
        )
    )


differences.sort(
    key=lambda x: (
        -x[1]
        if pd.notna(x[1])
        else 0
    )
)


print()
print("Top feature differences:")

for column, max_diff, mean_diff in differences[:15]:

    print(
        f"{column}: "
        f"max={max_diff:.10f}, "
        f"mean={mean_diff:.10f}"
    )


# --------------------------------------------------
# FLOW COUNT
# --------------------------------------------------

flow_count_diff = (
    merged["Flow Count_original"]
    - merged["Flow Count_recreated"]
).abs()


print()
print(
    "Flow Count mismatches:",
    int(
        (flow_count_diff > 0).sum()
    )
)

print(
    "Maximum Flow Count difference:",
    int(
        flow_count_diff.max()
    )
)


# --------------------------------------------------
# OVERALL
# --------------------------------------------------

numeric_diffs = [
    mean_diff
    for _, _, mean_diff in differences
    if pd.notna(mean_diff)
]

overall_mean = np.mean(
    numeric_diffs
)


print()
print(
    "Overall mean absolute difference:",
    overall_mean
)

print("==============================")

if overall_mean < 1e-6:

    print(
        "ORIGINAL PREPROCESSING "
        "REPRODUCED EXACTLY"
    )

else:

    print(
        "ORIGINAL PREPROCESSING "
        "REPRODUCTION FAILED"
    )