import pandas as pd
import numpy as np

INPUT = "data/processed/network_states.csv"
OUTPUT = "data/processed/forecast_targets.csv"

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv(INPUT)

df["TimeWindow"] = pd.to_datetime(df["TimeWindow"])

df = df.sort_values("TimeWindow").reset_index(drop=True)

print(f"Loaded {len(df)} states")
print(
    f"Time range: {df['TimeWindow'].min()} "
    f"-> {df['TimeWindow'].max()}"
)


# ---------------------------------------------------------
# Helper: check whether a future window is continuous
# ---------------------------------------------------------

def is_continuous_window(df, start_idx, end_idx):

    if end_idx >= len(df):
        return False

    window_times = (
        df.loc[start_idx:end_idx, "TimeWindow"]
        .reset_index(drop=True)
    )

    expected = pd.Series(
        pd.date_range(
            start=df.loc[start_idx, "TimeWindow"],
            periods=end_idx - start_idx + 1,
            freq="min"
        )
    )

    return window_times.equals(expected)


# ---------------------------------------------------------
# Create "infiltration within next N minutes" targets
# ---------------------------------------------------------

for horizon in [5, 10, 15]:

    target = []

    for i in range(len(df)):

        end_idx = i + horizon

        # Not enough future data
        if end_idx >= len(df):
            target.append(np.nan)
            continue

        # Do not cross missing-data gaps
        if not is_continuous_window(df, i, end_idx):
            target.append(np.nan)
            continue

        # Look at the next N minutes
        future_states = df.loc[
            i + 1:end_idx,
            "IsInfiltrationState"
        ]

        # 1 = infiltration occurs within next N minutes
        # 0 = no infiltration within next N minutes
        target.append(
            int(future_states.max() > 0)
        )

    df[f"Infiltration_Next_{horizon}m"] = target


# ---------------------------------------------------------
# Create "attack starts within next N minutes" targets
# ---------------------------------------------------------

for horizon in [5, 10, 15]:

    target = []

    for i in range(len(df)):

        end_idx = i + horizon

        # Not enough future data
        if end_idx >= len(df):
            target.append(np.nan)
            continue

        # Do not cross missing-data gaps
        if not is_continuous_window(df, i, end_idx):
            target.append(np.nan)
            continue

        current_state = df.loc[
            i,
            "IsInfiltrationState"
        ]

        # We are interested in forecasting a NEW attack.
        # If we are already inside an attack, this is not
        # an attack-start prediction sample.
        if current_state != 0:
            target.append(0)
            continue

        future_states = df.loc[
            i + 1:end_idx,
            "IsInfiltrationState"
        ].values

        attack_start = False

        previous_state = current_state

        for state in future_states:

            # Normal -> infiltration transition
            if previous_state == 0 and state == 1:
                attack_start = True
                break

            previous_state = state

        target.append(
            int(attack_start)
        )

    df[f"Attack_Start_Next_{horizon}m"] = target


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)


# ---------------------------------------------------------
# Print statistics
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("INFILTRATION TARGETS")
print("=" * 60)

for horizon in [5, 10, 15]:

    col = f"Infiltration_Next_{horizon}m"

    print(f"\n{col}")

    print(
        df[col]
        .value_counts(dropna=False)
        .sort_index()
    )


print("\n" + "=" * 60)
print("ATTACK START TARGETS")
print("=" * 60)

for horizon in [5, 10, 15]:

    col = f"Attack_Start_Next_{horizon}m"

    print(f"\n{col}")

    print(
        df[col]
        .value_counts(dropna=False)
        .sort_index()
    )


# ---------------------------------------------------------
# Show actual attack-start examples
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("ATTACK START EXAMPLES")
print("=" * 60)

for horizon in [5, 10, 15]:

    col = f"Attack_Start_Next_{horizon}m"

    positives = df[
        df[col] == 1
    ][
        [
            "TimeWindow",
            "IsInfiltrationState",
            col
        ]
    ]

    print(f"\n{col}:")
    print(positives.to_string(index=False))