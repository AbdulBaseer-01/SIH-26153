import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "data/processed/network_states.csv"

SEQUENCE_LENGTH = 10

OUTPUT_X = "data/processed/world_X_sequences.npy"
OUTPUT_Y = "data/processed/world_y_sequences.npy"

OUTPUT_TIMES = "data/processed/world_sequence_times.npy"

SCALER_FILE = "models/world_model_scaler.pkl"


# ============================================================
# LOAD NETWORK STATES
# ============================================================

print("Loading network states...")

df = pd.read_csv(INPUT_FILE)

print(f"States loaded: {len(df)}")


# ============================================================
# PARSE AND SORT TIME
# ============================================================

print("Parsing timestamps...")

df["TimeWindow"] = pd.to_datetime(
    df["TimeWindow"],
    errors="coerce"
)

df = df.dropna(subset=["TimeWindow"])

df = df.sort_values("TimeWindow").reset_index(drop=True)


# ============================================================
# SELECT MODEL FEATURES
# ============================================================

DROP_COLUMNS = [
    "TimeWindow",
    "Infiltration Ratio",
    "IsInfiltrationState"
]

feature_columns = [
    col for col in df.columns
    if col not in DROP_COLUMNS
]

print(f"Features selected: {len(feature_columns)}")

print("\nFeatures:")

for i, feature in enumerate(feature_columns, start=1):
    print(f"{i}. {feature}")


# ============================================================
# PREPARE FEATURES
# ============================================================

X = df[feature_columns].copy()

X = X.replace([np.inf, -np.inf], np.nan)

# Use training-normal data for median values
normal_mask = df["IsInfiltrationState"] == 0

normal_medians = X.loc[normal_mask].median()

X = X.fillna(normal_medians)

# Final safety fallback in case any column is still NaN
X = X.fillna(0)


# ============================================================
# GET LABELS AND TIMESTAMPS
# ============================================================

y = df["IsInfiltrationState"].astype(int).values

timestamps = df["TimeWindow"].values


# ============================================================
# FIND CONTINUOUS NORMAL TRAINING STATES
# ============================================================

print("\nFinding continuous normal states...")

normal_indices = []

for i in range(len(df) - SEQUENCE_LENGTH):

    # --------------------------------------------------------
    # Check that the complete 11-minute window is continuous
    #
    # 10 input states + 1 target state
    # --------------------------------------------------------

    window_times = timestamps[i:i + SEQUENCE_LENGTH + 1]

    time_diffs = np.diff(
        window_times.astype("datetime64[m]").astype(np.int64)
    )

    continuous = np.all(time_diffs == 1)

    if not continuous:
        continue

    # --------------------------------------------------------
    # World model training uses ONLY normal behavior.
    #
    # Therefore all 11 states must be normal:
    # 10 input states + predicted target state.
    # --------------------------------------------------------

    window_labels = y[i:i + SEQUENCE_LENGTH + 1]

    if np.all(window_labels == 0):
        normal_indices.append(i)


print(
    f"Continuous normal sequences available: "
    f"{len(normal_indices)}"
)


# ============================================================
# FIT SCALER ONLY ON NORMAL STATES
# ============================================================

print("\nFitting scaler on normal training states...")

normal_state_indices = []

for i in normal_indices:

    normal_state_indices.extend(
        range(i, i + SEQUENCE_LENGTH + 1)
    )

normal_state_indices = sorted(
    set(normal_state_indices)
)

X_normal = X.iloc[normal_state_indices]

scaler = StandardScaler()

scaler.fit(X_normal)

joblib.dump(
    scaler,
    SCALER_FILE
)

print(f"Scaler saved to: {SCALER_FILE}")


# ============================================================
# SCALE ALL STATES
# ============================================================

print("\nScaling network states...")

X_scaled = scaler.transform(X)


# ============================================================
# CREATE SEQUENCES
# ============================================================

print(
    f"\nCreating {SEQUENCE_LENGTH}-minute "
    "world-model sequences..."
)

X_sequences = []
y_sequences = []
sequence_times = []

for i in normal_indices:

    # --------------------------------------------------------
    # Input:
    # previous 10 network states
    # --------------------------------------------------------

    input_sequence = X_scaled[
        i:i + SEQUENCE_LENGTH
    ]

    # --------------------------------------------------------
    # Target:
    # the actual network state immediately after
    # those 10 states
    # --------------------------------------------------------

    target_state = X_scaled[
        i + SEQUENCE_LENGTH
    ]

    X_sequences.append(input_sequence)
    y_sequences.append(target_state)

    # Timestamp of the state being predicted
    sequence_times.append(
        timestamps[i + SEQUENCE_LENGTH]
    )


# Convert to NumPy arrays

X_sequences = np.array(
    X_sequences,
    dtype=np.float32
)

y_sequences = np.array(
    y_sequences,
    dtype=np.float32
)

sequence_times = np.array(
    sequence_times,
    dtype="datetime64[ns]"
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

np.save(
    OUTPUT_X,
    X_sequences
)

np.save(
    OUTPUT_Y,
    y_sequences
)

np.save(
    OUTPUT_TIMES,
    sequence_times
)


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("WORLD SEQUENCE CREATION COMPLETE")
print("========================================")

print(f"\nInput shape:  {X_sequences.shape}")
print(f"Target shape: {y_sequences.shape}")

print(
    f"\nMeaning:"
    f"\n{X_sequences.shape[0]} sequences"
    f"\n{X_sequences.shape[1]} previous minutes"
    f"\n{X_sequences.shape[2]} features"
    f"\n{y_sequences.shape[1]} predicted features"
)

print("\nAll sequences are:")

print("- Continuous 1-minute windows")
print("- Normal behavior only")
print("- 10 input states → 1 target state")

if len(sequence_times) > 0:

    print(
        f"\nTarget time range:"
        f"\n{sequence_times[0]}"
        f" → "
        f"{sequence_times[-1]}"
    )

print("\nFiles saved:")

print(OUTPUT_X)
print(OUTPUT_Y)
print(OUTPUT_TIMES)
print(SCALER_FILE)