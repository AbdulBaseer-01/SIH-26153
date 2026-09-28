import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

INPUT_FILE = "data/processed/network_states.csv"
MODEL_FILE = "models/world_model_predictor.keras"
SCALER_FILE = "models/world_model_scaler.pkl"

SEQUENCE_LENGTH = 10


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["TimeWindow"] = pd.to_datetime(df["TimeWindow"])
df = df.sort_values("TimeWindow").reset_index(drop=True)


excluded = [
    "TimeWindow",
    "Infiltration Ratio",
    "IsInfiltrationState"
]

feature_cols = [
    col for col in df.columns
    if col not in excluded
]


X_raw = df[feature_cols].copy()

X_raw = X_raw.replace([np.inf, -np.inf], np.nan)

normal_mask = df["IsInfiltrationState"] == 0

normal_medians = X_raw.loc[normal_mask].median()

X_raw = X_raw.fillna(normal_medians)
X_raw = X_raw.fillna(0)

X_raw = X_raw.values


# --------------------------------------------------
# Load model + scaler
# --------------------------------------------------

model = load_model(MODEL_FILE)

scaler = joblib.load(SCALER_FILE)


# --------------------------------------------------
# Create continuous sequences
# --------------------------------------------------

X_sequences = []
actual_states = []
timestamps = []
labels = []

for i in range(len(df) - SEQUENCE_LENGTH):

    input_times = df["TimeWindow"].iloc[
        i:i + SEQUENCE_LENGTH + 1
    ]

    diffs = input_times.diff().dropna()

    if not all(
        diffs == pd.Timedelta(minutes=1)
    ):
        continue

    X_sequences.append(
        X_raw[i:i + SEQUENCE_LENGTH]
    )

    actual_states.append(
        X_raw[i + SEQUENCE_LENGTH]
    )

    timestamps.append(
        df["TimeWindow"].iloc[i + SEQUENCE_LENGTH]
    )

    labels.append(
        df["IsInfiltrationState"].iloc[i + SEQUENCE_LENGTH]
    )


X_sequences = np.array(X_sequences)
actual_states = np.array(actual_states)


# --------------------------------------------------
# Scale
# --------------------------------------------------

X_scaled = scaler.transform(
    X_sequences.reshape(
        -1,
        X_sequences.shape[-1]
    )
).reshape(X_sequences.shape)

actual_scaled = scaler.transform(
    actual_states
)


# --------------------------------------------------
# Predict
# --------------------------------------------------

predicted_scaled = model.predict(
    X_scaled,
    verbose=0
)


# --------------------------------------------------
# Feature-level errors
# --------------------------------------------------

errors = np.abs(
    predicted_scaled - actual_scaled
)


# --------------------------------------------------
# Find top contributing features
# --------------------------------------------------

def show_features(timestamp):

    matches = [
        i for i, t in enumerate(timestamps)
        if t == timestamp
    ]

    if not matches:
        print("Timestamp not found.")
        return

    idx = matches[0]

    feature_errors = errors[idx]

    ranking = np.argsort(
        feature_errors
    )[::-1]

    print("\n" + "=" * 70)
    print(f"FEATURE ERROR ANALYSIS — {timestamp}")
    print("=" * 70)

    print(
        f"\nAttack state: "
        f"{labels[idx]}"
    )

    print("\nTop 15 contributing features:\n")

    for rank, feature_index in enumerate(
        ranking[:15],
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"{feature_cols[feature_index]:35s}"
            f" error={feature_errors[feature_index]:.4f}"
        )


# --------------------------------------------------
# Analyze important points
# --------------------------------------------------

show_features(
    pd.Timestamp("2018-03-01 02:01:00")
)

show_features(
    pd.Timestamp("2018-03-01 02:10:00")
)

show_features(
    pd.Timestamp("2018-03-01 09:58:00")
)

show_features(
    pd.Timestamp("2018-03-01 10:05:00")
)

show_features(
    pd.Timestamp("2018-03-01 09:51:00")
)