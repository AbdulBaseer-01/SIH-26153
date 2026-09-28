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

X = df[feature_cols].copy()

X = X.replace([np.inf, -np.inf], np.nan)

normal_mask = df["IsInfiltrationState"] == 0

X = X.fillna(X.loc[normal_mask].median())
X = X.fillna(0)

X = X.values


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = load_model(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)


# --------------------------------------------------
# Build sequences
# --------------------------------------------------

inputs = []
actual = []
times = []
labels = []

for i in range(len(df) - SEQUENCE_LENGTH):

    times_check = df["TimeWindow"].iloc[
        i:i + SEQUENCE_LENGTH + 1
    ]

    if not all(
        times_check.diff().dropna()
        == pd.Timedelta(minutes=1)
    ):
        continue

    inputs.append(
        X[i:i + SEQUENCE_LENGTH]
    )

    actual.append(
        X[i + SEQUENCE_LENGTH]
    )

    times.append(
        df["TimeWindow"].iloc[i + SEQUENCE_LENGTH]
    )

    labels.append(
        df["IsInfiltrationState"].iloc[i + SEQUENCE_LENGTH]
    )


inputs = np.array(inputs)
actual = np.array(actual)


# --------------------------------------------------
# Scale
# --------------------------------------------------

inputs_scaled = scaler.transform(
    inputs.reshape(-1, inputs.shape[-1])
).reshape(inputs.shape)

actual_scaled = scaler.transform(actual)


# --------------------------------------------------
# Predict
# --------------------------------------------------

predicted = model.predict(
    inputs_scaled,
    verbose=0
)


# --------------------------------------------------
# Absolute feature errors
# --------------------------------------------------

feature_errors = np.abs(
    predicted - actual_scaled
)


# --------------------------------------------------
# Calculate different anomaly scores
# --------------------------------------------------

mse = np.mean(
    feature_errors ** 2,
    axis=1
)

top5 = np.mean(
    np.sort(feature_errors, axis=1)[:, -5:],
    axis=1
)

top10 = np.mean(
    np.sort(feature_errors, axis=1)[:, -10:],
    axis=1
)


results = pd.DataFrame({
    "TimeWindow": times,
    "IsInfiltrationState": labels,
    "MSE": mse,
    "Top5_Error": top5,
    "Top10_Error": top10
})


# --------------------------------------------------
# Save
# --------------------------------------------------

output = "data/processed/anomaly_score_comparison.csv"

results.to_csv(
    output,
    index=False
)

print(f"Saved: {output}")


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 65)
print("ANOMALY SCORE COMPARISON")
print("=" * 65)


for label, name in [
    (0, "NORMAL"),
    (1, "INFILTRATION")
]:

    subset = results[
        results["IsInfiltrationState"] == label
    ]

    print(f"\n{name}")

    print(
        f"MSE     mean={subset['MSE'].mean():.4f} "
        f"median={subset['MSE'].median():.4f}"
    )

    print(
        f"Top-5   mean={subset['Top5_Error'].mean():.4f} "
        f"median={subset['Top5_Error'].median():.4f}"
    )

    print(
        f"Top-10  mean={subset['Top10_Error'].mean():.4f} "
        f"median={subset['Top10_Error'].median():.4f}"
    )


# --------------------------------------------------
# Important pre-attack points
# --------------------------------------------------

print("\n" + "=" * 65)
print("PRE-ATTACK SIGNALS")
print("=" * 65)


interesting_times = [
    "2018-03-01 01:33:00",
    "2018-03-01 01:35:00",
    "2018-03-01 09:43:00",
    "2018-03-01 09:49:00",
    "2018-03-01 09:51:00"
]

for timestamp in interesting_times:

    row = results[
        results["TimeWindow"]
        == pd.Timestamp(timestamp)
    ]

    if len(row) == 0:
        continue

    row = row.iloc[0]

    print(
        f"\n{timestamp}"
    )

    print(
        f"Label  : {row['IsInfiltrationState']}"
    )

    print(
        f"MSE    : {row['MSE']:.4f}"
    )

    print(
        f"Top-5  : {row['Top5_Error']:.4f}"
    )

    print(
        f"Top-10 : {row['Top10_Error']:.4f}"
    )