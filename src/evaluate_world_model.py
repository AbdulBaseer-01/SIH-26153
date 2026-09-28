import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

INPUT_FILE = "data/processed/network_states.csv"
MODEL_FILE = "models/world_model_predictor.keras"
SCALER_FILE = "models/world_model_scaler.pkl"

OUTPUT_FILE = "data/processed/world_model_scores.csv"

SEQUENCE_LENGTH = 10


print("Loading network states...")

df = pd.read_csv(INPUT_FILE)

df["TimeWindow"] = pd.to_datetime(df["TimeWindow"])
df = df.sort_values("TimeWindow").reset_index(drop=True)

print(f"States: {len(df)}")


# --------------------------------------------------
# Prepare features
# --------------------------------------------------

excluded = [
    "TimeWindow",
    "Infiltration Ratio",
    "IsInfiltrationState"
]

feature_cols = [
    col for col in df.columns
    if col not in excluded
]

print(f"Features: {len(feature_cols)}")


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

print("\nLoading world model...")

model = load_model(MODEL_FILE)

scaler = joblib.load(SCALER_FILE)

print("Model loaded.")
print("Scaler loaded.")


# --------------------------------------------------
# Create continuous sequences
# --------------------------------------------------

print("\nCreating evaluation sequences...")

X_sequences = []
y_actual = []
timestamps = []
labels = []

for i in range(len(df) - SEQUENCE_LENGTH):

    input_start = i
    input_end = i + SEQUENCE_LENGTH
    target_index = i + SEQUENCE_LENGTH

    input_times = df["TimeWindow"].iloc[input_start:input_end + 1]

    diffs = input_times.diff().dropna()

    # Require exactly 1-minute intervals
    if not all(diffs == pd.Timedelta(minutes=1)):
        continue

    X_sequences.append(
        X_raw[input_start:input_end]
    )

    y_actual.append(
        X_raw[target_index]
    )

    timestamps.append(
        df["TimeWindow"].iloc[target_index]
    )

    labels.append(
        df["IsInfiltrationState"].iloc[target_index]
    )


X_sequences = np.array(X_sequences)
y_actual = np.array(y_actual)

print(f"Evaluation sequences: {len(X_sequences)}")


# --------------------------------------------------
# Scale using training scaler
# --------------------------------------------------

X_scaled = scaler.transform(
    X_sequences.reshape(-1, X_sequences.shape[-1])
).reshape(X_sequences.shape)


y_actual_scaled = scaler.transform(y_actual)


# --------------------------------------------------
# Predict next state
# --------------------------------------------------

print("\nPredicting next states...")

y_pred_scaled = model.predict(
    X_scaled,
    verbose=1
)


# --------------------------------------------------
# Calculate errors
# --------------------------------------------------

mse_scores = np.mean(
    (y_pred_scaled - y_actual_scaled) ** 2,
    axis=1
)

mae_scores = np.mean(
    np.abs(y_pred_scaled - y_actual_scaled),
    axis=1
)


# --------------------------------------------------
# Save results
# --------------------------------------------------

results = pd.DataFrame({
    "TimeWindow": timestamps,
    "IsInfiltrationState": labels,
    "Prediction_MSE": mse_scores,
    "Prediction_MAE": mae_scores
})


results.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nEvaluation complete.")

print(f"Saved: {OUTPUT_FILE}")

print("\nScore summary:")
print(results["Prediction_MSE"].describe())


# --------------------------------------------------
# Normal vs attack
# --------------------------------------------------

normal_scores = results.loc[
    results["IsInfiltrationState"] == 0,
    "Prediction_MSE"
]

attack_scores = results.loc[
    results["IsInfiltrationState"] == 1,
    "Prediction_MSE"
]

print("\n==============================")
print("NORMAL STATES")
print("==============================")

print(normal_scores.describe())


print("\n==============================")
print("INFILTRATION STATES")
print("==============================")

print(attack_scores.describe())


# --------------------------------------------------
# Thresholds from PRE-ATTACK normal period
# --------------------------------------------------

pre_attack = results[
    (results["IsInfiltrationState"] == 0) &
    (results["TimeWindow"] < pd.Timestamp("2018-03-01 02:00:00"))
]

print("\n==============================")
print("PRE-ATTACK BASELINE")
print("==============================")

print(f"Samples: {len(pre_attack)}")

if len(pre_attack) > 0:

    baseline = pre_attack["Prediction_MSE"]

    for q in [0.90, 0.95, 0.99]:

        threshold = baseline.quantile(q)

        normal_detection_rate = (
            normal_scores > threshold
        ).mean()

        attack_detection_rate = (
            attack_scores > threshold
        ).mean()

        print(
            f"\n{int(q * 100)}th percentile threshold: "
            f"{threshold:.4f}"
        )

        print(
            f"Normal flagged: "
            f"{normal_detection_rate * 100:.2f}%"
        )

        print(
            f"Infiltration detected: "
            f"{attack_detection_rate * 100:.2f}%"
        )