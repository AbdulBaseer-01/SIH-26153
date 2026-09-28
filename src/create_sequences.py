import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# ==============================
# CONFIG
# ==============================

INPUT_FILE = "data/processed/network_states.csv"

SEQUENCE_LENGTH = 10

OUTPUT_X = "data/processed/X_sequences.npy"
OUTPUT_Y = "data/processed/y_sequences.npy"

SCALER_FILE = "models/scaler.pkl"


# ==============================
# LOAD DATA
# ==============================

print("Loading network states...")

df = pd.read_csv(INPUT_FILE)

print(f"States loaded: {len(df)}")


# ==============================
# SELECT FEATURES
# ==============================

DROP_COLUMNS = [
    "TimeWindow",
    "Infiltration Ratio",
    "IsInfiltrationState"
]

feature_columns = [
    col for col in df.columns
    if col not in DROP_COLUMNS
]

X = df[feature_columns].copy()

y = df["IsInfiltrationState"].values

print(f"Features selected: {len(feature_columns)}")


# ==============================
# HANDLE MISSING / INFINITE VALUES
# ==============================

X = X.replace([np.inf, -np.inf], np.nan)

X = X.fillna(X.median())


# ==============================
# NORMALIZE FEATURES
# ==============================

print("Normalizing features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, SCALER_FILE)

print(f"Scaler saved to: {SCALER_FILE}")


# ==============================
# CREATE TEMPORAL SEQUENCES
# ==============================

print(f"Creating sequences of {SEQUENCE_LENGTH} minutes...")

X_sequences = []
y_sequences = []

for i in range(len(X_scaled) - SEQUENCE_LENGTH):

    sequence = X_scaled[i:i + SEQUENCE_LENGTH]

    # Predict the state AFTER the sequence
    target = y[i + SEQUENCE_LENGTH]

    X_sequences.append(sequence)
    y_sequences.append(target)


X_sequences = np.array(X_sequences)

y_sequences = np.array(y_sequences)


# ==============================
# SAVE
# ==============================

np.save(OUTPUT_X, X_sequences)

np.save(OUTPUT_Y, y_sequences)


# ==============================
# RESULTS
# ==============================

print("\n==============================")
print("SEQUENCE CREATION COMPLETE")
print("==============================")

print(f"\nX shape: {X_sequences.shape}")
print(f"y shape: {y_sequences.shape}")

print("\nMeaning:")

print(
    f"{X_sequences.shape[0]} sequences × "
    f"{X_sequences.shape[1]} time steps × "
    f"{X_sequences.shape[2]} features"
)

print("\nTarget distribution:")

unique, counts = np.unique(y_sequences, return_counts=True)

for label, count in zip(unique, counts):

    name = "INFILTRATION" if label == 1 else "NORMAL"

    print(f"{name}: {count}")

print("\nFiles saved:")
print(OUTPUT_X)
print(OUTPUT_Y)