import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# CONFIG
# ============================================================

X_FILE = "data/processed/world_X_sequences.npy"
Y_FILE = "data/processed/world_y_sequences.npy"

MODEL_FILE = "models/world_model_predictor.keras"

EPOCHS = 100
BATCH_SIZE = 16

VALIDATION_SPLIT = 0.2


# ============================================================
# LOAD DATA
# ============================================================

print("Loading world-model sequences...")

X = np.load(X_FILE)
y = np.load(Y_FILE)

print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# CHRONOLOGICAL TRAIN / VALIDATION SPLIT
# ============================================================

split_index = int(
    len(X) * (1 - VALIDATION_SPLIT)
)

X_train = X[:split_index]
y_train = y[:split_index]

X_val = X[split_index:]
y_val = y[split_index:]

print("\nChronological split:")

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))

print("\nTraining range:")
print("First:", X_train.shape)

print("\nValidation range:")
print("First:", X_val.shape)


# ============================================================
# BUILD WORLD MODEL
# ============================================================

print("\nBuilding world model...")

model = Sequential([

    LSTM(
        128,
        input_shape=(
            X.shape[1],
            X.shape[2]
        ),
        return_sequences=True
    ),

    Dropout(0.2),

    LSTM(64),

    Dropout(0.2),

    Dense(128, activation="relu"),

    Dense(
        y.shape[1],
        activation="linear"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss=tf.keras.losses.Huber(),
    metrics=[
        tf.keras.metrics.MeanSquaredError(
            name="mse"
        ),
        tf.keras.metrics.MeanAbsoluteError(
            name="mae"
        )
    ]
)


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_FILE,
    monitor="val_loss",
    save_best_only=True
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...\n")

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping,
        checkpoint
    ],

    verbose=1
)


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n========================================")
print("WORLD MODEL EVALUATION")
print("========================================")

results = model.evaluate(
    X_val,
    y_val,
    verbose=0
)

for name, value in zip(
    model.metrics_names,
    results
):
    print(
        f"{name}: {value:.6f}"
    )


# ============================================================
# SAVE
# ============================================================

print("\nModel saved to:")

print(MODEL_FILE)