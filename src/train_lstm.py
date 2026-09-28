import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# ==============================
# CONFIG
# ==============================

X_FILE = "data/processed/X_sequences.npy"
Y_FILE = "data/processed/y_sequences.npy"

MODEL_FILE = "models/world_model_lstm.keras"

EPOCHS = 30
BATCH_SIZE = 16


# ==============================
# LOAD DATA
# ==============================

print("Loading sequences...")

X = np.load(X_FILE)
y = np.load(Y_FILE)

print("X shape:", X.shape)
print("y shape:", y.shape)


# ==============================
# TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==============================
# CLASS WEIGHTS
# ==============================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("\nClass weights:", class_weights)


# ==============================
# BUILD LSTM
# ==============================

print("\nBuilding World Model...")

model = Sequential([
    LSTM(
        64,
        input_shape=(X.shape[1], X.shape[2]),
        return_sequences=True
    ),

    Dropout(0.3),

    LSTM(32),

    Dropout(0.3),

    Dense(16, activation="relu"),

    Dense(1, activation="sigmoid")
])


# ==============================
# COMPILE
# ==============================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)


model.summary()


# ==============================
# CALLBACKS
# ==============================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_FILE,
    monitor="val_loss",
    save_best_only=True
)


# ==============================
# TRAIN
# ==============================

print("\nStarting training...\n")

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[
        early_stopping,
        checkpoint
    ],
    verbose=1
)


# ==============================
# EVALUATE
# ==============================

print("\n==============================")
print("EVALUATION")
print("==============================")

results = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")


print("\nModel saved to:")
print(MODEL_FILE)