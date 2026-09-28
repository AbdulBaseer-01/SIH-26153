import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==============================
# CONFIG
# ==============================

X_FILE = "data/processed/X_sequences.npy"
Y_FILE = "data/processed/y_sequences.npy"

MODEL_FILE = "models/world_model_lstm_chronological.keras"

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
# CHRONOLOGICAL SPLIT
# ==============================

split_index = int(len(X) * 0.80)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print("\n==============================")
print("CHRONOLOGICAL SPLIT")
print("==============================")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining distribution:")
print("Normal:", np.sum(y_train == 0))
print("Infiltration:", np.sum(y_train == 1))

print("\nTesting distribution:")
print("Normal:", np.sum(y_test == 0))
print("Infiltration:", np.sum(y_test == 1))


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
# BUILD MODEL
# ==============================

print("\nBuilding LSTM World Model...")

model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),

    LSTM(64, return_sequences=True),
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
    metrics=["accuracy"]
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

    validation_split=0.15,

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
# PREDICTIONS
# ==============================

print("\nGenerating predictions...")

probabilities = model.predict(X_test)

probabilities = probabilities.flatten()

predictions = (probabilities >= 0.5).astype(int)


# ==============================
# METRICS
# ==============================

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

cm = confusion_matrix(y_test, predictions)

tn, fp, fn, tp = cm.ravel()

false_positive_rate = fp / (fp + tn)


# ==============================
# RESULTS
# ==============================

print("\n==============================")
print("FINAL EVALUATION RESULTS")
print("==============================")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"False Positive Rate: {false_positive_rate:.4f}")


print("\n===== CONFUSION MATRIX =====")

print(cm)


print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Normal", "Infiltration"],
        zero_division=0
    )
)


# ==============================
# SAVE PREDICTIONS
# ==============================

np.save(
    "data/processed/lstm_test_probabilities.npy",
    probabilities
)

np.save(
    "data/processed/lstm_test_predictions.npy",
    predictions
)


print("\nModel saved to:")
print(MODEL_FILE)