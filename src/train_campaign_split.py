import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight

# ==============================
# LOAD DATA
# ==============================

print("Loading sequences...")

X = np.load("data/processed/X_sequences.npy")
y = np.load("data/processed/y_sequences.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

# ==============================
# CAMPAIGN-BASED SPLIT
# ==============================

# Attack Block 1:
# Sequence 50 → 146

# Attack Block 2:
# Sequence 377 → 434

# We train using everything BEFORE the second campaign,
# except the last 10 sequences as a safety buffer.

BUFFER = 10

# Training data:
# 0 → 366
train_end = 377 - BUFFER

train_indices = np.arange(0, train_end)

# Testing data:
# Normal before attack + Attack Block 2 + Normal after attack

test_start = 377 - 30
test_end = 435 + 30

# Safety limits
test_start = max(test_start, 0)
test_end = min(test_end, len(X))

test_indices = np.arange(test_start, test_end)

# Remove any overlap
test_indices = np.setdiff1d(test_indices, train_indices)

X_train = X[train_indices]
y_train = y[train_indices]

X_test = X[test_indices]
y_test = y[test_indices]

print("\n==============================")
print("CAMPAIGN-BASED SPLIT")
print("==============================")

print("\nTraining sequences:")
print(f"{train_indices[0]} → {train_indices[-1]}")

print("Testing sequences:")
print(f"{test_indices[0]} → {test_indices[-1]}")

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
# EARLY STOPPING
# ==============================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ==============================
# TRAIN
# ==============================

print("\nStarting training...\n")

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=16,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# ==============================
# PREDICTIONS
# ==============================

print("\nGenerating predictions...")

probabilities = model.predict(X_test)

predictions = (probabilities >= 0.5).astype(int).flatten()

# ==============================
# EVALUATION
# ==============================

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

cm = confusion_matrix(y_test, predictions)

# False positive rate
tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

print("\n==============================")
print("FINAL EVALUATION RESULTS")
print("==============================")

print(f"Accuracy:           {accuracy:.4f}")
print(f"Precision:          {precision:.4f}")
print(f"Recall:             {recall:.4f}")
print(f"F1 Score:           {f1:.4f}")
print(f"False Positive Rate:{fpr:.4f}")

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
# SAVE MODEL
# ==============================

model.save("models/world_model_campaign.keras")

print("\nModel saved to:")
print("models/world_model_campaign.keras")