import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)


INPUT = "data/processed/forecast_dataset.csv"

TARGET = "Attack_Start_Next_5m"

FEATURES = [
    "Prediction_MSE",
    "Prediction_MAE",
    "MSE_Rolling_3",
    "MSE_Rolling_5",
    "MSE_Rolling_10",
    "MSE_Max_5",
    "MSE_Max_10",
    "MSE_Change_1",
    "MSE_Change_3",
    "MSE_Ratio_10",
    "Above_95",
    "Above_99",
    "Above95_Count_5",
    "Above95_Count_10",
]


# ---------------------------------------------------------
# Load
# ---------------------------------------------------------

df = pd.read_csv(INPUT)

df["TimeWindow"] = pd.to_datetime(
    df["TimeWindow"]
)

print(f"Loaded {len(df)} rows")


# ---------------------------------------------------------
# Keep only normal states
# ---------------------------------------------------------

df = df[
    df["IsInfiltrationState"] == 0
].copy()

df = df.dropna(
    subset=[TARGET]
).copy()


# ---------------------------------------------------------
# Clean features
# ---------------------------------------------------------

df[FEATURES] = df[FEATURES].replace(
    [np.inf, -np.inf],
    np.nan
)

df[FEATURES] = df[FEATURES].fillna(0)


# ---------------------------------------------------------
# Campaign boundaries
# ---------------------------------------------------------

attack1_start = pd.Timestamp(
    "2018-03-01 02:00:00"
)

attack2_start = pd.Timestamp(
    "2018-03-01 09:57:00"
)


# ---------------------------------------------------------
# TRAIN = normal data around Attack 2
# TEST = normal data before Attack 1
# ---------------------------------------------------------

train = df[
    (
        df["TimeWindow"] >=
        pd.Timestamp("2018-03-01 09:00:00")
    )
    &
    (
        df["TimeWindow"] <
        attack2_start
    )
].copy()


test = df[
    (
        df["TimeWindow"] >=
        pd.Timestamp("2018-03-01 01:10:00")
    )
    &
    (
        df["TimeWindow"] <
        attack1_start
    )
].copy()


# ---------------------------------------------------------
# Prepare
# ---------------------------------------------------------

X_train = train[FEATURES]
y_train = train[TARGET].astype(int)

X_test = test[FEATURES]
y_test = test[TARGET].astype(int)


print("\nTRAIN — Attack 2 pre-attack period")
print(f"Rows: {len(train)}")
print(
    y_train.value_counts()
    .sort_index()
)


print("\nTEST — Attack 1 pre-attack period")
print(f"Rows: {len(test)}")
print(
    y_test.value_counts()
    .sort_index()
)


# ---------------------------------------------------------
# Scale
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ---------------------------------------------------------
# Train
# ---------------------------------------------------------

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)


# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------

probabilities = model.predict_proba(
    X_test_scaled
)[:, 1]

predictions = (
    probabilities >= 0.5
).astype(int)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("REVERSE 5-MINUTE FORECAST")
print("=" * 60)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

if len(np.unique(y_test)) > 1:

    print(
        f"ROC-AUC: "
        f"{roc_auc_score(y_test, probabilities):.4f}"
    )


# ---------------------------------------------------------
# Timeline around Attack 1
# ---------------------------------------------------------

results = test[
    [
        "TimeWindow",
        "Prediction_MSE",
        "MSE_Rolling_5",
        "MSE_Ratio_10",
        TARGET
    ]
].copy()

results["Forecast_Probability"] = probabilities

results["Forecast"] = predictions


print("\nForecast around Attack 1:")

print(
    results.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# Feature coefficients
# ---------------------------------------------------------

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Coefficient": model.coef_[0]
})

importance["AbsCoefficient"] = (
    importance["Coefficient"].abs()
)

importance = importance.sort_values(
    "AbsCoefficient",
    ascending=False
)

print("\nFeature coefficients:")

print(
    importance[
        [
            "Feature",
            "Coefficient"
        ]
    ].to_string(index=False)
)