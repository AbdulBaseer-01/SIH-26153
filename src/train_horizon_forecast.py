import os
import pickle

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


INPUT = "data/processed/forecast_dataset.csv"
MODEL_DIR = "models"

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

TARGETS = {
    "5m": "Attack_Start_Next_5m",
    "10m": "Attack_Start_Next_10m",
    "15m": "Attack_Start_Next_15m",
}

ATTACK_1_START = pd.Timestamp(
    "2018-03-01 02:00:00"
)

ATTACK_2_START = pd.Timestamp(
    "2018-03-01 09:57:00"
)


def load_data():

    df = pd.read_csv(INPUT)

    df["TimeWindow"] = pd.to_datetime(
        df["TimeWindow"]
    )

    # Only normal states can be used for
    # predicting a future attack.
    df = df[
        df["IsInfiltrationState"] == 0
    ].copy()

    for feature in FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

    df[FEATURES] = (
        df[FEATURES]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    return df


def train_one_horizon(
    df,
    horizon,
    target,
):

    print()
    print("=" * 70)
    print(f"{horizon}-MINUTE FORECAST")
    print("=" * 70)

    # ---------------------------------------------------------
    # TRAIN
    # Attack 2 pre-attack normal period
    # ---------------------------------------------------------

    train = df[
        (
            df["TimeWindow"]
            >= pd.Timestamp(
                "2018-03-01 09:00:00"
            )
        )
        &
        (
            df["TimeWindow"]
            < ATTACK_2_START
        )
    ].copy()

    # ---------------------------------------------------------
    # TEST
    # Attack 1 pre-attack normal period
    # ---------------------------------------------------------

    test = df[
        (
            df["TimeWindow"]
            >= pd.Timestamp(
                "2018-03-01 01:10:00"
            )
        )
        &
        (
            df["TimeWindow"]
            < ATTACK_1_START
        )
    ].copy()

    train = train.dropna(
        subset=[target]
    )

    test = test.dropna(
        subset=[target]
    )

    X_train = train[FEATURES]

    y_train = train[target].astype(int)

    X_test = test[FEATURES]

    y_test = test[target].astype(int)

    print(
        f"Training rows: {len(train)}"
    )

    print(
        "Training labels:"
    )

    print(
        y_train.value_counts()
        .sort_index()
    )

    print(
        f"Test rows: {len(test)}"
    )

    print(
        "Test labels:"
    )

    print(
        y_test.value_counts()
        .sort_index()
    )

    # ---------------------------------------------------------
    # SCALE
    # ---------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    # ---------------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------------

    probabilities = (
        model.predict_proba(
            X_test_scaled
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------

    print()

    print(
        "Confusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print()

    print(
        "Classification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    if len(
        np.unique(y_test)
    ) > 1:

        auc = roc_auc_score(
            y_test,
            probabilities,
        )

        print(
            f"ROC-AUC: {auc:.4f}"
        )

    else:

        auc = None

        print(
            "ROC-AUC: unavailable "
            "(test set contains one class)"
        )

    # ---------------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------------

    importance = pd.DataFrame({

        "Feature": FEATURES,

        "Coefficient": (
            model.coef_[0]
        ),

    })

    importance[
        "AbsCoefficient"
    ] = (
        importance[
            "Coefficient"
        ].abs()
    )

    importance = (
        importance.sort_values(
            "AbsCoefficient",
            ascending=False,
        )
    )

    print()

    print(
        "Top feature coefficients:"
    )

    print(
        importance[
            [
                "Feature",
                "Coefficient",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    model_path = (
        f"{MODEL_DIR}/"
        f"forecast_model_{horizon}.pkl"
    )

    scaler_path = (
        f"{MODEL_DIR}/"
        f"forecast_scaler_{horizon}.pkl"
    )

    with open(
        model_path,
        "wb",
    ) as f:

        pickle.dump(
            model,
            f,
        )

    with open(
        scaler_path,
        "wb",
    ) as f:

        pickle.dump(
            scaler,
            f,
        )

    print()

    print(
        f"Model saved: {model_path}"
    )

    print(
        f"Scaler saved: {scaler_path}"
    )

    return {
        "horizon": horizon,
        "target": target,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "auc": auc,
    }


def main():

    print("=" * 70)
    print(
        "TRAIN HORIZON-SPECIFIC FORECAST MODELS"
    )
    print("=" * 70)

    df = load_data()

    print(
        f"Loaded rows: {len(df)}"
    )

    results = []

    for horizon, target in TARGETS.items():

        result = train_one_horizon(
            df,
            horizon,
            target,
        )

        results.append(
            result
        )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    for result in results:

        print(
            f"{result['horizon']}: "
            f"{result['target']} -> "
            f"{result['model_path']}"
        )


if __name__ == "__main__":

    main()