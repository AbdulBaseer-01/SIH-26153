import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
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


def evaluate(
    name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    if (
        probabilities is not None
        and len(np.unique(y_test)) == 2
    ):
        auc = roc_auc_score(
            y_test,
            probabilities,
        )

        print(f"ROC-AUC: {auc:.4f}")

    return model


def main():

    print("=" * 70)
    print("HORIZON FORECAST V2")
    print("=" * 70)

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------------

    df = pd.read_csv(INPUT)

    df["TimeWindow"] = pd.to_datetime(
        df["TimeWindow"]
    )

    # --------------------------------------------------------------
    # ONLY USE STATES BEFORE ACTIVE INFILTRATION
    # --------------------------------------------------------------

    df = df[
        df["IsInfiltrationState"] == 0
    ].copy()

    df = df.sort_values(
        "TimeWindow"
    ).reset_index(
        drop=True
    )

    print(
        f"Rows available: {len(df)}"
    )

    print(
        f"Features: {len(FEATURES)}"
    )

    # --------------------------------------------------------------
    # CLEAN FEATURES
    # --------------------------------------------------------------

    X = (
        df[FEATURES]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    # --------------------------------------------------------------
    # TRAIN ONE MODEL FOR EACH HORIZON
    # --------------------------------------------------------------

    for horizon, target in TARGETS.items():

        print()
        print("#" * 70)
        print(
            f"# {horizon.upper()} ATTACK-START FORECAST"
        )
        print("#" * 70)

        # ----------------------------------------------------------
        # REMOVE ROWS WHERE FUTURE TARGET IS UNKNOWN
        # ----------------------------------------------------------

        valid = df[target].notna()

        X_h = X.loc[valid].copy()

        y = (
            df.loc[valid, target]
            .astype(int)
        )

        df_h = df.loc[valid].copy()

        print(
            f"\nValid rows for {horizon}: "
            f"{len(df_h)}"
        )

        print("\nOverall labels:")
        print(
            y.value_counts()
            .sort_index()
        )

        # ----------------------------------------------------------
        # TEMPORAL 70/30 SPLIT
        # ----------------------------------------------------------

        split = int(
            len(df_h) * 0.70
        )

        X_train = X_h.iloc[:split]
        X_test = X_h.iloc[split:]

        y_train = y.iloc[:split]
        y_test = y.iloc[split:]

        print("\nTemporal split:")

        print(
            "Train:",
            df_h["TimeWindow"].iloc[0],
            "->",
            df_h["TimeWindow"].iloc[
                split - 1
            ],
        )

        print(
            "Test :",
            df_h["TimeWindow"].iloc[
                split
            ],
            "->",
            df_h["TimeWindow"].iloc[-1],
        )

        print("\nTrain labels:")
        print(
            y_train.value_counts()
            .sort_index()
        )

        print("\nTest labels:")
        print(
            y_test.value_counts()
            .sort_index()
        )

        # ----------------------------------------------------------
        # MODELS
        # ----------------------------------------------------------

        models = {

            "LOGISTIC REGRESSION":
                Pipeline(
                    [
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                        (
                            "model",
                            LogisticRegression(
                                class_weight="balanced",
                                max_iter=2000,
                                random_state=42,
                            ),
                        ),
                    ]
                ),

            "RANDOM FOREST":
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=5,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),

            "HISTOGRAM GRADIENT BOOSTING":
                HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=42,
                ),
        }

        # ----------------------------------------------------------
        # TRAIN + EVALUATE
        # ----------------------------------------------------------

        for model_name, model in models.items():

            trained_model = evaluate(
                model_name,
                model,
                X_train,
                y_train,
                X_test,
                y_test,
            )

            safe_name = (
                model_name
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )

            output_path = (
                f"{MODEL_DIR}/"
                f"forecast_{horizon}_"
                f"{safe_name}.pkl"
            )

            joblib.dump(
                trained_model,
                output_path,
            )

            print(
                f"\nSaved: {output_path}"
            )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()