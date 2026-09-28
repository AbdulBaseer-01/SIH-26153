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


# --------------------------------------------------------------
# PRE-ATTACK WINDOWS
# --------------------------------------------------------------

CAMPAIGNS = {
    "ATTACK_1": {
        "start": "2018-03-01 01:10:00",
        "end": "2018-03-01 01:59:00",
    },
    "ATTACK_2": {
        "start": "2018-03-01 09:00:00",
        "end": "2018-03-01 09:56:00",
    },
}


def get_campaign_data(df, campaign):
    start = pd.Timestamp(
        CAMPAIGNS[campaign]["start"]
    )

    end = pd.Timestamp(
        CAMPAIGNS[campaign]["end"]
    )

    return df[
        (df["TimeWindow"] >= start)
        & (df["TimeWindow"] <= end)
    ].copy()


def build_model(name):
    if name == "LOGISTIC":

        return Pipeline(
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
        )

    if name == "RANDOM_FOREST":

        return RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    if name == "GRADIENT_BOOSTING":

        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        )

    raise ValueError(
        f"Unknown model: {name}"
    )


def evaluate_model(
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
):
    model = build_model(model_name)

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    print()
    print("-" * 60)
    print(model_name)
    print("-" * 60)

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    if hasattr(
        model,
        "predict_proba",
    ):

        probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        if len(
            np.unique(y_test)
        ) == 2:

            auc = roc_auc_score(
                y_test,
                probabilities,
            )

            print(
                f"ROC-AUC: {auc:.4f}"
            )

    return model


def main():

    print("=" * 70)
    print("CAMPAIGN-HOLDOUT HORIZON EVALUATION")
    print("=" * 70)

    df = pd.read_csv(INPUT)

    df["TimeWindow"] = pd.to_datetime(
        df["TimeWindow"]
    )

    # Only normal/pre-infiltration states.
    df = df[
        df["IsInfiltrationState"] == 0
    ].copy()

    df = df.sort_values(
        "TimeWindow"
    ).reset_index(
        drop=True
    )

    print(
        f"\nTotal normal rows: {len(df)}"
    )

    # ----------------------------------------------------------
    # RUN BOTH DIRECTIONS
    # ----------------------------------------------------------

    experiments = [
        (
            "ATTACK_1",
            "ATTACK_2",
        ),
        (
            "ATTACK_2",
            "ATTACK_1",
        ),
    ]

    for train_campaign, test_campaign in experiments:

        print()
        print("=" * 70)
        print(
            f"TRAIN: {train_campaign}  "
            f"->  TEST: {test_campaign}"
        )
        print("=" * 70)

        train_df = get_campaign_data(
            df,
            train_campaign,
        )

        test_df = get_campaign_data(
            df,
            test_campaign,
        )

        print(
            f"\nTraining rows: {len(train_df)}"
        )

        print(
            f"Testing rows: {len(test_df)}"
        )

        for horizon, target in TARGETS.items():

            print()
            print("#" * 70)
            print(
                f"# {horizon.upper()}"
            )
            print("#" * 70)

            # Remove undefined target rows.
            train_df_h = train_df[
                train_df[target].notna()
            ].copy()

            test_df_h = test_df[
                test_df[target].notna()
            ].copy()

            X_train = (
                train_df_h[FEATURES]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .fillna(0)
            )

            X_test = (
                test_df_h[FEATURES]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .fillna(0)
            )

            y_train = (
                train_df_h[target]
                .astype(int)
            )

            y_test = (
                test_df_h[target]
                .astype(int)
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

            # --------------------------------------------------
            # SAFETY CHECK
            # --------------------------------------------------

            if len(
                np.unique(y_train)
            ) < 2:

                print(
                    "\nSKIPPED: training set "
                    "does not contain both classes."
                )

                continue

            if len(
                np.unique(y_test)
            ) < 2:

                print(
                    "\nSKIPPED: test set "
                    "does not contain both classes."
                )

                continue

            # --------------------------------------------------
            # MODELS
            # --------------------------------------------------

            for model_name in [
                "LOGISTIC",
                "RANDOM_FOREST",
                "GRADIENT_BOOSTING",
            ]:

                evaluate_model(
                    model_name,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                )


if __name__ == "__main__":
    main()