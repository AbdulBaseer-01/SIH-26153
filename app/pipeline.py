from __future__ import annotations

import pandas as pd

from app.attack_stage import AttackStageMapper
from app.forecast import ForecastEngine
from app.inference import WorldModelInference


class NetworkPipeline:

    def __init__(self):

        print("Loading network pipeline...")

        self.world_model = WorldModelInference()
        self.forecast = ForecastEngine()
        self.attack_stage = AttackStageMapper()

        self.network_states = self.world_model.network_states

        print("Network pipeline ready.")

    # ==========================================================
    # SNAPSHOT
    # ==========================================================

    def get_snapshot(self, timestamp):

        timestamp = pd.Timestamp(timestamp)

        # ------------------------------------------------------
        # World-model analysis
        # ------------------------------------------------------

        world_result = self.world_model.analyze_timestamp(
            timestamp
        )

        if world_result is None:
            raise ValueError(
                f"Unable to analyze timestamp: {timestamp}"
            )

        # ------------------------------------------------------
        # Forecast engine
        # ------------------------------------------------------

        forecast_result = self.forecast.get_forecast(
            timestamp
        )

        if forecast_result is None:
            raise ValueError(
                f"No forecast available for timestamp: {timestamp}"
            )

        # ------------------------------------------------------
        # Locate network state
        # ------------------------------------------------------

        timestamp_index = self.world_model.get_timestamp_index(
            timestamp
        )

        if timestamp_index is None:
            raise ValueError(
                f"Timestamp not found in network states: {timestamp}"
            )

        state_row = self.network_states.iloc[
            timestamp_index
        ]

        # ------------------------------------------------------
        # Feature-level explanation
        # ------------------------------------------------------

        # Existing WorldModelInference method does not accept
        # top_n, so retrieve the complete result and slice it.
        top_features = (
            self.world_model.get_top_feature_errors(
                timestamp
            )
        )

        top_features = top_features[:10]

        # ------------------------------------------------------
        # Attack-stage interpretation
        # ------------------------------------------------------

        infiltration_ratio = state_row.get(
            "Infiltration Ratio",
            0.0,
        )

        attack_stage = self.attack_stage.classify(
            prediction_mse=world_result.get(
                "mse",
                0.0,
            ),
            forecast_score=forecast_result.get(
                "forecast_score",
                0.0,
            ),
            infiltration_ratio=infiltration_ratio,
            top_features=top_features,
            forecast_state=forecast_result.get(
                "forecast_state",
                "LOW",
            ),
        )

        # ------------------------------------------------------
        # Final snapshot
        # ------------------------------------------------------

        return {

            # ==================================================
            # TIME
            # ==================================================

            "timestamp": timestamp,

            # ==================================================
            # WORLD MODEL
            # ==================================================

            "prediction_mse": float(
                world_result.get(
                    "mse",
                    0.0,
                )
            ),

            "prediction_mae": float(
                world_result.get(
                    "mae",
                    0.0,
                )
            ),

            # ==================================================
            # FORECAST
            # ==================================================

            "risk_5m": forecast_result.get(
                "risk_5m"
            ),

            "risk_10m": forecast_result.get(
                "risk_10m"
            ),

            "risk_15m": forecast_result.get(
                "risk_15m"
            ),

            "forecast_5m": forecast_result.get(
                "forecast_5m"
            ),

            "forecast_10m": forecast_result.get(
                "forecast_10m"
            ),

            "forecast_15m": forecast_result.get(
                "forecast_15m"
            ),

            "forecast_score": forecast_result.get(
                "forecast_score"
            ),

            "forecast_state": forecast_result.get(
                "forecast_state"
            ),

            "forecast_reason": forecast_result.get(
                "reason"
            ),

            # ==================================================
            # GROUND TRUTH
            # ==================================================

            "infiltration_ratio": float(
                infiltration_ratio
            ),

            "is_infiltration": int(
                state_row.get(
                    "IsInfiltrationState",
                    0,
                )
            ),

            "ground_truth": forecast_result.get(
                "ground_truth"
            ),

            # ==================================================
            # EXPLAINABILITY
            # ==================================================

            "top_features": top_features,

            # ==================================================
            # ATT&CK-ORIENTED STAGE
            # ==================================================

            "attack_stage": attack_stage,
        }

    # ==========================================================
    # FULL TIMELINE
    # ==========================================================

    def get_timeline(self):

        return self.world_model.analyze_all()

    # ==========================================================
    # RECENT TIMELINE
    # ==========================================================

    def get_recent_timeline(
        self,
        timestamp,
        minutes=15,
    ):

        timestamp = pd.Timestamp(timestamp)

        start_time = (
            timestamp
            - pd.Timedelta(
                minutes=minutes
            )
        )

        timeline = self.get_timeline()

        if timeline.empty:
            return timeline

        timeline["TimeWindow"] = pd.to_datetime(
            timeline["TimeWindow"]
        )

        return timeline[
            (timeline["TimeWindow"] >= start_time)
            &
            (timeline["TimeWindow"] <= timestamp)
        ].copy()

    # ==========================================================
    # ATTACK WINDOW
    # ==========================================================

    def get_attack_window(
        self,
        timestamp,
        before_minutes=30,
        after_minutes=10,
    ):

        timestamp = pd.Timestamp(timestamp)

        start_time = (
            timestamp
            - pd.Timedelta(
                minutes=before_minutes
            )
        )

        end_time = (
            timestamp
            + pd.Timedelta(
                minutes=after_minutes
            )
        )

        timeline = self.get_timeline()

        if timeline.empty:
            return timeline

        timeline["TimeWindow"] = pd.to_datetime(
            timeline["TimeWindow"]
        )

        return timeline[
            (timeline["TimeWindow"] >= start_time)
            &
            (timeline["TimeWindow"] <= end_time)
        ].copy()

    # ==========================================================
    # HIGH-RISK EVENTS
    # ==========================================================

    def get_high_risk_events(
        self,
        state="HIGH",
    ):

        return self.forecast.get_high_risk_events(
            state=state
        )

    # ==========================================================
    # SUMMARY
    # ==========================================================

    def get_summary(self):

        world_summary = self.world_model.analyze_all()

        if world_summary.empty:
            return {}

        forecast_summary = (
            self.forecast.get_summary()
        )

        highest_anomaly_index = (
            world_summary[
                "Prediction_MSE"
            ].idxmax()
        )

        highest_anomaly = (
            world_summary.loc[
                highest_anomaly_index
            ]
        )

        return {

            "states_analyzed": len(
                world_summary
            ),

            "mean_mse": float(
                world_summary[
                    "Prediction_MSE"
                ].mean()
            ),

            "max_mse": float(
                world_summary[
                    "Prediction_MSE"
                ].max()
            ),

            "mean_mae": float(
                world_summary[
                    "Prediction_MAE"
                ].mean()
            ),

            "highest_anomaly_time":
                highest_anomaly[
                    "TimeWindow"
                ],

            "highest_anomaly_mse":
                float(
                    highest_anomaly[
                        "Prediction_MSE"
                    ]
                ),

            "forecast_summary":
                forecast_summary,
        }


# ==============================================================
# CLI TEST
# ==============================================================

if __name__ == "__main__":

    pipeline = NetworkPipeline()

    test_timestamp = pd.Timestamp(
        "2018-03-01 09:51:00"
    )

    print()
    print(
        "Testing pipeline at:",
        test_timestamp,
    )

    snapshot = pipeline.get_snapshot(
        test_timestamp
    )

    print()
    print("========== PIPELINE SNAPSHOT ==========")

    # ----------------------------------------------------------
    # World model
    # ----------------------------------------------------------

    print(
        "Timestamp:",
        snapshot["timestamp"]
    )

    print(
        "Prediction MSE:",
        snapshot["prediction_mse"]
    )

    print(
        "Prediction MAE:",
        snapshot["prediction_mae"]
    )

    # ----------------------------------------------------------
    # Forecast
    # ----------------------------------------------------------

    print(
        "5m Forecast:",
        snapshot["forecast_5m"]
    )

    print(
        "10m Forecast:",
        snapshot["forecast_10m"]
    )

    print(
        "15m Forecast:",
        snapshot["forecast_15m"]
    )

    print(
        "Forecast Score:",
        snapshot["forecast_score"]
    )

    print(
        "Forecast State:",
        snapshot["forecast_state"]
    )

    print(
        "Forecast Reason:",
        snapshot["forecast_reason"]
    )

    # ----------------------------------------------------------
    # Ground truth
    # ----------------------------------------------------------

    print(
        "Infiltration Ratio:",
        snapshot["infiltration_ratio"]
    )

    print(
        "Ground Truth:",
        snapshot["ground_truth"]
    )

    # ----------------------------------------------------------
    # Attack stage
    # ----------------------------------------------------------

    print()
    print("Attack Stage:")

    print(
        snapshot[
            "attack_stage"
        ]["stage"]
    )

    print(
        "Confidence:",
        snapshot[
            "attack_stage"
        ]["confidence"]
    )

    print()
    print("Stage Evidence:")

    for evidence in snapshot[
        "attack_stage"
    ]["evidence"]:

        print(
            "-",
            evidence
        )

    print()
    print("Stage Explanation:")

    print(
        snapshot[
            "attack_stage"
        ]["explanation"]
    )

    print()
    print("MITRE Context:")

    print(
        snapshot[
            "attack_stage"
        ]["mitre_context"]
    )

    # ----------------------------------------------------------
    # Feature errors
    # ----------------------------------------------------------

    print()
    print("Top Feature Errors:")

    for feature, error in snapshot[
        "top_features"
    ]:

        print(
            f"{feature}: {error:.4f}"
        )

    print(
        "======================================="
    )