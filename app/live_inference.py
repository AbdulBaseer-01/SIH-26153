import numpy as np
import pandas as pd


class LiveWorldModel:

    WINDOW_SIZE = 10

    def __init__(self, inference_engine=None):

        if inference_engine is None:
            from app.inference import WorldModelInference
            inference_engine = WorldModelInference()

        self.engine = inference_engine

        self.state_columns = (
            self.engine.get_state_columns()
        )

        self.model = self.engine.model
        self.scaler = self.engine.scaler

        # Recent completed network states
        self.state_history = []

        # Prediction waiting for its target state
        self.pending_prediction = None

        # Valid evaluated predictions only
        self.events = []

    # ---------------------------------------------------------
    # CHECK WHETHER LAST 10 STATES ARE CONTINUOUS
    # ---------------------------------------------------------

    def _history_is_continuous(self):

        if len(self.state_history) < self.WINDOW_SIZE:
            return False

        recent = self.state_history[
            -self.WINDOW_SIZE:
        ]

        timestamps = [
            pd.Timestamp(
                state["TimeWindow"]
            )
            for state in recent
        ]

        for i in range(1, len(timestamps)):

            if (
                timestamps[i]
                - timestamps[i - 1]
                != pd.Timedelta(minutes=1)
            ):

                return False

        return True

    # ---------------------------------------------------------
    # ADD COMPLETED 1-MINUTE STATE
    # ---------------------------------------------------------

    def add_state(self, state):

        if isinstance(state, pd.Series):

            state = state.to_dict()

        if not isinstance(state, dict):

            raise TypeError(
                "state must be a dict or pandas Series."
            )

        if "TimeWindow" not in state:

            raise ValueError(
                "State must contain TimeWindow."
            )

        timestamp = pd.Timestamp(
            state["TimeWindow"]
        )

        event = None

        # -----------------------------------------------------
        # EVALUATE PREVIOUS PREDICTION
        # -----------------------------------------------------

        if self.pending_prediction is not None:

            event = self._evaluate_prediction(
                state,
                self.pending_prediction,
            )

            # Only store valid evaluations.
            #
            # If the incoming state does not match the
            # prediction timestamp, the stream has a gap
            # or discontinuity and the prediction cannot
            # be evaluated.

            if event.get("Valid", False):

                self.events.append(event)

            else:

                event = None

            self.pending_prediction = None

        # -----------------------------------------------------
        # ADD NEW COMPLETED STATE
        # -----------------------------------------------------

        self.state_history.append(state)

        # Keep bounded history
        if len(self.state_history) > 100:

            self.state_history = (
                self.state_history[-100:]
            )

        # -----------------------------------------------------
        # REQUIRE 10 CONTINUOUS STATES
        # -----------------------------------------------------

        if not self._history_is_continuous():

            return event

        # -----------------------------------------------------
        # EXTRACT LAST 10 STATES
        # -----------------------------------------------------

        sequence_df = pd.DataFrame(
            self.state_history[
                -self.WINDOW_SIZE:
            ]
        )

        values = (
            sequence_df[
                self.state_columns
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
            .values
            .astype(float)
        )

        values_df = pd.DataFrame(
            values,
            columns=self.state_columns,
        )

        # -----------------------------------------------------
        # SCALE USING TRAINING SCALER
        # -----------------------------------------------------

        scaled = self.scaler.transform(
            values_df
        )

        # -----------------------------------------------------
        # ADD BATCH DIMENSION
        # -----------------------------------------------------

        sequence = np.expand_dims(
            scaled,
            axis=0,
        )

        # -----------------------------------------------------
        # PREDICT NEXT NETWORK STATE
        # -----------------------------------------------------

        predicted_state = self.model.predict(
            sequence,
            verbose=0,
        )[0]

        # -----------------------------------------------------
        # STORE PENDING PREDICTION
        # -----------------------------------------------------

        self.pending_prediction = {

            "prediction_time": (
                timestamp
                + pd.Timedelta(minutes=1)
            ),

            "source_time": timestamp,

            "predicted_state": (
                np.asarray(
                    predicted_state,
                    dtype=np.float32,
                )
            ),
        }

        return event

    # ---------------------------------------------------------
    # EVALUATE PREDICTION AGAINST ACTUAL STATE
    # ---------------------------------------------------------

    def _evaluate_prediction(
        self,
        actual_state,
        prediction,
    ):

        actual_timestamp = pd.Timestamp(
            actual_state["TimeWindow"]
        )

        expected_timestamp = pd.Timestamp(
            prediction["prediction_time"]
        )

        # -----------------------------------------------------
        # TARGET TIMESTAMP MUST MATCH
        # -----------------------------------------------------

        if actual_timestamp != expected_timestamp:

            return {

                "TimeWindow": actual_timestamp,

                "PredictionTime": expected_timestamp,

                "SourceTime": prediction[
                    "source_time"
                ],

                "MSE": np.nan,

                "MAE": np.nan,

                "Valid": False,

                "Error": (
                    "Prediction target timestamp "
                    "does not match actual state."
                ),
            }

        # -----------------------------------------------------
        # EXTRACT ACTUAL STATE
        # -----------------------------------------------------

        actual = (
            pd.Series(actual_state)
            .reindex(
                self.state_columns
            )
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
            .values
            .astype(float)
        )

        actual_df = pd.DataFrame(
            [actual],
            columns=self.state_columns,
        )

        # -----------------------------------------------------
        # SCALE ACTUAL STATE
        #
        # Model prediction is in scaled space, so the actual
        # target must also be transformed with the same scaler.
        # -----------------------------------------------------

        actual_scaled = (
            self.scaler.transform(
                actual_df
            )[0]
        )

        predicted = np.asarray(
            prediction["predicted_state"],
            dtype=float,
        )

        # -----------------------------------------------------
        # CALCULATE FEATURE ERRORS
        # -----------------------------------------------------

        errors = (
            predicted
            - actual_scaled
        )

        absolute_errors = np.abs(
            errors
        )

        squared_errors = (
            errors ** 2
        )

        mse = float(
            np.mean(
                squared_errors
            )
        )

        mae = float(
            np.mean(
                absolute_errors
            )
        )

        # -----------------------------------------------------
        # PER-FEATURE ERRORS
        # -----------------------------------------------------

        feature_errors = {

            feature: float(error)

            for feature, error in zip(
                self.state_columns,
                absolute_errors,
            )
        }

        # -----------------------------------------------------
        # TOP DRIVING FEATURE
        # -----------------------------------------------------

        top_feature = None
        top_feature_error = 0.0

        if feature_errors:

            top_feature = max(
                feature_errors,
                key=feature_errors.get,
            )

            top_feature_error = max(
                feature_errors.values()
            )

        # -----------------------------------------------------
        # RESULT
        # -----------------------------------------------------

        return {

            "TimeWindow": actual_timestamp,

            "PredictionTime": expected_timestamp,

            "SourceTime": prediction[
                "source_time"
            ],

            "MSE": mse,

            "MAE": mae,

            "Valid": True,

            "PredictedStateMagnitude": float(
                np.mean(
                    np.abs(predicted)
                )
            ),

            "ActualStateMagnitude": float(
                np.mean(
                    np.abs(actual_scaled)
                )
            ),

            "TopFeature": top_feature,

            "TopFeatureError": (
                top_feature_error
            ),

            "FeatureErrors": feature_errors,
        }

    # ---------------------------------------------------------
    # STATUS
    # ---------------------------------------------------------

    def ready(self):

        return self._history_is_continuous()

    def has_pending_prediction(self):

        return (
            self.pending_prediction
            is not None
        )

    def get_latest_event(self):

        if not self.events:

            return None

        return self.events[-1]

    def get_events(self):

        return pd.DataFrame(
            self.events
        )

    def get_history(self):

        return pd.DataFrame(
            self.state_history
        )

    # ---------------------------------------------------------
    # RESET
    # ---------------------------------------------------------

    def reset(self):

        self.state_history = []

        self.pending_prediction = None

        self.events = []