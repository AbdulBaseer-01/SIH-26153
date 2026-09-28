import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from app.config import (
    WORLD_MODEL_PATH,
    WORLD_SCALER_PATH,
    NETWORK_STATES_PATH,
)


class WorldModelInference:

    def __init__(self):

        print("Loading world model...")

        # ---------------------------------------------------------
        # Load trained world model
        # ---------------------------------------------------------

        self.model = load_model(
            WORLD_MODEL_PATH
        )

        # ---------------------------------------------------------
        # Load training scaler
        # ---------------------------------------------------------

        self.scaler = joblib.load(
            WORLD_SCALER_PATH
        )

        # ---------------------------------------------------------
        # Load processed network states
        # ---------------------------------------------------------

        self.network_states = pd.read_csv(
            NETWORK_STATES_PATH,
            parse_dates=["TimeWindow"],
        )

        print("World model loaded.")

    # =============================================================
    # FEATURE INFORMATION
    # =============================================================

    def get_state_columns(self):

        excluded = {
            "TimeWindow",
            "Infiltration Ratio",
            "IsInfiltrationState",
        }

        return [
            column
            for column in self.network_states.columns
            if column not in excluded
        ]

    # =============================================================
    # CHECK CONTINUOUS TIMELINE
    # =============================================================

    def is_continuous(
        self,
        start_index,
        end_index,
    ):
        """
        Check whether every timestamp between start_index
        and end_index is exactly one minute apart.
        """

        if start_index < 0:
            return False

        if end_index >= len(
            self.network_states
        ):
            return False

        timestamps = self.network_states.iloc[
            start_index:end_index + 1
        ]["TimeWindow"]

        timestamps = timestamps.reset_index(
            drop=True
        )

        if len(timestamps) <= 1:
            return True

        differences = timestamps.diff().dropna()

        return bool(
            (differences == pd.Timedelta(minutes=1)).all()
        )

    # =============================================================
    # SEQUENCE PREPARATION
    # =============================================================

    def prepare_sequence(
        self,
        end_index,
        sequence_length=10,
    ):

        state_columns = self.get_state_columns()

        start_index = (
            end_index
            - sequence_length
            + 1
        )

        if start_index < 0:

            raise ValueError(
                "Not enough historical states for "
                f"a {sequence_length}-step sequence."
            )

        # ---------------------------------------------------------
        # Historical continuity check
        # ---------------------------------------------------------

        if not self.is_continuous(
            start_index,
            end_index,
        ):

            raise ValueError(
                "Historical sequence contains "
                "a time gap."
            )

        sequence_df = self.network_states.iloc[
            start_index:end_index + 1
        ]

        # ---------------------------------------------------------
        # Extract features
        # ---------------------------------------------------------

        values = sequence_df[
            state_columns
        ].values.astype(float)

        # ---------------------------------------------------------
        # Clean invalid values
        # ---------------------------------------------------------

        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # ---------------------------------------------------------
        # Keep feature names for scaler
        # ---------------------------------------------------------

        values_df = pd.DataFrame(
            values,
            columns=state_columns,
        )

        # ---------------------------------------------------------
        # Scale
        # ---------------------------------------------------------

        scaled = self.scaler.transform(
            values_df
        )

        # ---------------------------------------------------------
        # Add batch dimension
        # ---------------------------------------------------------

        sequence = np.expand_dims(
            scaled,
            axis=0,
        )

        return sequence

    # =============================================================
    # NEXT STATE PREDICTION
    # =============================================================

    def predict_next_state(
        self,
        end_index,
    ):

        sequence = self.prepare_sequence(
            end_index
        )

        prediction = self.model.predict(
            sequence,
            verbose=0,
        )

        return prediction[0]

    # =============================================================
    # PREDICTION ERROR
    # =============================================================

    def calculate_prediction_error(
        self,
        end_index,
        prediction=None,
    ):

        state_columns = self.get_state_columns()

        target_index = end_index + 1

        if target_index >= len(
            self.network_states
        ):

            raise ValueError(
                "No future state is available "
                "for comparison."
            )

        # ---------------------------------------------------------
        # IMPORTANT:
        # Target must be exactly one minute after
        # the current state.
        # ---------------------------------------------------------

        current_time = self.network_states.iloc[
            end_index
        ]["TimeWindow"]

        target_time = self.network_states.iloc[
            target_index
        ]["TimeWindow"]

        if (
            target_time - current_time
            != pd.Timedelta(minutes=1)
        ):

            raise ValueError(
                "Future target contains a time gap."
            )

        # ---------------------------------------------------------
        # Generate prediction if needed
        # ---------------------------------------------------------

        if prediction is None:

            prediction = self.predict_next_state(
                end_index
            )

        # ---------------------------------------------------------
        # Actual future state
        # ---------------------------------------------------------

        actual = self.network_states.iloc[
            target_index
        ][state_columns].values.astype(float)

        actual = np.nan_to_num(
            actual,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # ---------------------------------------------------------
        # Scale actual state
        # ---------------------------------------------------------

        actual_df = pd.DataFrame(
            [actual],
            columns=state_columns,
        )

        actual_scaled = self.scaler.transform(
            actual_df
        )[0]

        # ---------------------------------------------------------
        # Feature errors
        # ---------------------------------------------------------

        errors = (
            prediction
            - actual_scaled
        )

        squared_error = (
            errors ** 2
        )

        absolute_error = np.abs(
            errors
        )

        mse = float(
            np.mean(
                squared_error
            )
        )

        mae = float(
            np.mean(
                absolute_error
            )
        )

        feature_errors = {
            feature: float(error)
            for feature, error in zip(
                state_columns,
                absolute_error,
            )
        }

        return {
            "mse": mse,
            "mae": mae,
            "feature_errors": feature_errors,
        }

    # =============================================================
    # TIMESTAMP LOOKUP
    # =============================================================

    def get_timestamp_index(
        self,
        timestamp,
    ):

        timestamp = pd.Timestamp(
            timestamp
        )

        matches = self.network_states.index[
            self.network_states[
                "TimeWindow"
            ] == timestamp
        ]

        if len(matches) == 0:

            raise ValueError(
                f"No network state found for {timestamp}"
            )

        return int(
            matches[0]
        )

    # =============================================================
    # SINGLE TIMESTAMP ANALYSIS
    # =============================================================

    def analyze_timestamp(
        self,
        timestamp,
    ):

        timestamp = pd.Timestamp(
            timestamp
        )

        index = self.get_timestamp_index(
            timestamp
        )

        prediction = self.predict_next_state(
            index
        )

        result = {
            "timestamp": timestamp,
            "state_index": index,
            "prediction": prediction,
        }

        # ---------------------------------------------------------
        # Compare against future state
        # ---------------------------------------------------------

        if index + 1 < len(
            self.network_states
        ):

            try:

                errors = (
                    self.calculate_prediction_error(
                        index,
                        prediction,
                    )
                )

                result.update(
                    errors
                )

                result["next_timestamp"] = (
                    self.network_states.iloc[
                        index + 1
                    ]["TimeWindow"]
                )

            except ValueError:

                # No valid continuous future
                # state exists.
                pass

        return result

    # =============================================================
    # BATCH ANALYSIS
    # =============================================================

    def analyze_all(
        self,
        sequence_length=10,
    ):
        """
        Run the world model across every valid
        continuous 10-state historical window.

        A valid prediction requires:

            10 continuous historical states
            +
            1 continuous future target state

        Dataset gaps are completely excluded.
        """

        results = []

        state_count = len(
            self.network_states
        )

        # ---------------------------------------------------------
        # Each end_index represents the final historical
        # state used to predict the next state.
        # ---------------------------------------------------------

        for end_index in range(
            sequence_length - 1,
            state_count - 1,
        ):

            try:

                # -------------------------------------------------
                # Historical sequence:
                #
                # end_index - 9 ... end_index
                # -------------------------------------------------

                sequence_start = (
                    end_index
                    - sequence_length
                    + 1
                )

                if not self.is_continuous(
                    sequence_start,
                    end_index,
                ):

                    continue

                # -------------------------------------------------
                # Future target:
                #
                # end_index + 1
                # -------------------------------------------------

                current_time = (
                    self.network_states.iloc[
                        end_index
                    ]["TimeWindow"]
                )

                next_time = (
                    self.network_states.iloc[
                        end_index + 1
                    ]["TimeWindow"]
                )

                if (
                    next_time - current_time
                    != pd.Timedelta(minutes=1)
                ):

                    continue

                # -------------------------------------------------
                # Predict
                # -------------------------------------------------

                prediction = (
                    self.predict_next_state(
                        end_index
                    )
                )

                # -------------------------------------------------
                # Calculate error
                # -------------------------------------------------

                errors = (
                    self.calculate_prediction_error(
                        end_index,
                        prediction,
                    )
                )

                # -------------------------------------------------
                # Current state
                # -------------------------------------------------

                current_row = (
                    self.network_states.iloc[
                        end_index
                    ]
                )

                results.append(
                    {
                        "TimeWindow": current_row[
                            "TimeWindow"
                        ],

                        "Prediction_MSE": errors[
                            "mse"
                        ],

                        "Prediction_MAE": errors[
                            "mae"
                        ],

                        "IsInfiltrationState": current_row[
                            "IsInfiltrationState"
                        ],

                        "Infiltration Ratio": current_row[
                            "Infiltration Ratio"
                        ],
                    }
                )

            except ValueError:

                # -------------------------------------------------
                # Skip invalid/gap-crossing windows.
                # -------------------------------------------------

                continue

        return pd.DataFrame(
            results
        )

    # =============================================================
    # TOP FEATURE ERRORS
    # =============================================================

    def get_top_feature_errors(
        self,
        timestamp,
        top_k=10,
    ):

        result = self.analyze_timestamp(
            timestamp
        )

        feature_errors = result.get(
            "feature_errors",
            {},
        )

        sorted_features = sorted(
            feature_errors.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return sorted_features[
            :top_k
        ]