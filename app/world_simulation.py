import numpy as np
import pandas as pd

from app.inference import WorldModelInference


class WorldModelSimulator:
    """
    Recursive K-step simulation using the trained LSTM world model.

    The model predicts:

        10 historical states -> next network state

    During simulation, each predicted state is fed back into
    the sequence to generate the following state.
    """

    def __init__(self):
        self.engine = WorldModelInference()

        self.model = self.engine.model
        self.scaler = self.engine.scaler
        self.network_states = self.engine.network_states
        self.state_columns = self.engine.get_state_columns()

    # ---------------------------------------------------------
    # TIMESTAMP
    # ---------------------------------------------------------

    def get_timestamp_index(self, timestamp):
        timestamp = pd.Timestamp(timestamp)

        matches = self.network_states.index[
            self.network_states["TimeWindow"] == timestamp
        ]

        if len(matches) == 0:
            raise ValueError(
                f"Timestamp {timestamp} does not exist."
            )

        return int(matches[0])

    # ---------------------------------------------------------
    # INITIAL HISTORY
    # ---------------------------------------------------------

    def get_initial_sequence(self, timestamp):
        """
        Return the last 10 continuous states before/current
        at the selected timestamp.
        """

        index = self.get_timestamp_index(timestamp)
        start_index = index - 9

        if start_index < 0:
            raise ValueError(
                "Not enough historical states for simulation."
            )

        if not self.engine.is_continuous(start_index, index):
            raise ValueError(
                "Selected timestamp does not have "
                "10 continuous historical states."
            )

        states = self.network_states.loc[
            start_index:index,
            self.state_columns,
        ].copy()

        if len(states) != 10:
            raise ValueError(
                f"Expected 10 states, got {len(states)}."
            )

        values = states.astype(float)

        # Keep feature names so sklearn does not warn.
        scaled = self.scaler.transform(values)

        return scaled.astype(np.float32)

    # ---------------------------------------------------------
    # SINGLE SIMULATION
    # ---------------------------------------------------------

    def simulate(self, timestamp, steps=5):

        if steps < 1:
            raise ValueError(
                "steps must be >= 1"
            )

        if steps > 30:
            raise ValueError(
                "Maximum simulation horizon is 30 minutes."
            )

        timestamp = pd.Timestamp(timestamp)

        sequence = self.get_initial_sequence(timestamp)

        current_time = timestamp

        results = []

        previous_predicted = None

        for step in range(1, steps + 1):

            # -------------------------------------------------
            # MODEL PREDICTION
            # -------------------------------------------------

            model_input = sequence.reshape(
                1,
                sequence.shape[0],
                sequence.shape[1],
            )

            predicted_scaled = self.model.predict(
                model_input,
                verbose=0,
            )[0]

            predicted_scaled = predicted_scaled.astype(
                np.float32
            )

            # -------------------------------------------------
            # BACK TO ORIGINAL FEATURE SPACE
            # -------------------------------------------------

            predicted_raw = self.scaler.inverse_transform(
                pd.DataFrame(
                    predicted_scaled.reshape(1, -1),
                    columns=self.state_columns,
                )
            )[0]

            # -------------------------------------------------
            # FUTURE STATE DEVIATION
            # -------------------------------------------------

            future_state_deviation = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            predicted_scaled
                        )
                    )
                )
            )

            mean_abs_deviation = float(
                np.mean(
                    np.abs(
                        predicted_scaled
                    )
                )
            )

            max_deviation = float(
                np.max(
                    np.abs(
                        predicted_scaled
                    )
                )
            )

            # -------------------------------------------------
            # TRANSITION SHOCK
            # -------------------------------------------------

            previous_state = sequence[-1]

            transition_shock = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            predicted_scaled
                            - previous_state
                        )
                    )
                )
            )

            # -------------------------------------------------
            # CHANGE BETWEEN PREDICTED STATES
            # -------------------------------------------------

            if previous_predicted is None:

                predicted_state_change = transition_shock

            else:

                predicted_state_change = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                predicted_scaled
                                - previous_predicted
                            )
                        )
                    )
                )

            # -------------------------------------------------
            # TIME
            # -------------------------------------------------

            current_time = (
                current_time
                + pd.Timedelta(minutes=1)
            )

            # -------------------------------------------------
            # STORE
            # -------------------------------------------------

            results.append(
                {
                    "Current_Time": timestamp,
                    "Future_Time": current_time,
                    "Step": step,
                    "Transition_Shock": transition_shock,
                    "Future_State_Deviation": future_state_deviation,
                    "Mean_Abs_Deviation": mean_abs_deviation,
                    "Max_Deviation": max_deviation,
                    "Predicted_State_Change":
                        predicted_state_change,
                }
            )

            # -------------------------------------------------
            # RECURSIVE FEEDBACK
            # -------------------------------------------------

            sequence = np.vstack(
                [
                    sequence[1:],
                    predicted_scaled,
                ]
            )

            previous_predicted = predicted_scaled

        results = pd.DataFrame(results)

        # -----------------------------------------------------
        # FUTURE STABILITY
        # -----------------------------------------------------

        if len(results) > 1:

            later_changes = results.loc[
                results["Step"] > 1,
                "Predicted_State_Change",
            ]

            future_stability = float(
                later_changes.mean()
            )

        else:

            future_stability = np.nan

        results["Future_Stability"] = (
            future_stability
        )

        # -----------------------------------------------------
        # SIMULATION RISK
        # -----------------------------------------------------
        #
        # This is NOT a probability.
        #
        # It combines the initial transition shock with
        # future-state deviation.
        #
        # The purpose is to create a relative simulation
        # signal for the dashboard.
        # -----------------------------------------------------

        results["Simulation_Risk"] = (
            results["Transition_Shock"]
            * 0.70
            +
            results["Future_State_Deviation"]
            * 0.30
        )

        return results

    # ---------------------------------------------------------
    # SIMULATE WITH FEATURE VALUES
    # ---------------------------------------------------------

    def simulate_states(self, timestamp, steps=5):

        timestamp = pd.Timestamp(timestamp)

        sequence = self.get_initial_sequence(timestamp)

        current_time = timestamp

        results = []

        for step in range(1, steps + 1):

            model_input = sequence.reshape(
                1,
                sequence.shape[0],
                sequence.shape[1],
            )

            predicted_scaled = self.model.predict(
                model_input,
                verbose=0,
            )[0].astype(np.float32)

            predicted_raw = self.scaler.inverse_transform(
                pd.DataFrame(
                    predicted_scaled.reshape(1, -1),
                    columns=self.state_columns,
                )
            )[0]

            future_state_deviation = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            predicted_scaled
                        )
                    )
                )
            )

            transition_shock = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            predicted_scaled
                            - sequence[-1]
                        )
                    )
                )
            )

            current_time = (
                current_time
                + pd.Timedelta(minutes=1)
            )

            row = {
                "Current_Time": timestamp,
                "Future_Time": current_time,
                "Step": step,
                "Transition_Shock":
                    transition_shock,
                "Future_State_Deviation":
                    future_state_deviation,
            }

            for feature, value in zip(
                self.state_columns,
                predicted_raw,
            ):
                row[feature] = float(value)

            results.append(row)

            sequence = np.vstack(
                [
                    sequence[1:],
                    predicted_scaled,
                ]
            )

        return pd.DataFrame(results)


# ---------------------------------------------------------
# CLI TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    simulator = WorldModelSimulator()

    timestamp = pd.Timestamp(
        "2018-03-01 09:51:00"
    )

    print()
    print("=" * 70)
    print("WORLD MODEL SIMULATION TEST")
    print("=" * 70)

    results = simulator.simulate(
        timestamp,
        steps=5,
    )

    print()
    print(
        results[
            [
                "Step",
                "Future_Time",
                "Transition_Shock",
                "Future_State_Deviation",
                "Predicted_State_Change",
                "Future_Stability",
                "Simulation_Risk",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 70)
    print("SIMULATION TEST PASSED")
    print("=" * 70)