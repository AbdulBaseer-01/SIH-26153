from collections import deque
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


class LiveStateBuilder:
    WINDOW_SIZE = "1min"

    def __init__(self, state_columns, median_path=None):
        self.state_columns = list(state_columns)

        if median_path is None:
            median_path = (
                Path(__file__).resolve().parent.parent
                / "models"
                / "live_feature_medians.pkl"
            )

        median_path = Path(median_path)

        with open(median_path, "rb") as f:
            self.medians = pickle.load(f)

        self.current_window = None
        self.current_flows = []
        self.state_history = deque(maxlen=100)

    def add_flows(self, flows):
        if flows is None:
            return []

        if isinstance(flows, pd.DataFrame):
            df = flows.copy()
        else:
            df = pd.DataFrame(flows)

        if df.empty:
            return []

        if "Timestamp" not in df.columns:
            raise ValueError("Input flows must contain a Timestamp column.")

        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"],
            errors="coerce",
            dayfirst=True
        )

        df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")

        completed_states = []

        for _, row in df.iterrows():
            timestamp = row["Timestamp"]
            window = timestamp.floor(self.WINDOW_SIZE)

            if self.current_window is None:
                self.current_window = window

            elif window < self.current_window:
                # Ignore late data from an already-finalized window.
                continue

            elif window > self.current_window:
                state = self._finalize_current_window()

                if state is not None:
                    completed_states.append(state)

                self.current_window = window

            self.current_flows.append(row.to_dict())

        return completed_states

    def _finalize_current_window(self):
        if self.current_window is None or not self.current_flows:
            return None

        df = pd.DataFrame(self.current_flows)

        state = {
            "TimeWindow": self.current_window
        }

        for column in self.state_columns:
            if column == "Flow Count":
                continue

            if column not in df.columns:
                state[column] = self.medians.get(column, 0.0)
                continue

            values = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            values = values.replace(
                [np.inf, -np.inf],
                np.nan
            )

            # IMPORTANT:
            # Match the training preprocessing exactly.
            median = self.medians.get(column, 0.0)

            values = values.fillna(median)

            state[column] = values.mean()

        state["Flow Count"] = len(df)

        # Keep exact state-column ordering.
        ordered_state = {
            "TimeWindow": self.current_window
        }

        for column in self.state_columns:
            ordered_state[column] = state.get(column, 0.0)

        state = ordered_state

        self.state_history.append(state)

        self.current_flows = []

        return state

    def flush(self):
        state = self._finalize_current_window()

        if state is None:
            return []

        self.current_window = None

        return [state]

    def get_history(self, n=None):
        history = list(self.state_history)

        if n is not None:
            history = history[-n:]

        if not history:
            return pd.DataFrame(
                columns=["TimeWindow"] + self.state_columns
            )

        return pd.DataFrame(history)

    def ready_for_world_model(self):
        return len(self.state_history) >= 10

    def get_model_sequence(self):
        if not self.ready_for_world_model():
            raise ValueError(
                f"Need at least 10 completed states. "
                f"Currently have {len(self.state_history)}."
            )

        history = self.get_history(10)

        return history[self.state_columns].to_numpy(
            dtype=np.float32
        )