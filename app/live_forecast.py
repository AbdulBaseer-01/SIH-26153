import numpy as np
import pandas as pd


class LiveForecastEngine:

    """
    Converts live world-model prediction errors into
    short-horizon forward risk signals.

    IMPORTANT:
    - Risk scores are NOT probabilities.
    - This is a rule-based forecasting layer.
    - It operates only on live prediction-error history.
    """

    def __init__(
        self,
        threshold_95=0.7582,
        threshold_99=1.0296,
    ):

        self.threshold_95 = float(
            threshold_95
        )

        self.threshold_99 = float(
            threshold_99
        )

        self.events = []

    # ---------------------------------------------------------
    # ADD WORLD-MODEL EVENT
    # ---------------------------------------------------------

    def add_event(self, event):

        if event is None:
            return None

        if not event.get("Valid", False):
            return None

        timestamp = pd.Timestamp(
            event["TimeWindow"]
        )

        mse = float(
            event["MSE"]
        )

        mae = float(
            event["MAE"]
        )

        record = {
            "TimeWindow": timestamp,
            "MSE": mse,
            "MAE": mae,
            "TopFeature": event.get(
                "TopFeature"
            ),
            "TopFeatureError": float(
                event.get(
                    "TopFeatureError",
                    0.0,
                )
            ),
        }

        self.events.append(record)

        # Keep enough history for rolling calculations
        if len(self.events) > 100:

            self.events = (
                self.events[-100:]
            )

        return self._calculate_current_forecast()

    # ---------------------------------------------------------
    # CALCULATE CURRENT FORECAST
    # ---------------------------------------------------------

    def _calculate_current_forecast(self):

        if not self.events:
            return None

        current = self.events[-1]

        mse = float(
            current["MSE"]
        )

        values = np.array(
            [
                float(event["MSE"])
                for event in self.events
            ],
            dtype=float,
        )

        # -----------------------------------------------------
        # RECENT WINDOWS
        # -----------------------------------------------------

        last_5 = values[-5:]

        last_10 = values[-10:]

        # -----------------------------------------------------
        # THRESHOLD COUNTS
        # -----------------------------------------------------

        above_95_5 = int(
            np.sum(
                last_5
                > self.threshold_95
            )
        )

        above_95_10 = int(
            np.sum(
                last_10
                > self.threshold_95
            )
        )

        # -----------------------------------------------------
        # HIGH-ANOMALY COUNTS
        # -----------------------------------------------------

        above_99_5 = int(
            np.sum(
                last_5
                > self.threshold_99
            )
        )

        above_99_10 = int(
            np.sum(
                last_10
                > self.threshold_99
            )
        )

        # -----------------------------------------------------
        # RATIO TO BASELINE
        #
        # The historical system used a ratio-based signal.
        # For live operation we estimate the baseline from
        # the preceding error history.
        # -----------------------------------------------------

        if len(values) >= 11:

            baseline_values = values[
                :-1
            ][-30:]

        else:

            baseline_values = values[
                :-1
            ]

        if len(baseline_values) == 0:

            baseline = self.threshold_95

        else:

            baseline = float(
                np.median(
                    baseline_values
                )
            )

            baseline = max(
                baseline,
                1e-6,
            )

        ratio_10 = (
            mse / baseline
        )

        # -----------------------------------------------------
        # CURRENT ANOMALY LEVEL
        # -----------------------------------------------------

        if mse > self.threshold_99:

            current_level = "HIGH"

        elif mse > self.threshold_95:

            current_level = "ELEVATED"

        else:

            current_level = "NORMAL"

        # -----------------------------------------------------
        # HORIZON SCORES
        # -----------------------------------------------------

        score_5 = self._calculate_score(
            mse=mse,
            above_95=above_95_5,
            above_99=above_99_5,
            ratio=ratio_10,
        )

        score_10 = self._calculate_score(
            mse=mse,
            above_95=above_95_10,
            above_99=above_99_10,
            ratio=ratio_10,
        )

        score_15 = self._calculate_score(
            mse=mse,
            above_95=above_95_10,
            above_99=above_99_10,
            ratio=ratio_10,
        )

        # -----------------------------------------------------
        # RISK STATES
        # -----------------------------------------------------

        state_5 = self._risk_state(
            score_5
        )

        state_10 = self._risk_state(
            score_10
        )

        state_15 = self._risk_state(
            score_15
        )

        # -----------------------------------------------------
        # REASON
        # -----------------------------------------------------

        reason = self._build_reason(
            mse=mse,
            ratio=ratio_10,
            above_95_5=above_95_5,
            above_95_10=above_95_10,
            above_99_5=above_99_5,
        )

        # -----------------------------------------------------
        # TOP FEATURE
        # -----------------------------------------------------

        top_feature = current[
            "TopFeature"
        ]

        return {

            "TimeWindow": current[
                "TimeWindow"
            ],

            "MSE": mse,

            "MAE": current[
                "MAE"
            ],

            "BaselineMSE": baseline,

            "MSEBaselineRatio": ratio_10,

            "CurrentLevel": current_level,

            "Above95_Count_5": above_95_5,

            "Above95_Count_10": above_95_10,

            "Above99_Count_5": above_99_5,

            "Above99_Count_10": above_99_10,

            "RiskScore_5m": score_5,

            "RiskScore_10m": score_10,

            "RiskScore_15m": score_15,

            "RiskState_5m": state_5,

            "RiskState_10m": state_10,

            "RiskState_15m": state_15,

            "TopFeature": top_feature,

            "TopFeatureError": current[
                "TopFeatureError"
            ],

            "Reason": reason,
        }

    # ---------------------------------------------------------
    # SCORE CALCULATION
    # ---------------------------------------------------------

    def _calculate_score(
        self,
        mse,
        above_95,
        above_99,
        ratio,
    ):

        score = 0.0

        # Current anomaly magnitude
        if mse > self.threshold_99:

            score += 4.0

        elif mse > self.threshold_95:

            score += 2.0

        # Persistence
        if above_95 >= 2:

            score += 2.0

        if above_95 >= 3:

            score += 1.0

        # Strong repeated anomalies
        if above_99 >= 2:

            score += 1.0

        # Relative jump from recent baseline
        if ratio >= 3:

            score += 2.0

        elif ratio >= 2:

            score += 1.0

        return float(
            min(score, 10.0)
        )

    # ---------------------------------------------------------
    # RISK STATE
    # ---------------------------------------------------------

    @staticmethod
    def _risk_state(score):

        if score >= 8:

            return "HIGH"

        if score >= 4:

            return "MEDIUM"

        return "LOW"

    # ---------------------------------------------------------
    # EXPLANATION
    # ---------------------------------------------------------

    def _build_reason(
        self,
        mse,
        ratio,
        above_95_5,
        above_95_10,
        above_99_5,
    ):

        reasons = []

        if mse > self.threshold_99:

            reasons.append(
                "Current prediction error is above the 99th-percentile baseline."
            )

        elif mse > self.threshold_95:

            reasons.append(
                "Current prediction error is above the 95th-percentile baseline."
            )

        if above_95_5 >= 2:

            reasons.append(
                f"{above_95_5} of the last 5 observations exceed the 95th-percentile threshold."
            )

        if above_95_10 >= 3:

            reasons.append(
                f"{above_95_10} of the last 10 observations exceed the 95th-percentile threshold."
            )

        if above_99_5 >= 2:

            reasons.append(
                "Repeated high-magnitude prediction errors detected."
            )

        if ratio >= 3:

            reasons.append(
                f"Current error is approximately {ratio:.1f}× the recent baseline."
            )

        if not reasons:

            reasons.append(
                "No strong persistent anomaly signal detected."
            )

        return " ".join(reasons)

    # ---------------------------------------------------------
    # GET LATEST FORECAST
    # ---------------------------------------------------------

    def get_latest(self):

        if not self.events:

            return None

        return self._calculate_current_forecast()

    # ---------------------------------------------------------
    # GET HISTORY
    # ---------------------------------------------------------

    def get_history(self):

        return pd.DataFrame(
            self.events
        )

    # ---------------------------------------------------------
    # RESET
    # ---------------------------------------------------------

    def reset(self):

        self.events = []