import pandas as pd

from app.config import FORECAST_RESULTS_PATH


class ForecastEngine:

    def __init__(self):
        print("Loading forecast results...")

        self.results = pd.read_csv(
            FORECAST_RESULTS_PATH,
            parse_dates=["TimeWindow"],
        )

        if self.results.empty:
            raise RuntimeError(
                "Forecast results file is empty."
            )

        print(
            f"Forecast results loaded: "
            f"{len(self.results)} states"
        )

    # =============================================================
    # ALL RESULTS
    # =============================================================

    def get_all(self):
        """
        Return the complete forecast timeline.
        """

        return self.results.copy()

    # =============================================================
    # SINGLE TIMESTAMP
    # =============================================================

    def get_timestamp(
        self,
        timestamp,
    ):
        """
        Return forecast information for one timestamp.
        """

        timestamp = pd.Timestamp(
            timestamp
        )

        matches = self.results[
            self.results["TimeWindow"] == timestamp
        ]

        if matches.empty:
            raise ValueError(
                f"No forecast result found for {timestamp}"
            )

        return matches.iloc[0].to_dict()

    # =============================================================
    # RECENT TIMELINE
    # =============================================================

    def get_recent(
        self,
        timestamp,
        minutes=15,
    ):
        """
        Return the previous N minutes of forecast data.
        """

        timestamp = pd.Timestamp(
            timestamp
        )

        start_time = (
            timestamp
            - pd.Timedelta(minutes=minutes)
        )

        return self.results[
            (
                self.results["TimeWindow"]
                >= start_time
            )
            &
            (
                self.results["TimeWindow"]
                <= timestamp
            )
        ].copy()

    # =============================================================
    # ATTACK WINDOW
    # =============================================================

    def get_attack_window(
        self,
        timestamp,
        before_minutes=30,
        after_minutes=10,
    ):
        """
        Return a timeline around a selected timestamp.
        """

        timestamp = pd.Timestamp(
            timestamp
        )

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

        return self.results[
            (
                self.results["TimeWindow"]
                >= start_time
            )
            &
            (
                self.results["TimeWindow"]
                <= end_time
            )
        ].copy()

    # =============================================================
    # CURRENT FORECAST
    # =============================================================

    def get_forecast(
        self,
        timestamp,
    ):
        """
        Return the important forecasting information
        for a selected timestamp.
        """

        row = self.get_timestamp(
            timestamp
        )

        return {
            "timestamp": row["TimeWindow"],

            "risk_5m": float(
                row["Risk_5m"]
            ),

            "risk_10m": float(
                row["Risk_10m"]
            ),

            "risk_15m": float(
                row["Risk_15m"]
            ),

            "forecast_5m": row[
                "Forecast_5m"
            ],

            "forecast_10m": row[
                "Forecast_10m"
            ],

            "forecast_15m": row[
                "Forecast_15m"
            ],

            "forecast_score": float(
                row["Forecast_Score"]
            ),

            "forecast_state": row[
                "Forecast_State"
            ],

            "reason": row[
                "Forecast_Reason"
            ],

            "ground_truth": row[
                "Ground_Truth"
            ],

            "is_infiltration": int(
                row[
                    "IsInfiltrationState"
                ]
            ),
        }

    # =============================================================
    # LATEST STATE
    # =============================================================

    def get_latest(self):
        """
        Return the latest available forecast.
        """

        row = self.results.iloc[-1]

        return row.to_dict()

    # =============================================================
    # HIGH-RISK EVENTS
    # =============================================================

    def get_high_risk_events(
        self,
        state="HIGH",
    ):
        """
        Return all timestamps matching a forecast state.

        Example:
            get_high_risk_events("HIGH")
        """

        return self.results[
            self.results["Forecast_State"]
            == state
        ].copy()

    # =============================================================
    # FORECAST COUNTS
    # =============================================================

    def get_state_counts(self):
        """
        Count LOW / MEDIUM / HIGH forecast states.
        """

        return (
            self.results[
                "Forecast_State"
            ]
            .value_counts()
            .to_dict()
        )

    # =============================================================
    # SUMMARY
    # =============================================================

    def get_summary(self):
        """
        Generate high-level forecast statistics.
        """

        highest_score_index = (
            self.results[
                "Forecast_Score"
            ].idxmax()
        )

        highest_score_row = (
            self.results.loc[
                highest_score_index
            ]
        )

        return {
            "states": len(
                self.results
            ),

            "low": int(
                (
                    self.results[
                        "Forecast_State"
                    ]
                    == "LOW"
                ).sum()
            ),

            "medium": int(
                (
                    self.results[
                        "Forecast_State"
                    ]
                    == "MEDIUM"
                ).sum()
            ),

            "high": int(
                (
                    self.results[
                        "Forecast_State"
                    ]
                    == "HIGH"
                ).sum()
            ),

            "max_forecast_score": float(
                highest_score_row[
                    "Forecast_Score"
                ]
            ),

            "max_score_time": (
                highest_score_row[
                    "TimeWindow"
                ]
            ),
        }


# =============================================================
# COMMAND-LINE TEST
# =============================================================

if __name__ == "__main__":

    engine = ForecastEngine()

    print()

    summary = engine.get_summary()

    print(
        "========== FORECAST SUMMARY =========="
    )

    for key, value in summary.items():
        print(
            f"{key}: {value}"
        )

    print(
        "======================================"
    )

    print()

    print(
        "========== HIGH-RISK EVENTS =========="
    )

    high_risk = (
        engine.get_high_risk_events(
            "HIGH"
        )
    )

    print(
        high_risk[
            [
                "TimeWindow",
                "Forecast_5m",
                "Forecast_10m",
                "Forecast_15m",
                "Forecast_Score",
                "Forecast_Reason",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print(
        "======================================"
    )