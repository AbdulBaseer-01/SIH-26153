import pandas as pd

from app.inference import WorldModelInference


class NetworkAnalyzer:

    def __init__(self):
        self.engine = WorldModelInference()

    # =============================================================
    # RUN WORLD-MODEL ANALYSIS
    # =============================================================

    def run(self):
        """
        Run world-model inference across the complete
        processed network timeline.

        Only continuous sequences are included.

        A valid prediction requires:
            10 continuous historical states
            +
            1 continuous future target state
        """

        print("Running network analysis...")

        results = self.engine.analyze_all()

        if results.empty:
            raise RuntimeError(
                "No valid network states were available "
                "for analysis."
            )

        print(
            f"Analysis complete: {len(results)} states"
        )

        return results

    # =============================================================
    # SUMMARY
    # =============================================================

    def get_summary(self, results):
        """
        Generate high-level statistics from the
        world-model analysis.
        """

        highest_anomaly_index = (
            results["Prediction_MSE"].idxmax()
        )

        highest_anomaly_row = results.loc[
            highest_anomaly_index
        ]

        return {
            "states_analyzed": len(results),

            "mean_mse": float(
                results["Prediction_MSE"].mean()
            ),

            "max_mse": float(
                results["Prediction_MSE"].max()
            ),

            "mean_mae": float(
                results["Prediction_MAE"].mean()
            ),

            "highest_anomaly_time": (
                highest_anomaly_row["TimeWindow"]
            ),

            "highest_anomaly_mse": float(
                highest_anomaly_row[
                    "Prediction_MSE"
                ]
            ),
        }


# =============================================================
# COMMAND-LINE TEST
# =============================================================

if __name__ == "__main__":

    analyzer = NetworkAnalyzer()

    # ---------------------------------------------------------
    # Run complete analysis
    # ---------------------------------------------------------

    results = analyzer.run()

    # ---------------------------------------------------------
    # Generate summary
    # ---------------------------------------------------------

    summary = analyzer.get_summary(
        results
    )

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------

    print()
    print(
        "========== ANALYSIS SUMMARY =========="
    )

    for key, value in summary.items():
        print(
            f"{key}: {value}"
        )

    print(
        "======================================"
    )