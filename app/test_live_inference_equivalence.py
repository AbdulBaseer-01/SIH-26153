import numpy as np
import pandas as pd

from app.inference import WorldModelInference
from app.live_inference import LiveWorldModel


NETWORK_STATES = "data/processed/network_states.csv"


def is_continuous_window(
    states,
    start_index,
    end_index,
):

    if start_index < 0:

        return False

    if end_index >= len(states):

        return False

    timestamps = states.iloc[
        start_index:end_index + 1
    ]["TimeWindow"]

    timestamps = pd.to_datetime(
        timestamps
    )

    differences = (
        timestamps.diff()
        .dropna()
    )

    return bool(
        (differences == pd.Timedelta(minutes=1))
        .all()
    )


def main():

    print("=" * 70)
    print("LIVE vs HISTORICAL INFERENCE EQUIVALENCE TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load states
    # ---------------------------------------------------------

    states = pd.read_csv(
        NETWORK_STATES
    )

    states["TimeWindow"] = pd.to_datetime(
        states["TimeWindow"]
    )

    print(
        f"Total states: {len(states)}"
    )

    # ---------------------------------------------------------
    # Historical engine
    # ---------------------------------------------------------

    historical = WorldModelInference()

    # ---------------------------------------------------------
    # Live engine
    # ---------------------------------------------------------

    live = LiveWorldModel(
        inference_engine=historical
    )

    # ---------------------------------------------------------
    # Replay
    # ---------------------------------------------------------

    for _, row in states.iterrows():

        live.add_state(row)

    live_events = live.get_events()

    print(
        f"Live prediction events: "
        f"{len(live_events)}"
    )

    # ---------------------------------------------------------
    # Compare
    # ---------------------------------------------------------

    results = []

    skipped_gap_events = 0

    skipped_invalid_events = 0

    for _, event in live_events.iterrows():

        timestamp = pd.Timestamp(
            event["TimeWindow"]
        )

        matches = states.index[
            states["TimeWindow"] == timestamp
        ]

        if len(matches) == 0:

            skipped_invalid_events += 1

            continue

        target_index = int(
            matches[0]
        )

        end_index = (
            target_index - 1
        )

        # -----------------------------------------------------
        # Need 10 states ending at end_index
        # -----------------------------------------------------

        start_index = (
            end_index - 10 + 1
        )

        if not is_continuous_window(
            states,
            start_index,
            end_index,
        ):

            skipped_gap_events += 1

            continue

        # -----------------------------------------------------
        # Target must also be exactly one minute later
        # -----------------------------------------------------

        current_time = pd.Timestamp(
            states.iloc[end_index][
                "TimeWindow"
            ]
        )

        target_time = pd.Timestamp(
            states.iloc[target_index][
                "TimeWindow"
            ]
        )

        if (
            target_time - current_time
            != pd.Timedelta(minutes=1)
        ):

            skipped_gap_events += 1

            continue

        # -----------------------------------------------------
        # Historical inference
        # -----------------------------------------------------

        historical_result = (
            historical.calculate_prediction_error(
                end_index
            )
        )

        historical_mse = float(
            historical_result["mse"]
        )

        historical_mae = float(
            historical_result["mae"]
        )

        # -----------------------------------------------------
        # Live inference
        # -----------------------------------------------------

        live_mse = float(
            event["MSE"]
        )

        live_mae = float(
            event["MAE"]
        )

        results.append({

            "TimeWindow": timestamp,

            "LiveMSE": live_mse,

            "HistoricalMSE": historical_mse,

            "MSEDifference": abs(
                live_mse
                - historical_mse
            ),

            "LiveMAE": live_mae,

            "HistoricalMAE": historical_mae,

            "MAEDifference": abs(
                live_mae
                - historical_mae
            ),
        })

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    comparison = pd.DataFrame(
        results
    )

    if comparison.empty:

        print()
        print(
            "ERROR: No valid comparisons."
        )

        return

    max_mse_difference = float(
        comparison[
            "MSEDifference"
        ].max()
    )

    mean_mse_difference = float(
        comparison[
            "MSEDifference"
        ].mean()
    )

    max_mae_difference = float(
        comparison[
            "MAEDifference"
        ].max()
    )

    mean_mae_difference = float(
        comparison[
            "MAEDifference"
        ].mean()
    )

    mse_mismatches = comparison[
        comparison[
            "MSEDifference"
        ] > 1e-6
    ]

    mae_mismatches = comparison[
        comparison[
            "MAEDifference"
        ] > 1e-6
    ]

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(
        f"Live prediction events: "
        f"{len(live_events)}"
    )

    print(
        f"Valid comparisons: "
        f"{len(comparison)}"
    )

    print(
        f"Skipped gap events: "
        f"{skipped_gap_events}"
    )

    print(
        f"Skipped invalid events: "
        f"{skipped_invalid_events}"
    )

    print()

    print(
        f"Maximum MSE difference: "
        f"{max_mse_difference:.12f}"
    )

    print(
        f"Mean MSE difference: "
        f"{mean_mse_difference:.12f}"
    )

    print(
        f"MSE mismatches (>1e-6): "
        f"{len(mse_mismatches)}"
    )

    print()

    print(
        f"Maximum MAE difference: "
        f"{max_mae_difference:.12f}"
    )

    print(
        f"Mean MAE difference: "
        f"{mean_mae_difference:.12f}"
    )

    print(
        f"MAE mismatches (>1e-6): "
        f"{len(mae_mismatches)}"
    )

    # ---------------------------------------------------------
    # First 5
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("FIRST 5 COMPARISONS")
    print("=" * 70)

    print(
        comparison[
            [
                "TimeWindow",
                "LiveMSE",
                "HistoricalMSE",
                "MSEDifference",
                "LiveMAE",
                "HistoricalMAE",
                "MAEDifference",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # Last 5
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("LAST 5 COMPARISONS")
    print("=" * 70)

    print(
        comparison[
            [
                "TimeWindow",
                "LiveMSE",
                "HistoricalMSE",
                "MSEDifference",
                "LiveMAE",
                "HistoricalMAE",
                "MAEDifference",
            ]
        ]
        .tail(5)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # Mismatches
    # ---------------------------------------------------------

    if len(mse_mismatches) > 0:

        print()
        print("=" * 70)
        print("MSE MISMATCHES")
        print("=" * 70)

        print(
            mse_mismatches[
                [
                    "TimeWindow",
                    "LiveMSE",
                    "HistoricalMSE",
                    "MSEDifference",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    # ---------------------------------------------------------
    # Verdict
    # ---------------------------------------------------------

    print()
    print("=" * 70)

    if (
        max_mse_difference <= 1e-6
        and max_mae_difference <= 1e-6
        and len(mse_mismatches) == 0
        and len(mae_mismatches) == 0
    ):

        print(
            "EQUIVALENCE TEST PASSED"
        )

        print()

        print(
            "Live and historical inference "
            "produce identical MSE and MAE "
            "for all continuous transitions."
        )

    else:

        print(
            "EQUIVALENCE TEST FAILED"
        )

        print()

        print(
            "Live and historical inference "
            "produce different results."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()