import pandas as pd

from app.world_simulation import WorldModelSimulator


def analyze_timestamp(simulator, timestamp, steps=15):

    print()
    print("=" * 70)
    print(f"SIMULATION FROM {timestamp}")
    print("=" * 70)

    results = simulator.simulate(
        timestamp,
        steps=steps,
    )

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

    return results


def main():

    simulator = WorldModelSimulator()

    test_points = [
        "2018-03-01 01:55:00",
        "2018-03-01 09:51:00",
        "2018-03-01 01:30:00",
        "2018-03-01 09:30:00",
    ]

    all_results = []

    for timestamp in test_points:

        results = analyze_timestamp(
            simulator,
            pd.Timestamp(timestamp),
            steps=15,
        )

        results["Start_Time"] = pd.Timestamp(timestamp)

        all_results.append(results)

    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    output_path = (
        "data/processed/"
        "world_simulation_analysis.csv"
    )

    combined.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("SAVED")
    print("=" * 70)
    print(output_path)


if __name__ == "__main__":
    main()