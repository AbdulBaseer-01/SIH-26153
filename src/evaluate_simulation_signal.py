import numpy as np
import pandas as pd

from app.world_simulation import WorldModelSimulator


CHECKPOINTS = {
    "Normal_1": "2018-03-01 01:30:00",
    "Pre_Attack_1": "2018-03-01 01:55:00",
    "Normal_2": "2018-03-01 09:30:00",
    "Pre_Attack_2": "2018-03-01 09:51:00",
}

STEPS = 15


def summarize(name, timestamp, results):
    print()
    print("=" * 80)
    print(f"{name} | {timestamp}")
    print("=" * 80)

    cols = [
        "Step",
        "Future_Time",
        "Transition_Shock",
        "Future_State_Deviation",
        "Predicted_State_Change",
        "Future_Stability",
        "Simulation_Risk",
    ]

    print(results[cols].to_string(index=False))

    print()
    print("SUMMARY")
    print("-" * 80)

    first = results.iloc[0]

    print(f"First-step shock:       {first['Transition_Shock']:.6f}")
    print(f"First-step deviation:   {first['Future_State_Deviation']:.6f}")
    print(f"First-step risk:        {first['Simulation_Risk']:.6f}")

    print(
        f"Max transition shock:    "
        f"{results['Transition_Shock'].max():.6f}"
    )

    print(
        f"Mean transition shock:   "
        f"{results['Transition_Shock'].mean():.6f}"
    )

    print(
        f"Max state deviation:     "
        f"{results['Future_State_Deviation'].max():.6f}"
    )

    print(
        f"Mean state deviation:    "
        f"{results['Future_State_Deviation'].mean():.6f}"
    )

    print(
        f"Max simulation risk:     "
        f"{results['Simulation_Risk'].max():.6f}"
    )

    print(
        f"Mean simulation risk:    "
        f"{results['Simulation_Risk'].mean():.6f}"
    )

    print(
        f"Future stability:        "
        f"{results['Future_Stability'].iloc[0]:.6f}"
    )

    risk_values = results["Simulation_Risk"].to_numpy()

    if len(risk_values) >= 2:
        delta = risk_values[-1] - risk_values[0]

        if delta > 0.1:
            trend = "INCREASING"
        elif delta < -0.1:
            trend = "DECREASING"
        else:
            trend = "STABLE"

        print(f"Risk trend:              {trend}")
        print(f"Risk change:             {delta:.6f}")


def main():
    simulator = WorldModelSimulator()

    print("=" * 80)
    print("SIMULATION-ONLY WORLD MODEL EVALUATION")
    print("=" * 80)
    print(f"Simulation horizon: {STEPS} minutes")
    print()
    print("IMPORTANT:")
    print("This evaluation uses ONLY the historical state at the")
    print("selected timestamp as input.")
    print("Actual future states are NOT used.")
    print()

    all_results = []

    for name, timestamp in CHECKPOINTS.items():

        try:
            results = simulator.simulate(
                timestamp,
                steps=STEPS,
            )

            summarize(name, timestamp, results)

            temp = results.copy()
            temp["Scenario"] = name
            all_results.append(temp)

        except Exception as exc:
            print()
            print(f"{name} FAILED")
            print(exc)

    if not all_results:
        raise RuntimeError("No simulation results were generated.")

    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    output_path = (
        "data/processed/"
        "simulation_signal_evaluation.csv"
    )

    combined.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)

    summary = (
        combined
        .groupby("Scenario")
        .agg(
            First_Shock=(
                "Transition_Shock",
                "first",
            ),
            Mean_Shock=(
                "Transition_Shock",
                "mean",
            ),
            Max_Shock=(
                "Transition_Shock",
                "max",
            ),
            Mean_Deviation=(
                "Future_State_Deviation",
                "mean",
            ),
            Max_Deviation=(
                "Future_State_Deviation",
                "max",
            ),
            Mean_Risk=(
                "Simulation_Risk",
                "mean",
            ),
            Max_Risk=(
                "Simulation_Risk",
                "max",
            ),
        )
    )

    print(summary.to_string())

    print()
    print("=" * 80)
    print(f"Saved: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()