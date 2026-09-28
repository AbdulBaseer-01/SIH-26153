import pandas as pd

from app.live_state import LiveStateBuilder


def main():

    print("Loading historical flow data...")

    df = pd.read_csv(
        "data/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
        nrows=5000
    )

    # Same model columns used by the world model.
    state_df = pd.read_csv(
        "data/processed/network_states.csv",
        nrows=1
    )

    state_columns = [
        column
        for column in state_df.columns
        if column not in [
            "TimeWindow",
            "Infiltration Ratio",
            "IsInfiltrationState"
        ]
    ]

    print()
    print("Model features:", len(state_columns))

    builder = LiveStateBuilder(
        state_columns=state_columns
    )

    states = builder.add_flows(df)

    # Flush the final incomplete window for testing.
    states.extend(builder.flush())

    result = pd.DataFrame(states)

    print()
    print("==============================")
    print("LIVE STATE BUILDER TEST")
    print("==============================")

    print("Completed states:", len(result))

    if not result.empty:

        print(
            "First state:",
            result.iloc[0]["TimeWindow"]
        )

        print(
            "Last state:",
            result.iloc[-1]["TimeWindow"]
        )

        print(
            "State feature count:",
            len(state_columns)
        )

        print()
        print("First state:")
        print(
            result.iloc[0][
                ["TimeWindow", "Flow Count"]
            ]
        )

        print()
        print(
            "Model sequence shape:",
            builder.get_model_sequence().shape
            if builder.ready_for_world_model()
            else "Not enough states"
        )

    print("==============================")


if __name__ == "__main__":
    main()