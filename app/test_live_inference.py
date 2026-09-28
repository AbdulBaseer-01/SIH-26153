import pandas as pd

from app.live_inference import LiveWorldModel


NETWORK_STATES = "data/processed/network_states.csv"


def main():

    print("=" * 60)
    print("CONTINUOUS WORLD MODEL TEST")
    print("=" * 60)

    states = pd.read_csv(NETWORK_STATES)

    states["TimeWindow"] = pd.to_datetime(
        states["TimeWindow"]
    )

    model = LiveWorldModel()

    print(f"Total states: {len(states)}")
    print()

    for i, (_, row) in enumerate(states.iterrows(), start=1):

        event = model.add_state(row)

        if i <= 12:

            print(
                f"State {i:03d} | "
                f"{row['TimeWindow']} | "
                f"Ready={model.ready()} | "
                f"Prediction={model.has_pending_prediction()}"
            )

        if event is not None and i <= 15:

            print(
                f"  → Prediction evaluated | "
                f"MSE={event['MSE']:.6f} | "
                f"MAE={event['MAE']:.6f}"
            )

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    print(f"States processed: {len(model.state_history)}")
    print(f"Predictions evaluated: {len(model.events)}")
    print(f"Pending prediction: {model.has_pending_prediction()}")

    if model.events:

        events = model.get_events()

        print()
        print("First 5 prediction errors:")
        print(
            events[
                ["TimeWindow", "MSE", "MAE"]
            ].head().to_string(index=False)
        )

        print()
        print("Last 5 prediction errors:")
        print(
            events[
                ["TimeWindow", "MSE", "MAE"]
            ].tail().to_string(index=False)
        )


if __name__ == "__main__":
    main()