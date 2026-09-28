import pandas as pd

from app.inference import WorldModelInference
from app.live_inference import LiveWorldModel
from app.live_forecast import LiveForecastEngine


NETWORK_STATES = (
    "data/processed/network_states.csv"
)


def main():

    print("=" * 70)
    print("LIVE WORLD MODEL + FORECAST TEST")
    print("=" * 70)

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
    # WORLD MODEL
    # ---------------------------------------------------------

    inference = WorldModelInference()

    live_model = LiveWorldModel(
        inference_engine=inference
    )

    # ---------------------------------------------------------
    # FORECAST ENGINE
    # ---------------------------------------------------------

    forecast = LiveForecastEngine()

    forecasts = []

    # ---------------------------------------------------------
    # REPLAY NETWORK STATES
    # ---------------------------------------------------------

    for index, row in states.iterrows():

        event = live_model.add_state(
            row
        )

        if event is not None:

            result = forecast.add_event(
                event
            )

            if result is not None:

                forecasts.append(
                    result
                )

        # Print a few checkpoints
        if index in (
            9,
            10,
            11,
            12,
            54,
            55,
            56,
            536,
            537,
            538,
        ):

            latest = (
                forecast.get_latest()
            )

            print()

            print(
                f"State {index + 1:03d} | "
                f"{row['TimeWindow']} "
            )

            if latest is not None:

                print(
                    f"  MSE       : "
                    f"{latest['MSE']:.6f}"
                )

                print(
                    f"  5m Risk   : "
                    f"{latest['RiskScore_5m']:.2f} "
                    f"({latest['RiskState_5m']})"
                )

                print(
                    f"  10m Risk  : "
                    f"{latest['RiskScore_10m']:.2f} "
                    f"({latest['RiskState_10m']})"
                )

                print(
                    f"  15m Risk  : "
                    f"{latest['RiskScore_15m']:.2f} "
                    f"({latest['RiskState_15m']})"
                )

                print(
                    f"  Reason    : "
                    f"{latest['Reason']}"
                )

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    result_df = pd.DataFrame(
        forecasts
    )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"World-model events: "
        f"{len(live_model.get_events())}"
    )

    print(
        f"Forecast events: "
        f"{len(result_df)}"
    )

    if result_df.empty:

        print(
            "ERROR: No forecast events generated."
        )

        return

    print()
    print("FIRST 5 FORECASTS")

    print(
        result_df[
            [
                "TimeWindow",
                "MSE",
                "RiskScore_5m",
                "RiskState_5m",
                "RiskScore_10m",
                "RiskState_10m",
                "RiskScore_15m",
                "RiskState_15m",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )

    print()
    print("LAST 5 FORECASTS")

    print(
        result_df[
            [
                "TimeWindow",
                "MSE",
                "RiskScore_5m",
                "RiskState_5m",
                "RiskScore_10m",
                "RiskState_10m",
                "RiskScore_15m",
                "RiskState_15m",
            ]
        ]
        .tail(5)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # ATTACK-WINDOW CHECKS
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("ATTACK WINDOW CHECKS")
    print("=" * 70)

    for timestamp in [
        "2018-03-01 01:51:00",
        "2018-03-01 01:55:00",
        "2018-03-01 01:59:00",
        "2018-03-01 09:47:00",
        "2018-03-01 09:51:00",
        "2018-03-01 09:55:00",
        "2018-03-01 09:56:00",
    ]:

        timestamp = pd.Timestamp(
            timestamp
        )

        matches = result_df[
            result_df["TimeWindow"]
            == timestamp
        ]

        if matches.empty:

            print(
                f"{timestamp} | "
                "No forecast"
            )

            continue

        row = matches.iloc[0]

        print(
            f"{timestamp} | "
            f"MSE={row['MSE']:.4f} | "
            f"5m={row['RiskScore_5m']:.1f} "
            f"{row['RiskState_5m']} | "
            f"10m={row['RiskScore_10m']:.1f} "
            f"{row['RiskState_10m']} | "
            f"15m={row['RiskScore_15m']:.1f} "
            f"{row['RiskState_15m']}"
        )

    print()
    print("=" * 70)
    print("LIVE FORECAST TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    main()