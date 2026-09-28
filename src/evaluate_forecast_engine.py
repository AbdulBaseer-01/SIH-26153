import pandas as pd
import numpy as np
from pathlib import Path


INPUT_FILE = Path("data/processed/forecast_engine_results.csv")
OUTPUT_FILE = Path("data/processed/forecast_engine_evaluation.csv")


ATTACKS = [
    {
        "name": "Attack 1",
        "start": pd.Timestamp("2018-03-01 02:00:00"),
        "end": pd.Timestamp("2018-03-01 03:36:00"),
    },
    {
        "name": "Attack 2",
        "start": pd.Timestamp("2018-03-01 09:57:00"),
        "end": pd.Timestamp("2018-03-01 10:54:00"),
    },
]


HORIZONS = ["5m", "10m", "15m"]


def evaluate_signal(window, state_column, level_name):
    """
    Evaluate whether a forecast signal appears before an attack.

    level_name:
        "MEDIUM/HIGH" -> MEDIUM or HIGH
        "HIGH"        -> HIGH only
    """

    if level_name == "MEDIUM/HIGH":
        mask = window[state_column].isin(["MEDIUM", "HIGH"])
    else:
        mask = window[state_column] == "HIGH"

    signal_rows = window[mask]

    if signal_rows.empty:
        return {
            "first_signal": None,
            "lead_minutes": None,
            "sustained_signal": None,
            "sustained_lead_minutes": None,
            "signal_count": 0,
            "signal_rate": 0.0,
        }

    first = signal_rows.iloc[0]

    first_signal = first["TimeWindow"]
    lead_minutes = int(
        (window.attrs["attack_start"] - first_signal).total_seconds() / 60
    )

    sustained_signal = None
    sustained_lead = None

    # Look for two consecutive signal minutes.
    signal_times = list(signal_rows["TimeWindow"])

    for i in range(len(signal_times) - 1):
        diff = (
            signal_times[i + 1] - signal_times[i]
        ).total_seconds() / 60

        if diff == 1:
            sustained_signal = signal_times[i]
            sustained_lead = int(
                (
                    window.attrs["attack_start"]
                    - sustained_signal
                ).total_seconds()
                / 60
            )
            break

    return {
        "first_signal": first_signal,
        "lead_minutes": lead_minutes,
        "sustained_signal": sustained_signal,
        "sustained_lead_minutes": sustained_lead,
        "signal_count": len(signal_rows),
        "signal_rate": len(signal_rows) / len(window),
    }


def main():

    df = pd.read_csv(INPUT_FILE, parse_dates=["TimeWindow"])
    df = df.sort_values("TimeWindow").reset_index(drop=True)

    print("=" * 70)
    print("FORECAST ENGINE EVALUATION")
    print("=" * 70)

    results = []

    # ---------------------------------------------------------
    # ATTACK EVENT EVALUATION
    # ---------------------------------------------------------

    for attack in ATTACKS:

        attack_start = attack["start"]

        # 30-minute pre-attack window
        start = attack_start - pd.Timedelta(minutes=30)
        end = attack_start - pd.Timedelta(minutes=1)

        window = df[
            (df["TimeWindow"] >= start)
            & (df["TimeWindow"] <= end)
        ].copy()

        window.attrs["attack_start"] = attack_start

        print()
        print("=" * 70)
        print(attack["name"])
        print("=" * 70)

        for horizon in HORIZONS:

            column = f"Forecast_{horizon}"

            print()
            print(column)

            for level in ["MEDIUM/HIGH", "HIGH"]:

                result = evaluate_signal(
                    window,
                    column,
                    level
                )

                print(f"  {level}:")

                print(
                    f"    First signal: "
                    f"{result['first_signal']}"
                )

                print(
                    f"    First lead: "
                    f"{result['lead_minutes']} min"
                )

                print(
                    f"    Sustained signal: "
                    f"{result['sustained_signal']}"
                )

                print(
                    f"    Sustained lead: "
                    f"{result['sustained_lead_minutes']} min"
                )

                print(
                    f"    Signal count: "
                    f"{result['signal_count']}/{len(window)}"
                )

                print(
                    f"    Signal rate: "
                    f"{result['signal_rate'] * 100:.2f}%"
                )

                results.append({
                    "Evaluation": "Attack",
                    "Attack": attack["name"],
                    "Horizon": horizon,
                    "Signal_Level": level,
                    "First_Signal": result["first_signal"],
                    "First_Lead_Minutes": result["lead_minutes"],
                    "Sustained_Signal": result["sustained_signal"],
                    "Sustained_Lead_Minutes": result[
                        "sustained_lead_minutes"
                    ],
                    "Signal_Count": result["signal_count"],
                    "Signal_Rate": result["signal_rate"],
                })

    # ---------------------------------------------------------
    # NORMAL PERIOD EVALUATION
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("NORMAL-PERIOD FALSE SIGNAL ANALYSIS")
    print("=" * 70)

    # Normal states are states labelled non-infiltration.
    normal = df[
        df["IsInfiltrationState"] == 0
    ].copy()

    # Exclude the attack periods themselves.
    for attack in ATTACKS:
        normal = normal[
            ~(
                (normal["TimeWindow"] >= attack["start"])
                & (normal["TimeWindow"] <= attack["end"])
            )
        ]

    for horizon in HORIZONS:

        column = f"Forecast_{horizon}"

        medium_high = normal[column].isin(
            ["MEDIUM", "HIGH"]
        ).sum()

        high_only = (
            normal[column] == "HIGH"
        ).sum()

        total = len(normal)

        mh_rate = medium_high / total
        high_rate = high_only / total

        print()
        print(column)

        print(
            f"  MEDIUM/HIGH: "
            f"{medium_high}/{total} "
            f"({mh_rate * 100:.2f}%)"
        )

        print(
            f"  HIGH only: "
            f"{high_only}/{total} "
            f"({high_rate * 100:.2f}%)"
        )

        results.append({
            "Evaluation": "Normal",
            "Attack": "Normal Period",
            "Horizon": horizon,
            "Signal_Level": "MEDIUM/HIGH",
            "First_Signal": None,
            "First_Lead_Minutes": None,
            "Sustained_Signal": None,
            "Sustained_Lead_Minutes": None,
            "Signal_Count": medium_high,
            "Signal_Rate": mh_rate,
        })

        results.append({
            "Evaluation": "Normal",
            "Attack": "Normal Period",
            "Horizon": horizon,
            "Signal_Level": "HIGH",
            "First_Signal": None,
            "First_Lead_Minutes": None,
            "Sustained_Signal": None,
            "Sustained_Lead_Minutes": None,
            "Signal_Count": high_only,
            "Signal_Rate": high_rate,
        })

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    result_df = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("SAVED")
    print("=" * 70)

    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()