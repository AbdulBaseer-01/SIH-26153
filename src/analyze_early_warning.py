import pandas as pd
import numpy as np

INPUT_FILE = "data/processed/world_model_scores.csv"

df = pd.read_csv(INPUT_FILE)
df["TimeWindow"] = pd.to_datetime(df["TimeWindow"])

# --------------------------------------------------
# Fixed threshold from the pre-Attack-1 normal period
# --------------------------------------------------

baseline = df[
    (df["TimeWindow"] >= pd.Timestamp("2018-03-01 01:10:00")) &
    (df["TimeWindow"] < pd.Timestamp("2018-03-01 02:00:00"))
]["Prediction_MSE"]

threshold = baseline.quantile(0.95)

print("=" * 60)
print("EARLY WARNING ANALYSIS")
print("=" * 60)

print(f"\nBaseline samples: {len(baseline)}")
print(f"95th percentile threshold: {threshold:.4f}")


# --------------------------------------------------
# Analyze one attack
# --------------------------------------------------

def analyze_attack(name, attack_start):

    attack_start = pd.Timestamp(attack_start)

    window_start = attack_start - pd.Timedelta(minutes=30)

    window = df[
        (df["TimeWindow"] >= window_start) &
        (df["TimeWindow"] < attack_start)
    ].copy()

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Attack starts: {attack_start}")
    print(f"Analysis window: {window_start} → {attack_start}")
    print(f"Samples: {len(window)}")

    if len(window) == 0:
        print("No data available.")
        return

    # 5-minute rolling average
    window["Rolling_MSE_5min"] = (
        window["Prediction_MSE"]
        .rolling(window=5, min_periods=1)
        .mean()
    )

    # Whether individual observations cross threshold
    window["Above_Threshold"] = (
        window["Prediction_MSE"] > threshold
    )

    # --------------------------------------------------
    # Individual threshold crossings
    # --------------------------------------------------

    crossings = window[
        window["Above_Threshold"]
    ]

    print(
        f"\nIndividual points above threshold: "
        f"{len(crossings)} / {len(window)}"
    )

    if len(crossings) > 0:

        first = crossings.iloc[0]

        lead = attack_start - first["TimeWindow"]

        print(
            f"First threshold crossing: "
            f"{first['TimeWindow']}"
        )

        print(
            f"Lead time: "
            f"{lead}"
        )

    else:

        print("No individual threshold crossing before attack.")


    # --------------------------------------------------
    # Sustained warning
    # 3 of the previous 5 minutes above threshold
    # --------------------------------------------------

    above = window["Above_Threshold"].astype(int)

    window["Above_Count_5min"] = (
        above
        .rolling(window=5, min_periods=5)
        .sum()
    )

    sustained = window[
        window["Above_Count_5min"] >= 3
    ]

    print(
        f"\nSustained warnings (3/5): "
        f"{len(sustained)}"
    )

    if len(sustained) > 0:

        first = sustained.iloc[0]

        lead = attack_start - first["TimeWindow"]

        print(
            f"First sustained warning: "
            f"{first['TimeWindow']}"
        )

        print(
            f"Sustained lead time: "
            f"{lead}"
        )

    else:

        print(
            "No sustained warning before attack."
        )


    # --------------------------------------------------
    # Score statistics
    # --------------------------------------------------

    print("\nPrediction error before attack:")

    print(
        f"Minimum : {window['Prediction_MSE'].min():.4f}"
    )

    print(
        f"Median  : {window['Prediction_MSE'].median():.4f}"
    )

    print(
        f"Mean    : {window['Prediction_MSE'].mean():.4f}"
    )

    print(
        f"Maximum : {window['Prediction_MSE'].max():.4f}"
    )


    # --------------------------------------------------
    # Print every minute
    # --------------------------------------------------

    print("\nTimeline:")
    print(
        window[
            [
                "TimeWindow",
                "Prediction_MSE",
                "Rolling_MSE_5min",
                "Above_Threshold"
            ]
        ].to_string(index=False)
    )


# --------------------------------------------------
# Attack 1
# --------------------------------------------------

analyze_attack(
    "ATTACK 1",
    "2018-03-01 02:00:00"
)


# --------------------------------------------------
# Attack 2
# --------------------------------------------------

analyze_attack(
    "ATTACK 2",
    "2018-03-01 09:57:00"
)