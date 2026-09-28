import pandas as pd
import numpy as np


WORLD_SCORES = "data/processed/world_model_scores.csv"
FORECAST_RESULTS = "data/processed/forecast_engine_results.csv"
WORLD_FORECAST = "data/processed/world_forecast_evaluation.csv"

OUTPUT = "data/processed/combined_warning_analysis.csv"


ATTACKS = {
    "Attack 1": {
        "start": pd.Timestamp("2018-03-01 02:00:00"),
        "pre_start": pd.Timestamp("2018-03-01 01:10:00"),
    },
    "Attack 2": {
        "start": pd.Timestamp("2018-03-01 09:57:00"),
        "pre_start": pd.Timestamp("2018-03-01 09:00:00"),
    },
}


def load_csv(path):
    df = pd.read_csv(path)

    for column in [
        "TimeWindow",
        "SourceTime",
        "TargetTime",
    ]:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column]
            )

    return df


def first_world_warning(
    world,
    start,
    attack_start,
    threshold,
):
    """
    Find first >99th-percentile world-model
    prediction error before the attack.
    """

    subset = world[
        (world["TimeWindow"] >= start)
        & (world["TimeWindow"] < attack_start)
    ].copy()

    if subset.empty:
        return None

    flagged = subset[
        subset["Prediction_MSE"] > threshold
    ]

    if flagged.empty:
        return None

    row = flagged.sort_values(
        "TimeWindow"
    ).iloc[0]

    warning_time = row["TimeWindow"]

    lead = (
        attack_start - warning_time
    ).total_seconds() / 60

    return {
        "WarningTime": warning_time,
        "LeadMinutes": lead,
        "MSE": float(
            row["Prediction_MSE"]
        ),
        "Threshold": float(threshold),
    }


def first_recursive_warning(
    forecast,
    attack_start,
    horizon,
):
    """
    Find the earliest recursive forecast source
    time whose target reaches the attack window
    and whose actual future state exceeds the
    normal 99th-percentile error.

    NOTE:
    This is an evaluation metric, not a deployable
    real-time prediction. It uses the actual future
    state to determine whether the forecasted state
    was anomalous.
    """

    subset = forecast[
        (forecast["ForecastStep"] == horizon)
        & (forecast["SourceTime"] < attack_start)
        & (forecast["TargetTime"] >= attack_start)
        & (forecast["Above99"] == 1)
    ].copy()

    if subset.empty:
        return None

    row = subset.sort_values(
        "SourceTime"
    ).iloc[0]

    warning_time = row["SourceTime"]

    lead = (
        attack_start - warning_time
    ).total_seconds() / 60

    return {
        "WarningTime": warning_time,
        "LeadMinutes": lead,
        "TargetTime": row["TargetTime"],
        "MSE": float(
            row["ActualMSE"]
        ),
    }


def first_engine_warning(
    forecast_engine,
    attack_start,
    horizon,
):
    """
    Find first HIGH forecast-engine signal
    before the attack.

    Uses the existing Forecast_5m,
    Forecast_10m and Forecast_15m columns.
    """

    state_column = (
        f"Forecast_{horizon}m"
    )

    score_column = (
        f"Risk_{horizon}m"
    )

    if state_column not in forecast_engine.columns:
        return None

    subset = forecast_engine[
        forecast_engine["TimeWindow"]
        < attack_start
    ].copy()

    subset = subset[
        subset[state_column] == "HIGH"
    ]

    if subset.empty:
        return None

    row = subset.sort_values(
        "TimeWindow"
    ).iloc[0]

    warning_time = row["TimeWindow"]

    lead = (
        attack_start - warning_time
    ).total_seconds() / 60

    score = np.nan

    if score_column in row.index:
        score = float(
            row[score_column]
        )

    return {
        "WarningTime": warning_time,
        "LeadMinutes": lead,
        "RiskScore": score,
    }


def main():

    print("=" * 72)
    print("COMBINED EARLY-WARNING ANALYSIS")
    print("=" * 72)

    # --------------------------------------------------------------
    # LOAD
    # --------------------------------------------------------------

    world = load_csv(
        WORLD_SCORES
    )

    forecast_engine = load_csv(
        FORECAST_RESULTS
    )

    world_forecast = load_csv(
        WORLD_FORECAST
    )

    print(
        f"\nWorld-model scores: "
        f"{len(world)} rows"
    )

    print(
        f"Forecast-engine rows: "
        f"{len(forecast_engine)}"
    )

    print(
        f"Recursive forecast rows: "
        f"{len(world_forecast)}"
    )

    # --------------------------------------------------------------
    # WORLD MODEL THRESHOLD
    # --------------------------------------------------------------

    normal = world[
        world["IsInfiltrationState"] == 0
    ]["Prediction_MSE"].dropna()

    threshold_95 = float(
        np.percentile(
            normal,
            95,
        )
    )

    threshold_99 = float(
        np.percentile(
            normal,
            99,
        )
    )

    print()
    print(
        f"World-model normal 95th: "
        f"{threshold_95:.6f}"
    )

    print(
        f"World-model normal 99th: "
        f"{threshold_99:.6f}"
    )

    # --------------------------------------------------------------
    # ANALYZE CAMPAIGNS
    # --------------------------------------------------------------

    output_rows = []

    for attack_name, config in ATTACKS.items():

        attack_start = config["start"]
        pre_start = config["pre_start"]

        print()
        print("=" * 72)
        print(
            f"{attack_name} "
            f"({attack_start.strftime('%H:%M')})"
        )
        print("=" * 72)

        # ----------------------------------------------------------
        # CURRENT WORLD MODEL
        # ----------------------------------------------------------

        world_warning = first_world_warning(
            world,
            pre_start,
            attack_start,
            threshold_99,
        )

        print()
        print(
            "World-model anomaly:"
        )

        if world_warning:

            print(
                f"  First warning: "
                f"{world_warning['WarningTime'].strftime('%H:%M')}"
            )

            print(
                f"  Lead time: "
                f"{world_warning['LeadMinutes']:.0f} min"
            )

            print(
                f"  MSE: "
                f"{world_warning['MSE']:.4f}"
            )

        else:

            print(
                "  No >99th-percentile warning"
            )

        output_rows.append(
            {
                "Attack": attack_name,
                "Signal": "World Model Anomaly",
                "Horizon": "Current",
                "WarningTime": (
                    world_warning["WarningTime"]
                    if world_warning
                    else pd.NaT
                ),
                "LeadMinutes": (
                    world_warning["LeadMinutes"]
                    if world_warning
                    else np.nan
                ),
            }
        )

        # ----------------------------------------------------------
        # RECURSIVE WORLD FORECAST
        # ----------------------------------------------------------

        for horizon in [
            1,
            5,
            10,
            15,
        ]:

            result = first_recursive_warning(
                world_forecast,
                attack_start,
                horizon,
            )

            print()
            print(
                f"Recursive forecast +{horizon}m:"
            )

            if result:

                print(
                    f"  First warning: "
                    f"{result['WarningTime'].strftime('%H:%M')}"
                )

                print(
                    f"  Lead time: "
                    f"{result['LeadMinutes']:.0f} min"
                )

                print(
                    f"  Target: "
                    f"{result['TargetTime'].strftime('%H:%M')}"
                )

                print(
                    f"  Actual target MSE: "
                    f"{result['MSE']:.4f}"
                )

            else:

                print(
                    "  No >99th-percentile warning"
                )

            output_rows.append(
                {
                    "Attack": attack_name,
                    "Signal": "Recursive World Forecast",
                    "Horizon": f"+{horizon}m",
                    "WarningTime": (
                        result["WarningTime"]
                        if result
                        else pd.NaT
                    ),
                    "LeadMinutes": (
                        result["LeadMinutes"]
                        if result
                        else np.nan
                    ),
                }
            )

        # ----------------------------------------------------------
        # EXISTING FORECAST ENGINE
        # ----------------------------------------------------------

        for horizon in [
            5,
            10,
            15,
        ]:

            result = first_engine_warning(
                forecast_engine,
                attack_start,
                horizon,
            )

            print()
            print(
                f"Existing forecast engine "
                f"+{horizon}m:"
            )

            if result:

                print(
                    f"  First HIGH: "
                    f"{result['WarningTime'].strftime('%H:%M')}"
                )

                print(
                    f"  Lead time: "
                    f"{result['LeadMinutes']:.0f} min"
                )

                if np.isfinite(
                    result["RiskScore"]
                ):

                    print(
                        f"  Risk score: "
                        f"{result['RiskScore']:.2f}/10"
                    )

            else:

                print(
                    "  No HIGH warning"
                )

            output_rows.append(
                {
                    "Attack": attack_name,
                    "Signal": "Forecast Engine",
                    "Horizon": f"+{horizon}m",
                    "WarningTime": (
                        result["WarningTime"]
                        if result
                        else pd.NaT
                    ),
                    "LeadMinutes": (
                        result["LeadMinutes"]
                        if result
                        else np.nan
                    ),
                }
            )

    # --------------------------------------------------------------
    # SAVE
    # --------------------------------------------------------------

    output = pd.DataFrame(
        output_rows
    )

    output.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 72)
    print("SUMMARY TABLE")
    print("=" * 72)

    print(
        output.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()