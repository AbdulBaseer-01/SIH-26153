import numpy as np
import pandas as pd

from app.inference import WorldModelInference


NETWORK_STATES = "data/processed/network_states.csv"
OUTPUT = "data/processed/world_forecast_evaluation.csv"

SEQUENCE_LENGTH = 10
MAX_HORIZON = 15


def load_data():
    states = pd.read_csv(
        NETWORK_STATES,
        parse_dates=["TimeWindow"],
    )

    states = states.sort_values(
        "TimeWindow"
    ).reset_index(drop=True)

    return states


def is_continuous(states, start, end):
    if start < 0 or end >= len(states):
        return False

    times = pd.to_datetime(
        states.iloc[start:end + 1]["TimeWindow"]
    ).reset_index(drop=True)

    if len(times) < 2:
        return True

    return all(
        times.iloc[i] - times.iloc[i - 1]
        == pd.Timedelta(minutes=1)
        for i in range(1, len(times))
    )


def prepare_sequence(engine, states, end_index):
    start_index = end_index - SEQUENCE_LENGTH + 1

    if start_index < 0:
        return None

    if not is_continuous(
        states,
        start_index,
        end_index,
    ):
        return None

    columns = engine.get_state_columns()

    values = (
        states.iloc[
            start_index:end_index + 1
        ][columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    scaled = engine.scaler.transform(
        values
    )

    return np.expand_dims(
        scaled,
        axis=0,
    )


def calculate_actual_error(
    engine,
    prediction,
    actual,
):
    columns = engine.get_state_columns()

    actual_values = (
        actual[columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    actual_scaled = engine.scaler.transform(
        pd.DataFrame(
            [actual_values.values],
            columns=columns,
        )
    )[0]

    errors = prediction - actual_scaled

    mse = float(
        np.mean(errors ** 2)
    )

    mae = float(
        np.mean(np.abs(errors))
    )

    return mse, mae


def calculate_baseline(
    engine,
    states,
):
    """
    Calculate normal-state prediction error
    using exactly the same one-step inference
    used by the existing historical pipeline.
    """

    errors = []

    for index in range(
        SEQUENCE_LENGTH - 1,
        len(states) - 1,
    ):

        target_index = index + 1

        if (
            pd.Timestamp(
                states.iloc[target_index]["TimeWindow"]
            )
            !=
            pd.Timestamp(
                states.iloc[index]["TimeWindow"]
            )
            + pd.Timedelta(minutes=1)
        ):
            continue

        # Target must be normal.
        target_normal = (
            int(
                states.iloc[target_index][
                    "IsInfiltrationState"
                ]
            )
            == 0
        )

        if not target_normal:
            continue

        try:
            sequence = prepare_sequence(
                engine,
                states,
                index,
            )

            if sequence is None:
                continue

            prediction = engine.model.predict(
                sequence,
                verbose=0,
            )[0]

            mse, _ = calculate_actual_error(
                engine,
                prediction,
                states.iloc[target_index],
            )

            if np.isfinite(mse):
                errors.append(mse)

        except Exception:
            continue

    errors = np.asarray(
        errors,
        dtype=float,
    )

    if len(errors) == 0:
        raise RuntimeError(
            "No normal prediction errors were collected. "
            "Check IsInfiltrationState values and "
            "network_states.csv."
        )

    return {
        "errors": errors,
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95)),
        "p99": float(np.percentile(errors, 99)),
    }


def recursive_forecast(
    engine,
    states,
    source_index,
):
    """
    Starting from the latest 10 real states,
    recursively predict the next 15 states.
    """

    sequence = prepare_sequence(
        engine,
        states,
        source_index,
    )

    if sequence is None:
        return None

    predictions = []

    current_sequence = sequence.copy()

    for step in range(
        1,
        MAX_HORIZON + 1,
    ):

        prediction = engine.model.predict(
            current_sequence,
            verbose=0,
        )[0]

        predictions.append(
            prediction.copy()
        )

        current_sequence = np.concatenate(
            [
                current_sequence[:, 1:, :],
                prediction.reshape(
                    1,
                    1,
                    -1,
                ),
            ],
            axis=1,
        )

    return predictions


def main():

    print("=" * 70)
    print("WORLD MODEL FORWARD FORECAST EVALUATION")
    print("=" * 70)

    states = load_data()

    engine = WorldModelInference()

    print(
        f"\nNetwork states: {len(states)}"
    )

    print(
        f"Model features: "
        f"{len(engine.get_state_columns())}"
    )

    print(
        f"Forecast horizon: "
        f"{MAX_HORIZON} minutes"
    )

    # --------------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------------

    print(
        "\nCalculating normal-state baseline..."
    )

    baseline = calculate_baseline(
        engine,
        states,
    )

    baseline_median = baseline["median"]
    baseline_95 = baseline["p95"]
    baseline_99 = baseline["p99"]

    print(
        f"Normal samples: "
        f"{len(baseline['errors'])}"
    )

    print(
        f"Normal median MSE: "
        f"{baseline_median:.6f}"
    )

    print(
        f"Normal 95th percentile: "
        f"{baseline_95:.6f}"
    )

    print(
        f"Normal 99th percentile: "
        f"{baseline_99:.6f}"
    )

    # --------------------------------------------------------------
    # FORWARD FORECASTS
    # --------------------------------------------------------------

    print(
        "\nRunning recursive forward forecasts..."
    )

    results = []

    for index in range(
        SEQUENCE_LENGTH - 1,
        len(states) - MAX_HORIZON,
    ):

        source_time = pd.Timestamp(
            states.iloc[index]["TimeWindow"]
        )

        predictions = recursive_forecast(
            engine,
            states,
            index,
        )

        if predictions is None:
            continue

        for step, prediction in enumerate(
            predictions,
            start=1,
        ):

            target_index = index + step

            actual = states.iloc[
                target_index
            ]

            target_time = pd.Timestamp(
                actual["TimeWindow"]
            )

            expected_time = (
                source_time
                + pd.Timedelta(
                    minutes=step
                )
            )

            # Stop when the dataset has a gap.
            if target_time != expected_time:
                break

            mse, mae = calculate_actual_error(
                engine,
                prediction,
                actual,
            )

            ratio = (
                mse / baseline_median
                if baseline_median > 0
                else np.nan
            )

            results.append(
                {
                    "SourceTime": source_time,
                    "ForecastStep": step,
                    "TargetTime": target_time,
                    "ActualMSE": mse,
                    "ActualMAE": mae,
                    "AnomalyRatio": ratio,
                    "Above95": int(
                        mse > baseline_95
                    ),
                    "Above99": int(
                        mse > baseline_99
                    ),
                    "ActualInfiltration": int(
                        actual[
                            "IsInfiltrationState"
                        ]
                    ),
                }
            )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT,
        index=False,
    )

    print(
        f"\nSaved: {OUTPUT}"
    )

    print(
        f"Rows generated: "
        f"{len(results_df)}"
    )

    # --------------------------------------------------------------
    # HORIZON SUMMARY
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("HORIZON SUMMARY")
    print("=" * 70)

    for horizon in [1, 5, 10, 15]:

        subset = results_df[
            results_df["ForecastStep"]
            == horizon
        ]

        if subset.empty:
            continue

        normal = subset[
            subset["ActualInfiltration"]
            == 0
        ]

        infiltration = subset[
            subset["ActualInfiltration"]
            == 1
        ]

        print()
        print(
            f"{horizon}-MINUTE FORECAST"
        )

        print(
            f"Normal samples: "
            f"{len(normal)}"
        )

        print(
            f"Infiltration samples: "
            f"{len(infiltration)}"
        )

        if not normal.empty:

            print(
                f"Normal mean MSE: "
                f"{normal['ActualMSE'].mean():.4f}"
            )

            print(
                f"Normal >99th: "
                f"{normal['Above99'].mean() * 100:.2f}%"
            )

        if not infiltration.empty:

            print(
                f"Infiltration mean MSE: "
                f"{infiltration['ActualMSE'].mean():.4f}"
            )

            print(
                f"Infiltration >99th: "
                f"{infiltration['Above99'].mean() * 100:.2f}%"
            )

    # --------------------------------------------------------------
    # PRE-ATTACK SOURCE WINDOWS
    # --------------------------------------------------------------

    campaigns = {
        "ATTACK_1": (
            pd.Timestamp(
                "2018-03-01 01:30:00"
            ),
            pd.Timestamp(
                "2018-03-01 01:59:00"
            ),
        ),
        "ATTACK_2": (
            pd.Timestamp(
                "2018-03-01 09:30:00"
            ),
            pd.Timestamp(
                "2018-03-01 09:56:00"
            ),
        ),
    }

    print()
    print("=" * 70)
    print("PRE-ATTACK FORECAST SIGNAL")
    print("=" * 70)

    for name, (
        start,
        end,
    ) in campaigns.items():

        subset = results_df[
            (results_df["SourceTime"] >= start)
            & (results_df["SourceTime"] <= end)
        ]

        print()
        print(name)

        if subset.empty:
            print("  No forecast data.")
            continue

        for horizon in [
            1,
            5,
            10,
            15,
        ]:

            h = subset[
                subset["ForecastStep"]
                == horizon
            ]

            if h.empty:
                continue

            print(
                f"  +{horizon:02d}m:"
                f" mean MSE="
                f"{h['ActualMSE'].mean():.4f},"
                f" >99th="
                f"{h['Above99'].mean() * 100:.1f}%"
            )

    # --------------------------------------------------------------
    # EARLY WARNING CHECK
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("EARLY-WARNING CHECK")
    print("=" * 70)

    for name, attack_start in {
        "ATTACK_1":
            pd.Timestamp(
                "2018-03-01 02:00:00"
            ),
        "ATTACK_2":
            pd.Timestamp(
                "2018-03-01 09:57:00"
            ),
    }.items():

        print()
        print(name)

        for horizon in [
            1,
            5,
            10,
            15,
        ]:

            h = results_df[
                results_df["ForecastStep"]
                == horizon
            ].copy()

            h = h[
                h["SourceTime"]
                < attack_start
            ]

            h = h[
                h["TargetTime"]
                >= attack_start
            ]

            h = h[
                h["Above99"]
                == 1
            ]

            if h.empty:
                print(
                    f"  +{horizon:02d}m: "
                    f"no >99th signal"
                )
                continue

            first = h.iloc[0]

            lead = (
                attack_start
                - first["SourceTime"]
            ).total_seconds() / 60

            print(
                f"  +{horizon:02d}m: "
                f"first >99th signal at "
                f"{first['SourceTime'].strftime('%H:%M')}, "
                f"lead={lead:.0f} min"
            )


if __name__ == "__main__":
    main()