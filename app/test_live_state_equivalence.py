import numpy as np
import pandas as pd

from app.live_state import LiveStateBuilder


RAW_FILE = "data/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
STATE_FILE = "data/processed/network_states.csv"


def main():

    print("Loading raw flow data...")
    raw = pd.read_csv(
        RAW_FILE,
        low_memory=False
    )

    print("Loading original network states...")
    original = pd.read_csv(
        STATE_FILE
    )

    raw.columns = raw.columns.str.strip()

    raw["Timestamp"] = pd.to_datetime(
        raw["Timestamp"],
        dayfirst=True,
        errors="coerce"
    )

    raw = raw.dropna(
        subset=["Timestamp"]
    )

    # Exact model feature contract.
    state_columns = [
        column
        for column in original.columns
        if column not in [
            "TimeWindow",
            "Infiltration Ratio",
            "IsInfiltrationState"
        ]
    ]

    print()
    print("Model features:", len(state_columns))

    # --------------------------------------------------
    # RUN LIVE BUILDER
    # --------------------------------------------------

    builder = LiveStateBuilder(
        state_columns=state_columns
    )

    live_states = builder.add_flows(raw)

    live_states.extend(
        builder.flush()
    )

    live = pd.DataFrame(
        live_states
    )

    # --------------------------------------------------
    # ALIGN
    # --------------------------------------------------

    original["TimeWindow"] = pd.to_datetime(
        original["TimeWindow"]
    )

    live["TimeWindow"] = pd.to_datetime(
        live["TimeWindow"]
    )

    original = original[
        ["TimeWindow"] + state_columns
    ]

    live = live[
        ["TimeWindow"] + state_columns
    ]

    merged = original.merge(
        live,
        on="TimeWindow",
        how="inner",
        suffixes=("_original", "_live")
    )

    print()
    print("==============================")
    print("LIVE STATE EQUIVALENCE TEST")
    print("==============================")

    print("Original states:", len(original))
    print("Live states:", len(live))
    print("Matched states:", len(merged))

    # --------------------------------------------------
    # NUMERIC COMPARISON
    # --------------------------------------------------

    differences = []

    for column in state_columns:

        original_values = pd.to_numeric(
            merged[f"{column}_original"],
            errors="coerce"
        )

        live_values = pd.to_numeric(
            merged[f"{column}_live"],
            errors="coerce"
        )

        diff = (
            original_values - live_values
        ).abs()

        diff = diff.replace(
            [np.inf, -np.inf],
            np.nan
        )

        max_diff = diff.max()

        mean_diff = diff.mean()

        differences.append(
            (
                column,
                max_diff,
                mean_diff
            )
        )

    differences.sort(
        key=lambda x: (
            -x[1] if pd.notna(x[1])
            else 0
        )
    )

    print()
    print("Top feature differences:")

    for column, max_diff, mean_diff in differences[:10]:

        print(
            f"{column}: "
            f"max={max_diff:.10f}, "
            f"mean={mean_diff:.10f}"
        )

    # --------------------------------------------------
    # FLOW COUNT
    # --------------------------------------------------

    flow_count_diff = (
        merged["Flow Count_original"]
        - merged["Flow Count_live"]
    ).abs()

    print()
    print(
        "Flow Count mismatches:",
        int((flow_count_diff > 0).sum())
    )

    print(
        "Maximum Flow Count difference:",
        int(flow_count_diff.max())
    )

    # --------------------------------------------------
    # OVERALL
    # --------------------------------------------------

    numeric_diffs = [
        mean_diff
        for _, _, mean_diff in differences
        if pd.notna(mean_diff)
    ]

    overall_mean = np.mean(
        numeric_diffs
    )

    print()
    print(
        "Overall mean absolute difference:",
        overall_mean
    )

    print("==============================")

    if overall_mean < 1e-6:
        print("EQUIVALENCE TEST PASSED")
    else:
        print(
            "EQUIVALENCE TEST FAILED"
        )


if __name__ == "__main__":
    main()