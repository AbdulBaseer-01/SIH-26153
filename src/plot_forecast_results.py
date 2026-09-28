import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

SCORES_FILE = Path("data/processed/world_model_scores.csv")
FORECAST_FILE = Path("data/processed/forecast_engine_results.csv")
STATES_FILE = Path("data/processed/network_states.csv")

OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

scores = pd.read_csv(
    SCORES_FILE,
    parse_dates=["TimeWindow"]
)

forecast = pd.read_csv(
    FORECAST_FILE,
    parse_dates=["TimeWindow"]
)

states = pd.read_csv(
    STATES_FILE,
    parse_dates=["TimeWindow"]
)


# ============================================================
# MERGE
# ============================================================

# Forecast file already contains Prediction_MSE,
# so don't merge that column from scores twice.

forecast_cols = [
    "TimeWindow",
    "Forecast_5m",
    "Forecast_10m",
    "Forecast_15m",
    "Forecast_Score",
    "Forecast_State",
]

df = forecast[forecast_cols].copy()

# Add Prediction_MSE from world model scores.
score_cols = [
    "TimeWindow",
    "Prediction_MSE",
    "Prediction_MAE",
]

df = df.merge(
    scores[score_cols],
    on="TimeWindow",
    how="left"
)

# Add ground-truth network state.
state_cols = [
    "TimeWindow",
    "IsInfiltrationState",
    "Infiltration Ratio",
]

df = df.merge(
    states[state_cols],
    on="TimeWindow",
    how="left"
)

df = df.sort_values("TimeWindow").reset_index(drop=True)


# ============================================================
# ATTACK WINDOWS
# ============================================================

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


# ============================================================
# PLOT 1
# FULL WORLD MODEL TIMELINE
# ============================================================

fig, ax = plt.subplots(figsize=(16, 7))

ax.plot(
    df["TimeWindow"],
    df["Prediction_MSE"],
    linewidth=1.4,
    label="World Model Prediction MSE"
)

for i, attack in enumerate(ATTACKS):

    ax.axvspan(
        attack["start"],
        attack["end"],
        alpha=0.20,
        label=attack["name"]
    )

    ax.axvline(
        attack["start"],
        linestyle="--",
        linewidth=1
    )


# Baseline threshold
baseline = df[
    (df["TimeWindow"] >= pd.Timestamp("2018-03-01 01:10:00"))
    & (df["TimeWindow"] <= pd.Timestamp("2018-03-01 01:59:00"))
]

threshold_95 = baseline["Prediction_MSE"].quantile(0.95)

ax.axhline(
    threshold_95,
    linestyle=":",
    linewidth=1.5,
    label="95th percentile baseline"
)

ax.set_title(
    "World Model Prediction Error Across Network Timeline"
)

ax.set_xlabel("Time")
ax.set_ylabel("Prediction MSE")

ax.legend()
ax.grid(alpha=0.25)

fig.autofmt_xdate()

plt.tight_layout()

output = OUTPUT_DIR / "world_model_full_timeline.png"

plt.savefig(
    output,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(f"SAVED: {output}")


# ============================================================
# PLOT 2
# ATTACK 2 CLOSE-UP
# ============================================================

start = pd.Timestamp("2018-03-01 09:35:00")
end = pd.Timestamp("2018-03-01 10:05:00")

attack2 = df[
    (df["TimeWindow"] >= start)
    & (df["TimeWindow"] <= end)
].copy()

fig, ax = plt.subplots(figsize=(16, 7))

ax.plot(
    attack2["TimeWindow"],
    attack2["Prediction_MSE"],
    marker="o",
    markersize=3,
    linewidth=1.8,
    label="Prediction MSE"
)

ax.axhline(
    threshold_95,
    linestyle=":",
    linewidth=1.5,
    label="95th percentile threshold"
)

ax.axvline(
    pd.Timestamp("2018-03-01 09:51:00"),
    linestyle="--",
    linewidth=1.5,
    label="First forecast signal — 09:51"
)

ax.axvline(
    pd.Timestamp("2018-03-01 09:57:00"),
    linestyle="--",
    linewidth=1.5,
    label="Attack starts — 09:57"
)

ax.axvspan(
    pd.Timestamp("2018-03-01 09:57:00"),
    pd.Timestamp("2018-03-01 10:05:00"),
    alpha=0.20
)

signal_row = attack2[
    attack2["TimeWindow"]
    == pd.Timestamp("2018-03-01 09:51:00")
]

if not signal_row.empty:

    signal_value = signal_row["Prediction_MSE"].iloc[0]

    ax.annotate(
        "6-minute pre-attack signal",
        xy=(
            pd.Timestamp("2018-03-01 09:51:00"),
            signal_value
        ),
        xytext=(
            pd.Timestamp("2018-03-01 09:43:00"),
            attack2["Prediction_MSE"].max() * 0.55
        ),
        arrowprops=dict(arrowstyle="->"),
    )


ax.set_title(
    "Attack 2 — Pre-Attack Anomaly and Forecast Signal"
)

ax.set_xlabel("Time")
ax.set_ylabel("Prediction MSE")

ax.legend()
ax.grid(alpha=0.25)

fig.autofmt_xdate()

plt.tight_layout()

output = OUTPUT_DIR / "attack2_forecast_closeup.png"

plt.savefig(
    output,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(f"SAVED: {output}")


# ============================================================
# PLOT 3
# FORECAST HORIZON COMPARISON
# ============================================================

start = pd.Timestamp("2018-03-01 09:40:00")
end = pd.Timestamp("2018-03-01 09:57:00")

window = df[
    (df["TimeWindow"] >= start)
    & (df["TimeWindow"] <= end)
].copy()

state_map = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
}

fig, ax = plt.subplots(figsize=(16, 6))

horizon_positions = {
    "Forecast_5m": 0,
    "Forecast_10m": 3,
    "Forecast_15m": 6,
}

for column, offset in horizon_positions.items():

    values = window[column].map(state_map)

    ax.step(
        window["TimeWindow"],
        values + offset,
        where="mid",
        linewidth=2,
        label=column.replace("Forecast_", "") + " horizon"
    )


ax.axvline(
    pd.Timestamp("2018-03-01 09:57:00"),
    linestyle="--",
    linewidth=1.5,
    label="Attack starts — 09:57"
)


ax.set_yticks([
    0, 1, 2,
    3, 4, 5,
    6, 7, 8
])

ax.set_yticklabels([
    "LOW", "MEDIUM", "HIGH",
    "LOW", "MEDIUM", "HIGH",
    "LOW", "MEDIUM", "HIGH"
])

ax.set_title(
    "Forecast State Across Prediction Horizons — Attack 2"
)

ax.set_xlabel("Time")
ax.set_ylabel("Forecast State")

ax.legend()
ax.grid(alpha=0.25)

fig.autofmt_xdate()

plt.tight_layout()

output = OUTPUT_DIR / "forecast_horizon_comparison.png"

plt.savefig(
    output,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(f"SAVED: {output}")


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("ALL PLOTS GENERATED")
print("=" * 70)

print("1.", OUTPUT_DIR / "world_model_full_timeline.png")
print("2.", OUTPUT_DIR / "attack2_forecast_closeup.png")
print("3.", OUTPUT_DIR / "forecast_horizon_comparison.png")