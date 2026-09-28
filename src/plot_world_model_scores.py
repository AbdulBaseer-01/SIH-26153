import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "data/processed/world_model_scores.csv"

df = pd.read_csv(INPUT_FILE)
df["TimeWindow"] = pd.to_datetime(df["TimeWindow"])

plt.figure(figsize=(15, 6))

# Plot the actual continuous prediction-error timeline
plt.plot(
    df["TimeWindow"],
    df["Prediction_MSE"],
    linewidth=1
)

# Attack periods
plt.axvspan(
    pd.Timestamp("2018-03-01 02:00:00"),
    pd.Timestamp("2018-03-01 03:37:00"),
    alpha=0.15,
    label="Attack 1"
)

plt.axvspan(
    pd.Timestamp("2018-03-01 09:57:00"),
    pd.Timestamp("2018-03-01 10:55:00"),
    alpha=0.15,
    label="Attack 2"
)

plt.xlabel("Time")
plt.ylabel("Next-State Prediction MSE")

plt.title(
    "Urban Network World Model — Next-State Prediction Error"
)

plt.legend()

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    "reports/world_model_prediction_error.png",
    dpi=200
)

plt.show()