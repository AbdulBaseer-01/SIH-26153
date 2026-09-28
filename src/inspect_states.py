import pandas as pd

FILE = "data/processed/network_states.csv"

df = pd.read_csv(FILE)

print("\n===== NETWORK STATES =====")
print(f"Shape: {df.shape}")

print("\n===== TIME RANGE =====")
print("Start:", df["TimeWindow"].min())
print("End:  ", df["TimeWindow"].max())

print("\n===== INFILTRATION STATES =====")

infiltration = df[df["IsInfiltrationState"] == 1]

print(f"Total infiltration states: {len(infiltration)}")

print("\nFirst 20 infiltration states:")

print(
    infiltration[
        [
            "TimeWindow",
            "Flow Count",
            "Infiltration Ratio",
            "IsInfiltrationState"
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\n===== STATE TRANSITIONS =====")

labels = df["IsInfiltrationState"].values

for i in range(1, len(labels)):
    if labels[i] != labels[i - 1]:

        previous_state = labels[i - 1]
        current_state = labels[i]

        print(
            f"{df.iloc[i]['TimeWindow']} : "
            f"{previous_state} → {current_state}"
        )