import pandas as pd

file_path = "data/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"

# Load only Timestamp and Label to save memory
df = pd.read_csv(
    file_path,
    usecols=["Timestamp", "Label"]
)

print("\n===== DATASET SHAPE =====")
print("Total flows:", len(df))

print("\n===== LABEL DISTRIBUTION =====")
print(df["Label"].value_counts())

print("\n===== UNIQUE LABELS =====")
for label in df["Label"].unique():
    print(label)

print("\n===== TIMESTAMP RANGE =====")

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    dayfirst=True,
    errors="coerce"
)

print("Earliest:", df["Timestamp"].min())
print("Latest:  ", df["Timestamp"].max())

print("\nInvalid timestamps:", df["Timestamp"].isna().sum())