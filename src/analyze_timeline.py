import pandas as pd

file_path = "data/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"

df = pd.read_csv(
    file_path,
    usecols=["Timestamp", "Label"]
)

# Remove accidental header rows / invalid timestamps
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    dayfirst=True,
    errors="coerce"
)

df = df.dropna(subset=["Timestamp"])

# Remove invalid label row
df = df[df["Label"] != "Label"]

# Sort chronologically
df = df.sort_values("Timestamp").reset_index(drop=True)

print("\n===== FIRST 10 CHRONOLOGICAL FLOWS =====")
print(df.head(10).to_string(index=False))

print("\n===== FIRST INFILTRATION =====")

infiltration = df[df["Label"] == "Infilteration"]

print(
    infiltration.head(10).to_string(index=False)
)

print("\n===== FIRST INFILTRATION TIME =====")
print(infiltration["Timestamp"].min())

print("\n===== LAST BENIGN TIME =====")
benign = df[df["Label"] == "Benign"]
print(benign["Timestamp"].max())

print("\n===== FLOWS PER HOUR =====")

df["Hour"] = df["Timestamp"].dt.floor("h")

timeline = pd.crosstab(
    df["Hour"],
    df["Label"]
)

print(timeline.to_string())