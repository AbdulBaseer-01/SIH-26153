import pandas as pd

file_path = "data/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"

# Load only a few rows first
df = pd.read_csv(file_path, nrows=5)

print("\n===== COLUMNS =====")
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")

print("\n===== FIRST 5 ROWS =====")
print(df.to_string())

print("\n===== DATASET INFO =====")
print("\nNumber of columns:", len(df.columns))