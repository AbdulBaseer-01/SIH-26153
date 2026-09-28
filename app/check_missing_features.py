import pandas as pd
import numpy as np


FILE = "data/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"


df = pd.read_csv(
    FILE,
    low_memory=False
)

df.columns = df.columns.str.strip()


print("==============================")
print("MISSING FEATURE CHECK")
print("==============================")


for column in [
    "Flow Byts/s",
    "Flow Pkts/s"
]:

    numeric = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    print()
    print(column)
    print("------------------------------")
    print("Rows:", len(df))
    print(
        "Original missing:",
        int(df[column].isna().sum())
    )
    print(
        "After numeric conversion:",
        int(numeric.isna().sum())
    )
    print(
        "Invalid/non-numeric:",
        int(
            (
                numeric.isna()
                & df[column].notna()
            ).sum()
        )
    )


print()
print("==============================")