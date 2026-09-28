import pandas as pd

# Load datasets
train = pd.read_csv("data/UNSW_NB15_training-set.csv")
test = pd.read_csv("data/UNSW_NB15_testing-set.csv")

print("\n===== TRAINING DATA =====")
print("Shape:", train.shape)

print("\nColumns:")
print(train.columns.tolist())

print("\nFirst 5 rows:")
print(train.head())

print("\n===== LABEL DISTRIBUTION =====")
print(train["label"].value_counts())

print("\n===== ATTACK CATEGORIES =====")
print(train["attack_cat"].value_counts())

print("\n===== DATA TYPES =====")
print(train.dtypes)