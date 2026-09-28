import pandas as pd

train = pd.read_csv("data/UNSW_NB15_training-set.csv")

print("\n===== FIRST 30 ATTACK LABELS =====")
print(train[["id", "attack_cat", "label"]].head(30).to_string(index=False))

print("\n===== LAST 30 ATTACK LABELS =====")
print(train[["id", "attack_cat", "label"]].tail(30).to_string(index=False))

print("\n===== ID CHECK =====")
print("First ID:", train["id"].iloc[0])
print("Last ID:", train["id"].iloc[-1])
print("Is ID sorted?", train["id"].is_monotonic_increasing)