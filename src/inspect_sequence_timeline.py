import numpy as np
import pandas as pd

states = pd.read_csv("data/processed/network_states.csv")

print("\n===== ALL INFILTRATION PERIODS =====\n")

states["TimeWindow"] = pd.to_datetime(states["TimeWindow"])

attack_states = states[states["IsInfiltrationState"] == 1]

# Find continuous attack periods
attack_times = attack_states["TimeWindow"].tolist()

start = attack_times[0]
previous = attack_times[0]

periods = []

for current in attack_times[1:]:

    # If gap is more than 1 minute, a new attack period starts
    if (current - previous).total_seconds() > 60:
        periods.append((start, previous))
        start = current

    previous = current

periods.append((start, previous))

for i, (start, end) in enumerate(periods, 1):

    duration = (end - start).total_seconds() / 60 + 1

    print(f"Attack Period {i}")
    print(f"Start: {start}")
    print(f"End:   {end}")
    print(f"Duration: {duration:.0f} minutes")
    print()

print("===== SEQUENCE TARGET DISTRIBUTION =====\n")

y = np.load("data/processed/y_sequences.npy")

print("Total sequences:", len(y))
print("Normal:", np.sum(y == 0))
print("Infiltration:", np.sum(y == 1))

print("\n===== ATTACK SEQUENCE INDICES =====\n")

attack_indices = np.where(y == 1)[0]

print("First attack sequence:", attack_indices[0])
print("Last attack sequence:", attack_indices[-1])

# Find gaps between attack indices
previous = attack_indices[0]
start = attack_indices[0]

sequence_periods = []

for current in attack_indices[1:]:

    if current != previous + 1:
        sequence_periods.append((start, previous))
        start = current

    previous = current

sequence_periods.append((start, previous))

for i, (start, end) in enumerate(sequence_periods, 1):
    print(f"Attack Sequence Block {i}: {start} → {end}")