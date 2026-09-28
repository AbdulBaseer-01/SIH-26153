\# AI World Model for Proactive Cyber Defence



\## 1. Problem



Conventional network intrusion detection primarily analyses the current network state to identify suspicious or malicious activity.



This project introduces a temporal AI world model that learns normal network behaviour and predicts how the network state is expected to evolve. Deviations between predicted and observed states are used to identify anomalous behaviour and provide an early warning signal.



The system additionally performs recursive forward simulation to estimate how the network may evolve over the next several time steps.



\---



\## 2. System Overview



The system processes network traffic as a sequence of time-dependent network states.



```text

Network Traffic

&#x20;     │

&#x20;     ▼

Flow / Packet Feature Extraction

&#x20;     │

&#x20;     ▼

1-Minute Network State Construction

&#x20;     │

&#x20;     ▼

┌─────────────────────────────────────┐

│       Temporal World Model          │

│              LSTM                   │

│                                     │

│ S(t-9) ... S(t) → Predicted S(t+1) │

└──────────────────┬──────────────────┘

&#x20;                  │

&#x20;         ┌────────┴─────────┐

&#x20;         ▼                  ▼

&#x20;  Prediction Error    Forward Simulation

&#x20;         │                  │

&#x20;         ▼                  ▼

&#x20;  Anomaly Detection    K-Step Rollout

&#x20;         │                  │

&#x20;         └────────┬─────────┘

&#x20;                  ▼

&#x20;         Threat Interpretation

&#x20;                  │

&#x20;         ┌────────┴─────────┐

&#x20;         ▼                  ▼

&#x20;   Attack Stage       Feature Explanation

&#x20;         │                  │

&#x20;         └────────┬─────────┘

&#x20;                  ▼

&#x20;            Streamlit UI

