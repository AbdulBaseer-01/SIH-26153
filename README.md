\# AI World Model for Proactive Network Defence



\## Smart India Hackathon 2026 — Problem Statement 26153



\*\*Team:\*\* Byte Masters  

\*\*Project:\*\* AI Network Attack Forecasting using World Models



\---



\## Overview



Traditional network security systems often focus on detecting malicious activity after it has already occurred.



This project approaches network security as a \*\*temporal prediction problem\*\*.



Instead of only asking:



> Is this network traffic malicious?



the system asks:



> Does the network appear to be evolving differently from the behaviour the model expects?



The system learns how normal network behaviour evolves over time, predicts the next network state, measures the deviation between predicted and observed behaviour, and uses forward simulation to generate early-warning risk signals.



\### Core Pipeline



```text

Network Traffic

&#x20;     ↓

Feature Processing

&#x20;     ↓

1-Minute Network States

&#x20;     ↓

10-State Temporal Window

&#x20;     ↓

LSTM World Model

&#x20;     ↓

Next-State Prediction

&#x20;     ↓

Prediction Deviation

&#x20;     ↓

Risk / Attack-Stage Signal

&#x20;     ↓

K-Step Forward Simulation

&#x20;     ↓

Streamlit Dashboard

```



\---



\## Solution



The system models network behaviour as a sequence of changing states rather than treating every network flow as an isolated event.



The current prototype:



1\. Processes network-flow traffic.

2\. Aggregates traffic into one-minute network states.

3\. Represents each state using network features.

4\. Uses the previous 10 states as temporal context.

5\. Uses an LSTM World Model to predict the next network state.

6\. Compares the predicted state with the observed state.

7\. Uses prediction error as an anomaly/deviation signal.

8\. Generates forward-looking risk signals.

9\. Provides a high-level attack-stage interpretation.

10\. Identifies features contributing to the prediction deviation.

11\. Recursively simulates future network states.

12\. Presents the results through an offline Streamlit dashboard.



\---



\## Current Prototype



The working MVP currently includes:



\- CIC-IDS-2018 flow-level network traffic

\- Timestamp-aware preprocessing

\- 1-minute network-state construction

\- 79-feature network-state representation

\- 10-minute temporal input windows

\- LSTM-based next-state prediction

\- Prediction MSE and MAE analysis

\- Feature-level deviation analysis

\- 5-minute, 10-minute and 15-minute forecast risk signals

\- Simplified attack-stage interpretation

\- Recursive K-step World Model simulation

\- Offline Streamlit dashboard



\### Current Input



The primary working prototype uses:



```text

CIC-IDS-2018

Thursday-01-03-2018 Traffic CSV

```



\### Current World Model



```text

10 historical network states

&#x20;           ↓

&#x20;      LSTM Model

&#x20;           ↓

&#x20;Predicted next network state

&#x20;           ↓

&#x20;Compare with observed state

&#x20;           ↓

&#x20;Prediction deviation

```



\---



\## System Architecture



```text

┌───────────────────────────────┐

│       Network Telemetry       │

│       Flow-level Data         │

└───────────────┬───────────────┘

&#x20;               │

&#x20;               ▼

┌───────────────────────────────┐

│      Feature Processing       │

│ Cleaning • Imputation •       │

│ Timestamp Processing          │

└───────────────┬───────────────┘

&#x20;               │

&#x20;               ▼

┌───────────────────────────────┐

│     Temporal State Builder    │

│       1-Minute States         │

└───────────────┬───────────────┘

&#x20;               │

&#x20;               ▼

┌───────────────────────────────┐

│       LSTM World Model        │

│                               │

│ 10 States → Next-State        │

│ Prediction                    │

└───────────────┬───────────────┘

&#x20;               │

&#x20;        ┌──────┴───────┐

&#x20;        │              │

&#x20;        ▼              ▼

┌───────────────┐ ┌────────────────┐

│   Deviation   │ │ K-Step Forward │

│   Analysis    │ │   Simulation   │

└───────┬───────┘ └───────┬────────┘

&#x20;       │                 │

&#x20;       └────────┬────────┘

&#x20;                │

&#x20;                ▼

┌───────────────────────────────┐

│ Risk + Attack-Stage Analysis  │

│ + Driving Feature Errors      │

└───────────────┬───────────────┘

&#x20;               │

&#x20;               ▼

┌───────────────────────────────┐

│      Streamlit Dashboard      │

└───────────────────────────────┘

```



\---



\## Repository Structure



```text

SIH-26153/

│

├── README.md

├── requirements.txt

│

├── app/

│   ├── dashboard.py

│   ├── config.py

│   ├── inference.py

│   ├── analysis.py

│   ├── pipeline.py

│   ├── forecast.py

│   ├── live\_state.py

│   ├── live\_inference.py

│   ├── live\_forecast.py

│   ├── attack\_stage.py

│   └── world\_simulation.py

│

├── src/

│   ├── preprocess\_states.py

│   ├── create\_world\_sequences.py

│   ├── train\_world\_model.py

│   ├── evaluate\_world\_model.py

│   ├── build\_anomaly\_features.py

│   ├── create\_forecast\_targets.py

│   ├── create\_forecast\_dataset.py

│   ├── build\_forecast\_engine.py

│   ├── evaluate\_simulation\_signal.py

│   └── other analysis scripts

│

├── models/

│   ├── world\_model\_predictor.keras

│   ├── world\_model\_lstm.keras

│   ├── world\_model\_lstm\_chronological.keras

│   ├── world\_model\_campaign.keras

│   ├── scaler.pkl

│   └── live\_feature\_medians.pkl

│

├── data/

│   ├── raw datasets

│   └── processed/

│

├── notebooks/

└── reports/

```



\---



\## Technology Stack



\### Programming

\- Python 3.11



\### Machine Learning

\- TensorFlow

\- Keras

\- LSTM

\- Scikit-learn



\### Data Processing

\- Pandas

\- NumPy

\- Joblib



\### Dashboard

\- Streamlit

\- Plotly



\### Datasets

\- CIC-IDS-2018

\- UNSW-NB15



\---



\## Installation



\### 1. Create the Conda Environment



Open \*\*Anaconda Prompt\*\*:



```bash

conda create -n sih python=3.11

```



Activate it:



```bash

conda activate sih

```



\### 2. Move Into the Project



```bash

cd C:\\Users\\abdul\\SIH-26153

```



\### 3. Install Dependencies



```bash

pip install -r requirements.txt

```



If `requirements.txt` is unavailable:



```bash

pip install tensorflow pandas numpy scikit-learn streamlit plotly joblib

```



\---



\## Dataset Setup



The primary prototype uses CIC-IDS-2018 flow-level network traffic.



Place the required CSV inside:



```text

data/

```



The main dataset used by the prototype is:



```text

data/Thursday-01-03-2018\_TrafficForML\_CICFlowMeter.csv

```



The project also uses UNSW-NB15 for additional analysis and experimentation.



\### Large Dataset Notice



The raw CIC-IDS-2018 CSV is large and may not be included in the GitHub repository.



If the file is not present after cloning the repository, download the required dataset separately and place it at:



```text

data/Thursday-01-03-2018\_TrafficForML\_CICFlowMeter.csv

```



\---



\## Running the Dashboard



Activate the environment:



```bash

conda activate sih

```



Move into the project:



```bash

cd C:\\Users\\abdul\\SIH-26153

```



Run Streamlit:



```bash

streamlit run app\\dashboard.py

```



Streamlit will provide a local address similar to:



```text

http://localhost:8501

```



Open the address in your browser.



The dashboard runs locally and does not require a cloud API for the core inference workflow.



\---



\## Dashboard Features



The Streamlit dashboard provides an offline visual interface for analysing network behaviour.



\### Network Analysis



A timestamp selector allows analysis of different points in the network timeline.



\### Model Information



The dashboard displays:



\- World Model

\- Number of state features

\- Temporal input window

\- Simulation mode

\- Dataset

\- State resolution



\### Network Anomaly Timeline



Displays World Model next-state prediction error over time.



The primary anomaly metric is:



```text

Prediction MSE

```



A larger prediction error indicates that the observed network state differs more strongly from the state predicted by the World Model.



\### Forward Threat Forecast



Provides risk signals over:



```text

5 minutes

10 minutes

15 minutes

```



\### Attack Stage



Provides a high-level interpretation of the current network behaviour.



\### Driving Feature Errors



Displays network features contributing to the prediction deviation at the selected timestamp.



\### K-Step World Model Simulation



Recursively predicts future network states using the trained World Model.



\---



\## How the World Model Works



The current World Model uses an LSTM to learn temporal network-state transitions.



Each network state represents one minute of network activity.



The model receives the previous 10 continuous network states:



```text

S(t-9)

&#x20;  ↓

S(t-8)

&#x20;  ↓

...

&#x20;  ↓

S(t-1)

&#x20;  ↓

S(t)

&#x20;  ↓

LSTM World Model

&#x20;  ↓

Predicted S(t+1)

```



The predicted state is then compared against the actual observed state.



The difference between these states becomes the primary anomaly/deviation signal.



\---



\## Prediction Deviation



The system compares:



```text

Expected Network State

&#x20;         ↓

&#x20;     World Model

&#x20;         ↓

&#x20;  Predicted State

&#x20;         ↓

Compare with observed state

&#x20;         ↓

Prediction Deviation

```



The main metric currently used is:



```text

Mean Squared Error (MSE)

```



A high prediction error means that the network has evolved differently from what the model expected.



This provides a temporal anomaly signal without requiring the World Model to directly classify every flow as malicious.



\---



\## Forecast Risk



The prototype also generates forward-looking risk signals using the anomaly timeline and forecast features.



The dashboard provides signals for:



```text

5-minute horizon

10-minute horizon

15-minute horizon

```



These values are \*\*risk scores/signals\*\*.



They are \*\*not calibrated probabilities\*\*.



Therefore, a score such as `9/10` should not be interpreted as a `90% probability of attack`.



\---



\## K-Step World Model Simulation



The World Model can recursively simulate future network states.



```text

Historical States

&#x20;      ↓

Predict S(t+1)

&#x20;      ↓

Feed prediction back

&#x20;      ↓

Predict S(t+2)

&#x20;      ↓

Feed prediction back

&#x20;      ↓

Predict S(t+3)

&#x20;      ↓

...

&#x20;      ↓

Predict S(t+K)

```



The simulator produces several signals, including:



\- Future State Deviation

\- Mean Absolute Deviation

\- Maximum Deviation

\- Transition Shock

\- Predicted State Change

\- Future Stability

\- Simulation Risk



The simulation allows the system to examine how the learned World Model expects network behaviour to evolve beyond the current state.



\### Important



`Simulation Risk` is a relative model-derived signal.



It is not a calibrated probability.



\---



\## Attack-Stage Interpretation



The prototype contains a simplified attack-stage interpretation layer.



Possible outputs include:



```text

Normal



Suspicious Network Activity



Suspicious / Pre-Attack Anomaly



Reconnaissance / Discovery



Command \& Control-Like Behaviour



Active Infiltration / Exploitation

```



The purpose of this layer is to provide a high-level security interpretation of the model output.



It is designed around the attack progression concept described in the SIH problem statement.



The current implementation does \*\*not\*\* claim definitive identification of a specific MITRE ATT\&CK technique.



\---



\## Reproducing the Pipeline



The main processing and training stages are available in `src/`.



\### Step 1 — Build Network States



```bash

python src/preprocess\_states.py

```



This processes the flow-level traffic and converts it into one-minute network states.



Output:



```text

data/processed/network\_states.csv

```



\### Step 2 — Create World Model Sequences



```bash

python src/create\_world\_sequences.py

```



This creates continuous temporal sequences for World Model training.



\### Step 3 — Train the World Model



```bash

python src/train\_world\_model.py

```



The trained World Model is saved under:



```text

models/

```



\### Step 4 — Evaluate the World Model



```bash

python src/evaluate\_world\_model.py

```



This evaluates next-state prediction error across the network timeline.



\### Step 5 — Build Anomaly Features



```bash

python src/build\_anomaly\_features.py

```



This creates additional features derived from the World Model anomaly signal.



\### Step 6 — Create Forecast Targets



```bash

python src/create\_forecast\_targets.py

```



This generates future attack/infiltration targets for different forecasting horizons.



\### Step 7 — Create Forecast Dataset



```bash

python src/create\_forecast\_dataset.py

```



\### Step 8 — Run the Dashboard



```bash

streamlit run app/dashboard.py

```



\---



\## Evaluation



The project includes analysis and evaluation of:



\- Next-state prediction

\- Prediction MSE

\- Prediction MAE

\- Network anomaly behaviour

\- Pre-attack behaviour

\- Forecast risk signals

\- Detection lead time

\- False-warning behaviour

\- Unseen attack campaigns

\- Logistic Regression baseline

\- K-step simulation



The corresponding analysis scripts are available under:



```text

src/

```



\---



\## Attack Campaign Analysis



The CIC-IDS-2018 sample used in the prototype contains observed infiltration periods.



The system analyses the transition between:



```text

Normal Network Behaviour

&#x20;         ↓

Pre-Attack Behaviour

&#x20;         ↓

Observed Infiltration

&#x20;         ↓

Return Toward Normal Behaviour

```



This allows the project to investigate temporal attack progression rather than relying only on isolated malicious flows.



\---



\## Unseen Attack Validation



The project also includes a separate campaign-split experiment.



The purpose is to test whether the learned behaviour model can generalize beyond the attack campaign used during training.



This experiment demonstrates that unseen-attack generalization remains a significant challenge.



The current prototype should therefore be treated as a research/MVP system rather than a production intrusion detection system.



\---



\## Limitations



The current implementation is a research prototype.



Current limitations include:



\- The working dashboard primarily uses flow-level CSV data.

\- Full PCAP/packet-level ingestion is not currently the primary dashboard input.

\- The current World Model uses an LSTM.

\- Current risk scores are not calibrated probabilities.

\- The attack-stage layer is a simplified interpretation layer.

\- Full MITRE ATT\&CK technique mapping is not implemented.

\- Current explainability uses feature-level prediction errors.

\- SHAP/attention-based explainability is not currently implemented.

\- Generalization to unseen attack campaigns remains challenging.

\- The current system is not intended to replace a production IDS/SOC platform.



\---



\## Proposed Extensions



The complete system can be extended further to cover the broader SIH problem statement.



\### PCAP and Packet-Level Processing



Future versions can support:



\- PCAP input

\- Packet-level feature extraction

\- TTL statistics

\- TCP window characteristics

\- Payload-size distributions

\- Port-scan signatures

\- Retransmission statistics

\- Packet timing characteristics



\### Graph-Based Network States



Network states can be represented as graphs containing:



```text

Hosts

&#x20; ↓

Connections

&#x20; ↓

Network Graph

```



This would allow graph-based models such as GNNs to be incorporated.



\### Alternative World Models



Potential extensions include:



\- Transformer

\- GNN

\- Hybrid temporal/graph models

\- Latent-state World Models



\### Calibrated Forecasting



Future versions can provide calibrated:



```text

Infiltration Probability

```



instead of the current relative risk signals.



\### Explainable AI



Potential extensions include:



\- SHAP

\- Attention-based attribution

\- Feature importance

\- Temporal feature contribution



\### MITRE ATT\&CK Mapping



Future versions can map predicted behaviour to specific MITRE ATT\&CK techniques when sufficient evidence is available.



\---



\## Offline Operation



The core prototype is designed to run locally.



```text

Dataset

&#x20;  ↓

Local preprocessing

&#x20;  ↓

Local model

&#x20;  ↓

Local inference

&#x20;  ↓

Local dashboard

```



No cloud inference API is required.



\---



\## Quick Start



For an evaluator who only wants to run the existing dashboard:



```bash

conda create -n sih python=3.11

conda activate sih



cd C:\\Users\\abdul\\SIH-26153



pip install -r requirements.txt



streamlit run app\\dashboard.py

```



Then open:



```text

http://localhost:8501

```



\---



\## Project Resources



\- Source Code: GitHub Repository

\- Architecture Document: `BYTE\_Masters\_Architecture\_Document.pdf`

\- SIH Presentation: Project submission deck



\---



\## Team



\### Team Byte Masters



\*\*Smart India Hackathon 2026\*\*



\*\*Problem Statement:\*\* 26153



\*\*Project:\*\* AI Network Attack Forecasting using World Models



\---



\## Disclaimer



This repository contains a research/prototype implementation developed for the Smart India Hackathon problem statement.



The anomaly, forecast and simulation values generated by the system are model-derived analytical signals. They should not be interpreted as calibrated probabilities or definitive security verdicts.

