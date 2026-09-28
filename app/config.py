from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"

WORLD_MODEL_PATH = MODEL_DIR / "world_model_predictor.keras"
WORLD_SCALER_PATH = MODEL_DIR / "world_model_scaler.pkl"

NETWORK_STATES_PATH = PROCESSED_DIR / "network_states.csv"
WORLD_SCORES_PATH = PROCESSED_DIR / "world_model_scores.csv"
ANOMALY_FEATURES_PATH = PROCESSED_DIR / "anomaly_features.csv"
FORECAST_RESULTS_PATH = PROCESSED_DIR / "forecast_engine_results.csv"