"""Shared repository configuration; safe to import in any environment."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_PATH", "models/model.joblib")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
