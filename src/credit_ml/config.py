from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR= BASE_DIR / "models"
PIPELINE_FILENAME="pipeline.joblib"

METADATA_FILENAME="metadata.json"

DEFAULT_THRESHOLD= 0.22999999999999932 # Recall 0.6 =<