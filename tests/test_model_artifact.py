import joblib
from pathlib import Path

from credit_ml.config import LATEST_FILE

def test_model_artifact_loads():
    model_path = Path(f"models/{LATEST_FILE.read_text(encoding="utf-8").strip()}/pipeline.joblib")
    
    model = joblib.load(model_path)
    
    assert model is not None
    assert hasattr(model, "predict_proba")