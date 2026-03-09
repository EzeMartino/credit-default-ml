import joblib
from pathlib import Path

def test_model_artifact_loads():
    model_path = Path("models/pipeline.joblib")
    
    model = joblib.load(model_path)
    
    assert model is not None
    assert hasattr(model, "predict_proba")