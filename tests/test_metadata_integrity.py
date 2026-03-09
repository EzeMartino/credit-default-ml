from curses import meta
import json
from pathlib import Path

def test_metadata_integrity():
    path = Path("models/metadata.json")
    
    meta = json.load(path.open())
    
    required_fields = [
        "model_type",
        "model_version",
        "training_timestamp_utc",
        "dataset_name",
        "target_name",
        "threshold",
        "feature_count",
        "training_rows",
        "positive_class_rate",
        "input_features",
        "engineered_features",
        "validation_metrics"
    ]
    
    for field in required_fields:
        assert field in meta, f"Required field '{field}' is missing from metadata"
    
    assert isinstance(meta["input_features"], list)
    assert len(meta["input_features"]) > 0