import json
import joblib

from credit_ml.config import MODEL_DIR, PIPELINE_FILENAME, METADATA_FILENAME

def get_artifact_paths():
    pipeline_path = MODEL_DIR / PIPELINE_FILENAME
    metadata_path = MODEL_DIR / METADATA_FILENAME
    return pipeline_path, metadata_path
    
    

def load_pipeline():
    pipeline_path, _ = get_artifact_paths()
    
    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Pipeline artifact not found at {pipeline_path}. "
            "Run training and export artifacts first."
        )
        
    return joblib.load(pipeline_path)
    

def load_metadata():
    _, metadata_path = get_artifact_paths()

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found at {metadata_path}. "
            "Run training and export artifacts first."
        )

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    required_keys = ["model_type", "training_timestamp_utc"]
    for key in required_keys:
        if key not in metadata:
            raise ValueError(f"Metadata missing required key: {key}")

    return metadata