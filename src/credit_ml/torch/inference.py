import json
import joblib
import torch
from pathlib import Path

from credit_ml.torch.model import CreditMLP

METADATA_PATH = Path("models/torch_metadata.json")


def load_torch_artifacts():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    scaler = joblib.load(meta["scaler_path"])
    model = CreditMLP(input_dim=meta["input_dim"])
    model.load_state_dict(torch.load(meta["weights_path"], map_location=torch.device("cpu")))
    model.eval()
    
    return model, scaler, meta