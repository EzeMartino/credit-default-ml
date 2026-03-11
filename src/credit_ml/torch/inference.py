import json
import joblib
import torch
from pathlib import Path
import pandas as pd

from credit_ml.torch.model import CreditMLP

METADATA_PATH = Path("models/torch_metadata.json")

def predict_torch(model, scaler, X: pd.DataFrame):
    X_scaled = scaler.transform(X)
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    model.eval()
    
    with torch.no_grad():
        logits = model(X_tensor)
        preds = torch.sigmoid(logits)
    
    return preds.cpu().numpy()

def load_torch_artifacts():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    scaler = joblib.load(meta["scaler_path"])
    model = CreditMLP(input_dim=meta["input_dim"])
    model.load_state_dict(torch.load(meta["weights_path"], map_location=torch.device("cpu")))
    model.eval()
    
    return model, scaler, meta