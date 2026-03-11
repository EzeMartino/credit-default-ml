import json

import joblib
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from credit_ml.data.io import load_raw_credit_xls
from credit_ml.features.build import TARGET_COL
from credit_ml.torch.dataset import build_tensor_dataset, build_dataloader
from credit_ml.torch.model import CreditMLP

RAW_PATH = Path("data/raw/credit_default.xls")
MODEL_OUT = Path("models/torch_model.pt")
SCALER_OUT = Path("models/torch_scaler.joblib")
METADATA_OUT = Path("models/torch_metadata.json")

def train_torch_model(
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.001
) -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"The dataset doesn't exist in {RAW_PATH.resolve()}")
    
    df = load_raw_credit_xls(RAW_PATH)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL, "ID"], errors="ignore")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    ## Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # build_tensor_dataset receives a DataFrame, so we need to convert back
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    
    train_dataset = build_tensor_dataset(X_train_scaled, y_train)
    val_dataset = build_tensor_dataset(X_val_scaled, y_val)
    
    train_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = CreditMLP(input_dim=X.shape[1])
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_predictions = []
        val_targets = []
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                
                val_loss += loss.item()
                val_predictions.extend(predictions.tolist())
                val_targets.extend(y_batch.tolist())
        
        avg_val_loss = val_loss / len(val_loader)
        val_roc_auc = roc_auc_score(val_targets, val_predictions)
        
        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Validation Loss: {avg_val_loss:.4f} | "
              f"Validation ROC AUC: {val_roc_auc:.4f}"
            )
    
    
    
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUT)
    
    # Also save the scaler and metadata for reproducibility and inference
    SCALER_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_OUT)
    
    metadata = {
        "model_type": "torch_mlp",
        "input_dim": X.shape[1],
        "training_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "validation_roc_auc": float(val_roc_auc),
        "scaler_path": "models/torch_scaler.joblib",
        "weights_path": "models/torch_model.pt",
    }
    
    with open(METADATA_OUT, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[OK] Torch model saved to {MODEL_OUT.resolve()}")
    
if __name__ == "__main__":
    train_torch_model()