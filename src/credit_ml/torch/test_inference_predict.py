from pathlib import Path

from credit_ml.data.io import load_raw_credit_xls
from credit_ml.features.build import TARGET_COL
from credit_ml.torch.inference import load_torch_artifacts, predict_torch

RAW_PATH = Path("data/raw/credit_default.xls")

df = load_raw_credit_xls(RAW_PATH)
X = df.drop(columns=[TARGET_COL, "ID"], errors="ignore")

model, scaler, meta = load_torch_artifacts()

preds = predict_torch(model, scaler, X)

print(preds)
print(preds.shape)