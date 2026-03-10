from pathlib import Path

from sklearn.model_selection import train_test_split
import torch
from credit_ml.features.build import TARGET_COL
from credit_ml.data.io import load_raw_credit_xls
from credit_ml.torch.dataset import build_tensor_dataset, build_dataloader
from credit_ml.torch.model import CreditMLP

RAW_PATH = Path("data/raw/credit_default.xls")


df = load_raw_credit_xls(RAW_PATH)

y = df[TARGET_COL].astype(int)
X = df.drop(columns=[TARGET_COL, "ID"], errors="ignore")
# Simple split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dataset = build_tensor_dataset(X_train, y_train)
loader = build_dataloader(dataset, batch_size=128, shuffle=True)

for X_batch,y_batch in loader:
    print(X_batch.shape, X_batch.dtype)
    print(y_batch.shape, y_batch.dtype)
    break


## Model
model = CreditMLP(input_dim=23)

X = torch.randn(128, 23)

y= model(X)

print(y.shape, y.dtype)