import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

def build_tensor_dataset(X: pd.DataFrame, y: pd.Series) -> TensorDataset:

    # Convert inputs to PyTorch tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    # Create and return a TensorDataset
    return TensorDataset(X_tensor, y_tensor)

def build_dataloader(dataset, batch_size=128, shuffle=True) -> DataLoader:
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )