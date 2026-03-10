import torch.nn as nn
import torch

class CreditMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x).squeeze(1)