import numpy as np
import pandas as pd

TARGET_COL = "default payment next month"

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Prevent 0 division and blank cells from excel 
    if "BILL_AMT" in df.columns and "LIMIT_BAL" in df.columns:
        denom = df["LIMIT_BAL"].replace(0, np.nan)
        df["credit_utilization"] = (df["BILL_AMT1"]/ denom).fillna(0.0)
    else:
        df["credit_utilization"] = 0.0
        
    # Interaction utilization * PAY_0 (if PAY_0 doesn't exist, 0)
    if "PAY_0" in df.columns:
        df["util_x_pay0"] = df["credit_utilization"] * df["PAY_0"].astype(float)
    else:
        df["util_x_pay0"] = 0.0
        
    return df