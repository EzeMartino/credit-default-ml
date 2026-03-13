import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import torch

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split


from credit_ml.data.io import load_raw_credit_xls
from credit_ml.features.build import TARGET_COL
from credit_ml.modeling.pipeline import build_pipeline
from credit_ml.torch.inference import load_torch_artifacts, predict_torch

RAW_PATH = Path("data/raw/credit_default.xls")
REPORT_PATH = Path("reports/model_roc_comparison.png")

def main():
    # Load the dataset
    df = load_raw_credit_xls(RAW_PATH)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL, "ID"], errors="ignore")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Logistic Regression
    latest_dir = Path("models/latest.txt").read_text(encoding="utf-8").strip()
    sklearn_model_dir = Path("models") / latest_dir
    logistic_pipeline = joblib.load(sklearn_model_dir / "pipeline.joblib")

    logistic_probs = logistic_pipeline.predict_proba(X_val)[:, 1]
    
    # Torch MLP
    torch_model, torch_scaler, _ = load_torch_artifacts()
    torch_probs = predict_torch(torch_model, torch_scaler, X_val)
    
    # Random Forest
    rf = build_pipeline("rf")

    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_val)[:, 1]
    
    
    # Curves and AUC
    log_fpr, log_tpr, _ = roc_curve(y_val, logistic_probs)
    rf_fpr, rf_tpr, _ = roc_curve(y_val, rf_probs)
    torch_fpr, torch_tpr, _ = roc_curve(y_val, torch_probs)

    log_auc = roc_auc_score(y_val, logistic_probs)
    rf_auc = roc_auc_score(y_val, rf_probs)
    torch_auc = roc_auc_score(y_val, torch_probs)
    
    # Plotting
    Path("reports").mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(log_fpr, log_tpr, label=f"Logistic Regression (AUC = {log_auc:.3f})")
    plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.3f})")
    plt.plot(torch_fpr, torch_tpr, label=f"Torch MLP (AUC = {torch_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Baseline")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_PATH, dpi=150)
    plt.close()
    
    metrics = {
        "logistic_auc": float(log_auc),
        "rf_auc": float(rf_auc),
        "torch_auc": float(torch_auc)
    }

    with open("reports/model_auc_comparison.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Logistic Regression AUC: {float(log_auc):.4f}")
    print(f"Random Forest AUC: {float(rf_auc):.4f}")
    print(f"Torch MLP AUC: {float(torch_auc):.4f}")

    print(f"[OK] Saved ROC comparison plot to {REPORT_PATH}")

if __name__ == "__main__":
    main()