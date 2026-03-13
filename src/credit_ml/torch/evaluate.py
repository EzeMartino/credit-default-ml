import json
from pathlib import Path

from credit_ml.features.build import TARGET_COL
from credit_ml.torch.inference import load_torch_artifacts, predict_torch
from credit_ml.data.io import load_raw_credit_xls

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

RAW_PATH = Path("data/raw/credit_default.xls")

def evaluate_torch_model():
    # Load the dataset
    df = load_raw_credit_xls(RAW_PATH)
    
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL, "ID"], errors="ignore")
    
    # Load the model and any necessary artifacts
    model, scaler, _ = load_torch_artifacts()

    # Make predictions
    preds = predict_torch(model, scaler, X)

    # Calculate evaluation metrics
    auc_roc = roc_auc_score(y, preds)
    auc_pr = average_precision_score(y, preds)
    brier_score = brier_score_loss(y, preds)

    report = {
        'model_type': 'torch_mlp',
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'brier_score': brier_score,
        'n_samples': len(y)
    }
    
    Path("reports").mkdir(parents=True, exist_ok=True)
    
    with open("reports/torch_evaluation.json", "w") as f:
        json.dump(report, f, indent=2) 
        
    print(f"ROC-AUC: {auc_roc:.4f}")
    print(f"PR-AUC: {auc_pr:.4f}")
    print(f"Brier Score: {brier_score:.4f}")
    print("[OK] Saved report to reports/torch_evaluation.json")
        
if __name__ == "__main__":
    evaluate_torch_model()