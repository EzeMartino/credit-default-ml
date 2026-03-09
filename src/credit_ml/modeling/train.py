import json
from datetime import datetime, timezone
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

from credit_ml.config import MODEL_DIR, DEFAULT_THRESHOLD
from credit_ml.features.build import TARGET_COL
from credit_ml.modeling.pipeline import build_pipeline
from credit_ml.data.io import load_raw_credit_xls
from credit_ml.api.versioning import compute_model_version

RAW_PATH = Path("data/raw/credit_default.xls")

def train_and_export() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"The dataset doesn't exist in {RAW_PATH.resolve()}")
    
    # Excel to Dataframe
    df = load_raw_credit_xls(RAW_PATH)
    
    # Normalization: sometimes Excel brings spaces at the end
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Did not found the target '{TARGET_COL}'. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
        
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL, "ID"], errors="ignore")
    
    # Simple split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline = build_pipeline('log_reg')
    pipeline.fit(X_train, y_train)
    
    # Fast metrics for metadata (it isn't a "serious" evaluation, it's sanity)
    proba = pipeline.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, proba)
    pr_auc = average_precision_score(y_val, proba)
    brier = brier_score_loss(y_val, proba)
    positive_rate = y_train.mean()
    
    
    staging_dir = MODEL_DIR / "_staging"
    staging_dir.mkdir(exist_ok=True)
    
    # Save pipeline
    import joblib
    pipeline_path = staging_dir / "pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    
    metadata = {
        "model_type": "logreg",
        "model_version": None,  # Placeholder, will be updated after computing the version
        
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        
        "dataset_name": "taiwan_credit_default",
        
        "target_name": str(TARGET_COL),
        
        "threshold": float(DEFAULT_THRESHOLD),
        
        "feature_count": X.shape[1],
        "training_rows": X_train.shape[0],
        
        "positive_class_rate": float(positive_rate),
        
        "input_features": list(X.columns),  # Crude columns expected from the API
        
        "engineered_features": [
            "credit_utilization", 
            "util_x_pay0"
        ],
        
        "validation_metrics": {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "brier_score": float(brier),
        },
    }
    
    metadata_path = staging_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Exported: {pipeline_path}")
    print(f"[OK] Exported: {metadata_path}")
    print(f"[OK] ROC-AUC(val)={roc_auc:.4f} | Brier(val)={brier:.4f}")
    
    model_version = compute_model_version(pipeline_path, metadata_path)
    metadata["model_version"] = model_version


    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    final_model_dir = MODEL_DIR / f"model_{model_version}"
    
    if final_model_dir.exists():
        raise FileExistsError(f"Model directory already exists: {final_model_dir}")

    staging_dir.rename(final_model_dir)
    
    (MODEL_DIR / "latest.txt").write_text(final_model_dir.name, encoding="utf-8")




if __name__ == "__main__":
    train_and_export()