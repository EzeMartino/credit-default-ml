import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

from credit_ml.config import MODEL_DIR, DEFAULT_THRESHOLD
from credit_ml.features.build import TARGET_COL, add_features
from credit_ml.modeling.artifacts import get_artifact_paths
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
    brier = brier_score_loss(y_val, proba)
    
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Save pipeline
    import joblib
    pipeline_path = MODEL_DIR / "pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    
    metadata = {
        "model_type": "logreg",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "threshold": float(DEFAULT_THRESHOLD),
        "data_source": str(RAW_PATH),
        "target_col_used": str(TARGET_COL),
        "features_expected": list(X.columns),  # columnas CRUDAS que espera la API
        "features_engineered": ["credit_utilization", "util_x_pay0"],
        "quick_metrics": {
            "roc_auc_val": float(roc_auc),
            "brier_val": float(brier),
        },
    }
    
    metadata_path = MODEL_DIR / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Exported: {pipeline_path}")
    print(f"[OK] Exported: {metadata_path}")
    print(f"[OK] ROC-AUC(val)={roc_auc:.4f} | Brier(val)={brier:.4f}")
    
    model_version = compute_model_version(pipeline_path, metadata_path)
    metadata["model_version"] = model_version

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    train_and_export()