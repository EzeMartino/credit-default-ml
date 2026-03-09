import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from credit_ml.models.train_logreg import build_model, load_data
from credit_ml.evaluations.evaluate import evaluate_cv_scores, evaluate_cv_thresholded

import json
from pathlib import Path
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="reports")
    return p.parse_args()

RANDOM_STATE = 42

def main():
    args = parse_args()
    REPORTS = Path(args.output_dir)
    REPORTS.mkdir(parents=True, exist_ok=True)
    
    X, y = load_data("data/raw/credit_default.xls")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Dummy baseline
    dummy = DummyClassifier(strategy="most_frequent")

    # Log model
    log_model = build_model(X, 'log_reg')
    
    # RF model
    rf_model = build_model(X, 'rf')
    
    MODELS = {
        "dummy_most_frequent": dummy,
        "logreg_log_payamt": log_model,
        "random_forest": rf_model,
    }

    SCORINGS = ["roc_auc", "average_precision"]
    THRESHOLD = 0.6
    
    results = {}

    for name, model in MODELS.items():
        results[name] = {}
        results[name]["cv_score_metrics"] = evaluate_cv_scores(model, X, y, cv, SCORINGS)
        results[name]["cv_threshold_metrics"] = evaluate_cv_thresholded(model, X, y, cv, threshold=THRESHOLD)
    
    results = dict(sorted(results.items()))
    
    Path("reports").mkdir(exist_ok=True)

    with open(REPORTS / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Guardar y_true una sola vez
    np.save(REPORTS / "oof_y_true.npy", y.to_numpy())

    for name, model in MODELS.items():
        if name == "dummy_most_frequent":
            continue  # optional, it does not contribute to the threshold.
        y_proba_oof = cross_val_predict(
            model, X, y, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        np.save(REPORTS / f"oof_{name}_proba.npy", y_proba_oof)
            
if __name__ == "__main__":
    main()