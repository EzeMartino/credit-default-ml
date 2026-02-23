import pandas as pd
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from src.models.train_logreg import build_model, load_data
from src.models.evaluate import evaluate_model

import json
from pathlib import Path


RANDOM_STATE = 42

def main():

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

    SCORINGS = ["roc_auc", "average_precision", "f1"]

    results = {}

    for name, model in MODELS.items():
        results[name] = {}
        for s in SCORINGS:
            out = evaluate_model(model, X, y, cv, scoring=s)
            results[name][s] = {
                "mean": float(out["mean"]),
                "std": float(out["std"])
            }
    results = dict(sorted(results.items()))
    Path("reports").mkdir(exist_ok=True)

    with open("reports/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
            
if __name__ == "__main__":
    main()