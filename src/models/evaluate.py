from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
import numpy as np


def evaluate_cv_scores(model, X, y, cv, scorings):
    """
    Métricas basadas en score (no requieren threshold): ROC-AUC, PR-AUC, etc.
    """
    out = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scorings,
        n_jobs=-1,
        return_train_score=False,
    )
    results = {}
    for s in scorings:
        key = f"test_{s}"
        results[s] = {
            "mean": float(np.mean(out[key])),
            "std": float(np.std(out[key])),
        }
    return results


def evaluate_cv_thresholded(model, X, y, cv, threshold=0.5):
    """
    Métricas de decisión @ threshold: precision/recall/f1/accuracy.
    Usa probabilidades out-of-fold con cross_val_predict.
    """
    y_proba = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    y_pred = (y_proba >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, y_pred)),
        # opcional: guardar también score-based global sobre OOF
        "roc_auc_oof": float(roc_auc_score(y, y_proba)),
        "pr_auc_oof": float(average_precision_score(y, y_proba)),
    }
