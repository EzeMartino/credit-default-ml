from sklearn.model_selection import cross_val_score
import numpy as np


def evaluate_model(model, X, y, cv, scoring="roc_auc"):
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring
    )
    
    return{
        "mean": np.mean(scores),
        "std": np.std(scores),
        "scores": scores
    }
    
