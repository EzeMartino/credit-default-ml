def test_stratified_kfold_preserves_rate():
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from src.credit_ml.models.train_logreg import load_data

    X, y = load_data("data/raw/credit_default.xls")
    y = y.to_numpy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    base_rate = y.mean()
    for _, test_idx in cv.split(np.zeros_like(y), y):
        fold_rate = y[test_idx].mean()
        assert abs(fold_rate - base_rate) < 0.01  # tolerancia razonable