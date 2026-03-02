def test_logreg_pipeline_fit_small():
    from src.credit_ml.models.train_logreg import load_data, build_model

    X, y = load_data("data/raw/credit_default.xls")
    Xs, ys = X.iloc[:500], y.iloc[:500]

    model = build_model(Xs, "log_reg")
    model.fit(Xs, ys)

    proba = model.predict_proba(Xs)[:, 1]
    assert proba.shape[0] == len(Xs)