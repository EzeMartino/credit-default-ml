def test_load_data_basic():
    from src.credit_ml.models.train_logreg import load_data, TARGET

    X, y = load_data("data/raw/credit_default.xls")
    assert len(X) == len(y)
    assert y.name == TARGET
    assert set(y.unique()).issubset({0, 1})