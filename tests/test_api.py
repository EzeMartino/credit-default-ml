from fastapi.testclient import TestClient
from credit_ml.api.main import app

client = TestClient(app)

def get_valid_payload():
    return {
        "records": [
            {   
                "LIMIT_BAL": 20000,
                "SEX": 2,
                "EDUCATION": 2,
                "MARRIAGE": 1,
                "AGE": 24,
                "PAY_0": 2,
                "PAY_2": 2,
                "PAY_3": -1,
                "PAY_4": -1,
                "PAY_5": -2,
                "PAY_6": -2,
                "BILL_AMT1": 3913,
                "BILL_AMT2": 3102,
                "BILL_AMT3": 689,
                "BILL_AMT4": 0,
                "BILL_AMT5": 0,
                "BILL_AMT6": 0,
                "PAY_AMT1": 0,
                "PAY_AMT2": 689,
                "PAY_AMT3": 0,
                "PAY_AMT4": 0,
                "PAY_AMT5": 0,
                "PAY_AMT6": 0,
            }
        ]
    }
    
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    
def test_predict_success():
    payload = get_valid_payload()
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    assert "proba_default" in data["predictions"][0]
    assert "label" in data["predictions"][0]
    
def test_predict_missing_feature_returns_422():
    payload = get_valid_payload()
    del payload["records"][0]["AGE"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    
def test_batch_partial_missing_returns_422():
    payload = get_valid_payload()
    
    second = payload["records"][0].copy()
    del second["AGE"]
    
    payload["records"].append(second)
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 422
    
def test_predict_ignores_extra_columns():
    payload = get_valid_payload()
    payload["records"][0]["EXTRA_FIELD"] = 123

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
def test_predict_null_value_returns_422():
    payload = get_valid_payload()
    payload["records"][0]["AGE"] = None

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    
def test_predict_missing_column_all_records_returns_422():
    payload = get_valid_payload()
    del payload["records"][0]["AGE"]

    # we make sure that there is not another record that includes AGE
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["msg"] == "missing required features"
    assert "AGE" in detail["missing"]
    
def test_predict_missing_by_row_returns_422():
    payload = get_valid_payload()
    second = payload["records"][0].copy()
    del second["AGE"]
    payload["records"].append(second)

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["msg"] == "missing required features in some records"
    assert 1 in detail["rows_with_missing"]
    assert "AGE" in detail["missing_by_row"]["1"]
    
def test_predict_batch_too_large_returns_422():
    payload = get_valid_payload()

    payload["records"] = payload["records"] * 2000

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    
def test_meta_endpoint():
    r = client.get("/meta")
    assert r.status_code == 200
    data = r.json()
    assert "model_type" in data
    assert "threshold" in data
    assert "features_expected" in data
    
def test_meta_includes_model_version():
    r = client.get("/meta")
    assert r.status_code == 200
    data = r.json()
    assert "model_version" in data
    assert isinstance(data["model_version"], str)
    assert len(data["model_version"]) >= 8