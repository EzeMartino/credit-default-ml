# Credit Default ML
![Tests](https://github.com/EzeMartino/credit-default-ml/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Docker](https://img.shields.io/badge/docker-ready-blue)

End-to-end Machine Learning project including:
- Feature engineering
- Model evaluation and calibration
- Threshold optimization
- Production-style FastAPI inference service
- Automated testing
- Dockerized deployment


## System Components

### Training Layer
- Data profiling
- Feature engineering
- Cross-validation
- Threshold optimization
- Calibration analysis

### Model Artifacts
- Serialized pipeline (pipeline.joblib)
- Metadata with threshold and model info

### Inference Layer
- FastAPI batch prediction endpoint
- Input validation
- Logging and error handling

### Deployment
- Docker container
- Minimal runtime dependencies


## Problem

Binary classification task:
Predict whether a client will default on their credit card payment next month.

Target:
`default_payment_next_month`

* 1 = default
* 0 = no default


## Dataset

Taiwan credit card clients dataset.

Shape:

* 30,000 rows
* 25 columns

No missing values detected.

Target rate:

* Positive class (default): 22.12%
* Majority class baseline accuracy: 0.7788


## Architecture Overview

Pipeline flow:
```
Raw Data
   ↓
Feature Engineering
   ↓
Model Training (Logistic Regression)
   ↓
Artifact Export (pipeline.joblib + metadata.json)
   ↓
FastAPI Service
   ↓
Batch Predictions
```

![alt text](docs/Architecture%20Overview-mermaid.png)


## Project Structure

```
credit-default-ml/
├── data/
│   └── raw/
├── docs/
├── notebooks/
├── reports/
├── src/
│   └── credit_ml/
│       ├── api/
│       ├── training/
│       ├── modeling/
│       └── ...
├── tests/
├── Dockerfile
├── pyproject.toml
├── requirements.dev.txt
└── requirements.inference.txt
```


## Evaluation Philosophy

Primary (ranking) metric:
- ROC-AUC

Operational constraint:
- Threshold-based policy 
- Either Recall ≥ X
- Or Precision ≥ X

Performance is never discussed without:
- ROC-AUC (ranking quality)
- Operational metric under explicit threshold


## Setup

python -m venv venv
venv\Scripts\activate
pip install -r requirements.dev.txt
pip install -e .


## Quick Start

python -m credit_ml.training.train
uvicorn src.credit_ml.api.main:app --reload --port 8010
Swagger docs: `http://127.0.0.1:8010/docs`

Optional developer shortcuts using Makefile:

make setup
make train
make serve
make test

## Training the Model

#### Install
pip install -e .

#### To train the Logistic Regression model with log-transformed features:

python -m credit_ml.modeling.train

This will:

* Load the dataset
* Perform 5-fold stratified cross-validation
* Report mean and standard deviation of ROC-AUC
* Train a holdout model (80/20 split)
* Report holdout ROC-AUC


## Model Summary

Random Forest achieved stronger ranking performance, while Logistic Regression offered better interpretability and lower deployment complexity.

The deployed API currently uses Logistic Regression.

**Current production threshold:** `0.23`  
Selected through threshold analysis to satisfy the chosen operational precision/recall constraint on validation data.

See `docs/model-selection.md` for the full comparison.

See `docs/model-card.md` for detailed model documentation.

### Feature Engineering

The project includes engineered features such as `credit_utilization` and an interaction term `utilization_x_pay0`.

These additions improved Logistic Regression performance and reduced operational workload under selected threshold constraints.

See `docs/model-evaluation.md` for detailed results.


### Calibration Analysis

Calibration was evaluated to ensure probability stability under threshold-based decision rules.

Brier Score:
- Logistic Regression: 0.1425
- Random Forest: 0.1346

Random Forest showed slightly better calibration, but the difference was not enough to outweigh Logistic Regression’s interpretability and deployment simplicity.

See `docs/model-evaluation.md` for full details.


### Reproducibility

From a clean environment:

1. Install dependencies
2. Train the model
3. Run tests
4. Start the API
5. Send a batch prediction request


## API Inference Service

The project includes a production-style inference API built with FastAPI.

The API loads the trained pipeline artifact and exposes a batch prediction endpoint.

Features:
- Batch inference
- Strict input validation
- Explicit error handling (422 vs 500)
- Structured request logging
- Threshold-based classification using the threshold stored in model metadata

### Run API locally
pip install -r requirements.dev.txt
pip install -e .
python -m credit_ml.modeling.train
pytest -q
uvicorn src.credit_ml.api.main:app --reload

Open API docs:
http://127.0.0.1:8010/docs

Endpoints:
- GET /health
- GET /meta
- GET /metrics
- POST /predict


### Quick API Test

#### Using curl:
```curl -X POST "http://127.0.0.1:8010/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
        "PAY_AMT6": 0
      }
    ]
  }'
```

#### Using Python:
```import requests

payload = {
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
            "PAY_AMT6": 0
        }
    ]
}

response = requests.post("http://127.0.0.1:8010/predict", json=payload)
print(response.status_code)
print(response.json())
```

#### Expected Response
{
  "request_id": "example-request-id",
  "model_version": "cee7073b7535",
  "model_type": "logreg",
  "threshold": 0.23,
  "predictions": [
    {
      "proba_default": 0.59,
      "label": 1
    }
  ]
}


## API Validation Rules

The API supports batch predictions with a maximum of 1000 records per request.
This prevents resource exhaustion and improves API stability.

Input validation is applied before model inference.

Rules:
- Missing required features → 422 error
- Missing values inside records → 422 error
- Extra columns → ignored
- Batch requests must have valid data in all records
This prevents silent inference failures and enforces a strict API contract.


### Logging

The API implements structured request logging.

Each request logs:
- request_id
- number of records
- latency
- HTTP status code

Example log:
predict_request_ok request_id=... n_records=1 latency_ms=42 status=200

Errors are logged with stack traces for debugging.


### Monitoring

GET /metrics

Returns:
- total_requests
- error_requests
- average_latency_ms
- model_version


## Docker Deployment

The API can be executed inside a Docker container.

This ensures the service runs with identical dependencies across environments.


### Build image
docker build -t credit-ml-api:latest .


### Run container
docker run --rm -p 8010:8010 credit-ml-api:latest


### Dockerized Architecture

The service includes a minimal runtime dependency set.

Runtime dependencies are separated from development dependencies:
- requirements.inference.txt → runtime environment
- requirements.dev.txt → development tools (pytest, notebooks, etc.)

This reduces container size and avoids OS-specific packages such as pywinpty.


## Testing

The project includes a pytest suite covering training, metadata, API behavior and validation rules.

Run tests with:

pytest -q

The test suite validates:

- Data loading and target consistency
- Logistic pipeline training on a small subset
- Stratified cross-validation preserves class distribution
- Threshold analysis output schema

Tests are designed to be lightweight (< 15 seconds) and prevent silent pipeline regressions. All tests must pass before pushing new experimental changes.


## Continuous Integration

The repository includes a GitHub Actions pipeline that:

1. Installs dependencies
2. Trains the model
3. Runs the full test suite

This ensures that model artifacts and the API remain reproducible and stable.


## Documentation & Utilities

Generate automated dataset report:
python src/data/load\_and\_profile.py
--input data/raw/credit\_default.xls
--out reports/profile\_summary.json

This script validates:
* column structure
* target distribution
* baseline trivial model
* skewness statistics
* PAY\_X unique values

Additional technical details are available in:
- `docs/decision-log.md`
- `docs/model-selection.md`


## Limitations

- The deployed model prioritizes interpretability over maximum ranking performance.
- Threshold selection depends on operational assumptions and may need recalibration under new data distributions.
- The current API is inference-only and does not include automated retraining.


## Neural Network Experiment

A small MLP model was implemented using PyTorch to explore non-linear modeling.

Architecture:
Input → 64 → 32 → 1

Improvements tested:
- Batch Normalization
- Dropout
- BCEWithLogitsLoss

Final validation ROC-AUC: ~0.77

Despite modeling non-linear interactions, tree-based models still outperform neural networks on this tabular dataset.


## Next Steps

- Add model drift monitoring
- Introduce retraining workflow orchestration
- Expand evaluation reports for calibration and threshold sensitivity
- Compare deployed Logistic Regression against a more production-constrained Random Forest serving setup