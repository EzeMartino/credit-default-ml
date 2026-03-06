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


## Project Structure

```
credit-default-ml/
├── data/
├── notebooks/
├── reports/
├── src/
├── tests/
```


## Setup

Create virtual environment:
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.dev.txt
pip install -e .


## Data Profiling Script

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


## Training the Model

#### install
pip install -e .

#### To train the Logistic Regression model with log-transformed features:

python -m credit_ml.modeling.train

This will:

* Load the dataset
* Perform 5-fold stratified cross-validation
* Report mean and standard deviation of ROC-AUC
* Train a holdout model (80/20 split)
* Report holdout ROC-AUC


## Model Comparison Summary

|Model|CV ROC-AUC|CV std|Precision@20%|Recall@20%|
|-|-|-|-|-|
|Logistic (log-transformed)|0.747|0.005|0.55|0.497|
|Random Forest|0.780|0.005|0.565|0.511|

The Random Forest model demonstrates superior ranking performance, suggesting non-linear structure in the problem. However, the Logistic model remains more interpretable and production-ready at this stage.

> For operational constraint-based analysis, see:
> reports/model_comparison_v2.md


## Baseline Comparison (this section corresponds to previous scripts in transition)

To run a baseline comparison between the Logistic Regression, Random Forest and a Dummy model

python -m models.train\_baseline --output_dir reports/

This script:
* Runs 5-fold stratified CV
* Evaluates Dummy, Logistic and Random Forest
* Computes ROC-AUC, PR-AUC and F1
* Stores results in reports/baseline_results.json

### Run threshold analysis 
python -m evaluations.threshold_analysis --output_dir reports/

This script:
* Generates out-of-fold probabilities
* Searches thresholds under operational constraints:
    * Maximize precision subject to Recall ≥ target
    * Maximize recall subject to Precision ≥ target
* Reports precision, recall, flagged_rate and confusion matrix
* Saves results to reports/threshold_analysis.json

### Key Findings
- Logistic ROC-AUC: ~0.747
- Random Forest ROC-AUC: ~0.779
- Under Recall ≥ 0.60 → RF reduces operational workload.
- Under Precision ≥ 0.50 → trade-off depends on FN cost.

### Executive Summary

- Logistic Regression provides strong baseline performance and interpretability.
- Random Forest improves ranking quality and operational efficiency under recall constraints.
- Final model choice depends on regulatory interpretability requirements vs operational cost priorities.


## Feature Engineering Improvements

An explicit interaction feature was introduced:

utilization_x_pay0 = credit_utilization × PAY_0

This interaction models financial stress under delayed payment conditions.

Impact on Logistic Regression:

- ROC-AUC: 0.747 → 0.756
- PR-AUC: 0.507 → 0.516
- Reduced flagged_rate under Recall ≥ 0.60
- Increased recall under Precision ≥ 0.50

This demonstrates the importance of explicit interaction terms for linear models.


## Calibration Analysis

python -m evaluations.calibration_analysis --output_dir reports/

Brier Score:
- Logistic: 0.1425
- Random Forest: 0.1346

Random Forest showed slightly better probability calibration, especially in high-risk regions.

Calibration matters because threshold-based operational policies rely on probability stability.


## Reproducibility

From a clean environment:

1. Install dependencies
2. Run data profiling
3. Run baseline comparison
4. Run threshold analysis
5. Run Calibration analysis
6. Review reports/ folder


## Testing

The project includes a minimal pytest suite to ensure pipeline integrity.

Run tests with:

pytest -q

The test suite validates:

- Data loading and target consistency
- Logistic pipeline training on a small subset
- Stratified cross-validation preserves class distribution
- Threshold analysis output schema

Tests are designed to be lightweight (< 15 seconds) and prevent silent pipeline regressions. All tests must pass before pushing new experimental changes.


## API Inference Service

The project includes a production-style inference API built with FastAPI.

The API loads the trained pipeline artifact and exposes a batch prediction endpoint.

Features:
- Batch inference
- Strict input validation
- Explicit error handling (422 vs 500)
- Structured request logging
- Threshold-based classification

### Run API locally
pip install -r requirements.inference.txt
pip install -e .

python -m uvicorn credit_ml.api.main:app --reload --port 8010

Open API docs:
http://127.0.0.1:8010/docs

Endpoints:
- GET /health
- GET /meta
- GET /metrics
- POST /predict


## Docker Deployment

The API can be executed inside a Docker container.

This ensures the service runs with identical dependencies across environments.


### Build image
docker build -t credit-ml-api:latest .


### Run container
docker run --rm -p 8010:8010 credit-ml-api:latest


## Example Prediction Request

Endpoint:
POST /predict

Example payload:
{
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

Example response:
{
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


## Logging

The API implements structures request logging.

Each requests logs:
- request_id
- number of records
- latency
- HTTP status code

Example log:
predict_request_ok request_id=... n_records=1 latency_ms=42 status=200

Errors are logged with stack traces for debugging.


## Monitoring

GET /metrics

Returns:
- total_requests
- error_requests
- average_latency_ms
- model_version


## Dockerized Architecture

The service includes a minimal runtime dependency set.

Runtime dependencies are separated from development dependencies:
- requirements.inference.txt → runtime environment
- requirements.dev.txt → development tools (pytest, notebooks, etc.)

This reduces container size and avoids OS-specific packages such as pywinpty.


## Quick API Test

#### Using curl:
curl -X POST "http://127.0.0.1:8010/predict" \
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

#### Using Python:
import requests

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
