# Technical Decision Log

This document explains the main engineering and modeling decisions made during the development of this project.

The goal is to make the reasoning behind design choices explicit and reproducible.

---

## 1. Why Logistic Regression instead of Random Forest

Although Random Forest achieved slightly higher performance in some experiments, Logistic Regression was chosen for deployment.

Reasons:
- **Interpretability**: Logistic Regression coefficients can be inspected and explained to stakeholders.
- **Regulatory contexts**: Credit risk systems often require explainable models.
- **Operational simplicity**: Logistic Regression is lightweight and easier to maintain in production.
- **Infrastructure cost**: The model is cheaper to run and scales easily.

Given similar ranking performance, the simpler and more interpretable model was preferred.

---

## 2. Why a Batch-Only Prediction API

The API was designed as a **batch prediction service** rather than a single-record endpoint.

Reasons:
- Credit risk evaluation is usually performed **in batches of clients**
- Batch prediction reduces overhead per request
- Allows efficient vectorized inference
- Simplifies infrastructure design

The endpoint still supports a single record, but its main use case is batch scoring.

---

## 3. Why Strict Validation Instead of Silent Imputation

The API performs strict validation of input features.

If required features are missing or contain null values, the request returns **HTTP 422**.

Reasons:
- Prevent predictions based on incomplete or incorrect data
- Avoid silent data quality issues
- Force upstream systems to provide correct inputs
- Maintain prediction reliability

Silent imputation may hide upstream errors and degrade model performance.

---

## 4. Why Limit Batch Size

The API enforces a maximum batch size.

MAX_RECORDS = 1000

Reasons:
- Prevent excessive memory usage
- Protect the API from accidental overload
- Ensure predictable latency
- Maintain service stability

Large workloads should be split into multiple requests.

---

## 5. Why Model Versioning via Artifact Hash

The deployed model version is computed using a **SHA256 hash** of the artifacts.

Artifacts included in the version hash:
- pipeline.joblib
- metadata.json

Reasons:
- Guarantees deterministic model versioning
- Detects any change in the trained artifacts
- Enables traceability between predictions and the exact model used

This allows reproducible inference and easier debugging.

---

## 6. Why Feature Engineering for Linear Models

Logistic Regression is a **linear model**, meaning it cannot naturally capture nonlinear relationships.

To address this limitation, engineered features were added:
- **credit_utilization**
- **credit_utilization × payment delay interaction**

These features allow the model to capture interactions that would otherwise be missed by a purely linear formulation.

---

## 7. Why Separate Runtime and Development Dependencies

Two dependency files are used:
- requirements.inference.txt
- requirements.dev.txt

Reasons:
- Keep the production environment minimal
- Reduce Docker image size
- Avoid installing unnecessary development tools in production
- Improve build reproducibility

---

## 8. Why API-Level Observability

The API includes basic observability features:
- request tracing via `request_id`
- structured logging
- `/metrics` endpoint
- latency tracking
- error counting

These features help diagnose issues in production and monitor service health.

---

## 9. Why Dockerized Deployment

The service is deployed using Docker.

Reasons:
- Environment reproducibility
- Simplified deployment
- Infrastructure portability
- Isolation of dependencies

This allows the service to run consistently across environments.

---

## Summary

This project was designed not only as a machine learning model but as a **small production-ready ML service**.

Key engineering goals:
- reproducibility
- observability
- maintainability
- operational simplicity
