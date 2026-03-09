from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1)


class Prediction(BaseModel):
    proba_default: float
    label: int


class PredictResponse(BaseModel):
    request_id: str
    model_version: str
    model_type: str
    threshold: float
    predictions: list[Prediction]
    

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class MetaResponse(BaseModel):
    model_type: str
    model_version: str
    dataset_name: str
    training_timestamp_utc: str 
    target_name: str
    threshold: float
    feature_count: int | None = None
    training_rows: int | None = None
    positive_class_rate: float | None = None
    input_features: list[str]
    engineered_features: list[str]
    validation_metrics: dict[str, float] | None = None


class MetricsResponse(BaseModel):
    requests_total: int
    errors_total: int
    avg_latency_ms: float
    model_version: str