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
    model_version: str
    model_type: str
    trained_at: str | None = None
    threshold: float
    features_expected: list[str]
    features_engineered: list[str]


class MetricsResponse(BaseModel):
    requests_total: int
    errors_total: int
    avg_latency_ms: float
    model_version: str