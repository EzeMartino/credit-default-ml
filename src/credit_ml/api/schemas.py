from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1)


class Prediction(BaseModel):
    proba_default: float
    label: int


class PredictResponse(BaseModel):
    model_type: str
    threshold: float
    predictions: list[Prediction]