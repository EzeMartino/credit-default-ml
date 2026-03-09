from __future__ import annotations
import logging
import time
import uuid

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from credit_ml.api.schemas import HealthResponse, MetaResponse, MetricsResponse, PredictRequest, PredictResponse, Prediction
from credit_ml.api.deps import get_pipeline, get_metadata
from credit_ml.modeling.artifacts import get_artifact_paths
from credit_ml.api.validation import validate_and_prepare_df
from credit_ml.config import DEFAULT_THRESHOLD
from credit_ml.api.logging_conf import setup_logging
from credit_ml.api.versioning import compute_model_version

pipe = get_pipeline()
meta = get_metadata()

PIPELINE_PATH, METADATA_PATH = get_artifact_paths()
MODEL_VERSION = meta.get("model_version") or compute_model_version(PIPELINE_PATH, METADATA_PATH)
MAX_RECORDS = 1000
MAX_INFERENCE_TIME_MS = 2000

EXCLUDE_PATHS = {"/health", "/meta", "/metrics", "/docs", "/redoc", "/openapi.json"}
REQUEST_COUNT = 0
ERROR_COUNT = 0
TOTAL_LATENCY = 0

setup_logging()  # Configure logging at the start of the application
logger = logging.getLogger("credit_ml.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- startup ----
    try:
        expected = meta.get("input_features", [])

        if expected:
            dummy = {c: 0 for c in expected}
            df = pd.DataFrame([dummy])
            _ = pipe.predict_proba(df)

        logger.info("warmup_ok")

    except Exception as e:
        logger.exception(f"warmup_failed error_type={type(e).__name__}")

    yield

    # ---- shutdown ----
    logger.info("api_shutdown")
    
app = FastAPI(title="Credit Default ML API", version="0.1.0", lifespan=lifespan)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    global REQUEST_COUNT, ERROR_COUNT, TOTAL_LATENCY
    
    # Ignorar preflight y endpoints auxiliares
    if request.method == "OPTIONS" or request.url.path in EXCLUDE_PATHS:
        return await call_next(request)

    t0 = time.time()
    REQUEST_COUNT += 1

    try:
        response = await call_next(request)
        if response.status_code >= 400:
            ERROR_COUNT += 1
        return response
    except Exception:
        ERROR_COUNT += 1
        raise
    finally:
        TOTAL_LATENCY += int((time.time() - t0) * 1000)


@app.exception_handler(Exception)
def unhandled_exception_handler(request, exc):
    logger.exception(
        f"unhandled_exception path={request.url.path} "
        f"error_type={type(exc).__name__}"
    )
     
    return JSONResponse(
        status_code=500,
        content={"msg": "internal_server_error", "error_type": type(exc).__name__},
    )

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=MODEL_VERSION
    )

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    
    request_id = str(uuid.uuid4())
    t0 = time.time()
    n_records = len(payload.records)
        
    logger.info(
        f"predict_request_received request_id={request_id} n_records={n_records} model_version={MODEL_VERSION}"
    )
    
    if n_records > MAX_RECORDS:
        latency_ms = int((time.time() - t0) * 1000)
         
        logger.warning(
            f"predict_request_too_large request_id={request_id} "
            f"n_records={n_records} latency_ms={latency_ms} status=422 "
            f"model_version={MODEL_VERSION}"
        )

        raise HTTPException(
            status_code=422,
            detail={
                "msg": "batch size exceeds limit",
                "max_records": MAX_RECORDS,
                "received": n_records,
                "request_id": request_id,
                "model_version": MODEL_VERSION,
            },
        )
    
    try:
        df = pd.DataFrame(payload.records)
        expected = meta.get("input_features", [])

        df = validate_and_prepare_df(df=df, expected=expected, request_id=request_id)
        proba = pipe.predict_proba(df)[:, 1]
        
        threshold = float(meta.get("threshold", DEFAULT_THRESHOLD))
        labels = (proba >= threshold).astype(int)
        
        preds = [Prediction(proba_default=float(proba), label=int(lab)) 
                for proba, lab in zip(proba, labels)
                ]
        
        latency_ms = int((time.time() - t0) * 1000)

        logger.info(
            f"predict_request_ok request_id={request_id} "
            f"n_records={n_records} latency_ms={latency_ms} status=200 "
            f"model_version={MODEL_VERSION}"
        )

        if latency_ms > MAX_INFERENCE_TIME_MS:
            raise HTTPException(
                status_code=503,
                detail={
                    "msg": "inference_timeout",
                    "latency_ms": latency_ms,
                    "request_id": request_id,
                    "model_version": MODEL_VERSION
                }
            )
        
        return PredictResponse(
            request_id=request_id,
            model_version= MODEL_VERSION,
            model_type=meta.get("model_type", "unknown"),
            threshold=threshold,
            predictions=preds,
        )
    except HTTPException as e:
        latency_ms = int((time.time() - t0) * 1000)
        
        logger.warning(
            f"predict_request_invalid request_id={request_id} "
            f"n_records={n_records} latency_ms={latency_ms} " 
            f"status={e.status_code} "
            f"model_version={MODEL_VERSION}"
        )
        raise e
    
@app.get("/meta", response_model=MetaResponse)
def meta_info():
    return MetaResponse(
        model_type=meta.get("model_type"),
        model_version=MODEL_VERSION,
        training_timestamp_utc=meta.get("training_timestamp_utc"),
        dataset_name=meta.get("dataset_name"),
        target_name=meta.get("target_name"),
        threshold=meta.get("threshold"),
        feature_count=meta.get("feature_count"),
        training_rows=meta.get("training_rows"),
        positive_class_rate=meta.get("positive_class_rate"),
        input_features=meta.get("input_features", []),
        engineered_features=meta.get("engineered_features", []),
        validation_metrics=meta.get("validation_metrics", {}),
    )

@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    avg_latency = (
        TOTAL_LATENCY / REQUEST_COUNT
        if REQUEST_COUNT > 0
        else 0
    )

    return MetricsResponse(
        requests_total=REQUEST_COUNT,
        errors_total=ERROR_COUNT,
        avg_latency_ms=avg_latency,
        model_version=MODEL_VERSION,
    )
