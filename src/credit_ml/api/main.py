from __future__ import annotations
from pathlib import Path
import logging
import time
import uuid

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from credit_ml.api.schemas import PredictRequest, PredictResponse, Prediction
from credit_ml.api.deps import get_pipeline, get_metadata
from credit_ml.api.validation import validate_and_prepare_df
from credit_ml.config import DEFAULT_THRESHOLD
from credit_ml.api.logging_conf import setup_logging

setup_logging()  # Configure logging at the start of the application
logger = logging.getLogger("credit_ml.api")

app = FastAPI(title="Credit Default ML API", version="0.1.0")

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

@app.get("/health")
def health():
    return {"status": "ok", "loaded_from": str(Path(__file__).resolve())}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    
    request_id = str(uuid.uuid4())
    t0 = time.time()
    n_records = len(payload.records)

    logger.info(
        f"predict_request_received request_id={request_id} n_records={n_records}"
    )
    
    pipe = get_pipeline()
    meta = get_metadata()
    
    try:
        df = pd.DataFrame(payload.records)
        expected = meta.get("features_expected", [])

        df = validate_and_prepare_df(df=df, expected=expected, request_id=request_id)
        proba = pipe.predict_proba(df)[:, 1]
        
        threshold = float(meta.get("threshold", DEFAULT_THRESHOLD))
        labels = (proba >= threshold).astype(int)
        
        preds = [Prediction(proba_default=float(p), label=int(l)) 
                for p, l in zip(proba, labels)
                ]
        
        latency_ms = int((time.time() - t0) * 1000)

        logger.info(
            f"predict_request_ok request_id={request_id} "
            f"n_records={n_records} latency_ms={latency_ms} status=200"
        )
        
        return PredictResponse(
            model_type=meta.get("model_type", "unknown"),
            threshold=threshold,
            predictions=preds,
        )
    except HTTPException as e:
        latency_ms = int((time.time() - t0) * 1000)
        
        logger.warning(
            f"predict_request_invalid request_id={request_id} "
            f"n_records={n_records} latency_ms={latency_ms}" 
            f"status={e.status_code}"
        )
        raise e