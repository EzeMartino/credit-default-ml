from __future__ import annotations
from pathlib import Path

import time
import uuid
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse


from credit_ml.api.schemas import PredictRequest, PredictResponse, Prediction
from credit_ml.api.deps import get_pipeline, get_metadata
from credit_ml.config import DEFAULT_THRESHOLD

app = FastAPI(title="Credit Default ML API", version="0.1.0")

@app.exception_handler(Exception)
def unhandled_exception_handler(request, exc):
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
    
    pipe = get_pipeline()
    meta = get_metadata()
    
    df = pd.DataFrame(payload.records)

    expected = meta["features_expected"]

   # 1) Missing columns in DF (no record had it)
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=422,
            detail={"msg": "missing required features", "missing": missing_cols, "request_id": request_id},
        )

    # 2) Missing values in row (NaN) in expected columns
    nan_mask = df[expected].isna()

    if nan_mask.any().any():
        bad_rows = nan_mask.any(axis=1)
        row_idx = [int(i) for i, bad in enumerate(bad_rows) if bad]

        missing_by_row = {}
        for i in row_idx[:50]:  # don't explode answer if too many, just show 50
            cols_missing = [col for col in expected if pd.isna(df.at[i, col])]
            missing_by_row[str(i)] = cols_missing

        raise HTTPException(
            status_code=422,
            detail={
                "msg": "missing required features in some records",
                "rows_with_missing": row_idx,
                "missing_by_row": missing_by_row,
                "request_id": request_id,
            },
        )

    # If everything is ok, reorder columns
    df = df[expected]
    
    proba = pipe.predict_proba(df)[:, 1]
    threshold = float(meta.get("threshold", DEFAULT_THRESHOLD))
    labels = (proba >= threshold).astype(int)
    
    preds = [Prediction(proba_default=float(p), label=int(l)) 
             for p, l in zip(proba, labels)
            ]
    
    _ = time.time() - t0 #To log latency if needed
    
    return PredictResponse(
        model_type=meta.get("model_type", "unknown"),
        threshold=threshold,
        predictions=preds,
    )