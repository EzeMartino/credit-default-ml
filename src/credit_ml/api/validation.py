from __future__ import annotations

import pandas as pd
from fastapi import HTTPException

def validate_and_prepare_df(
    df: pd.DataFrame,
    expected: list[str],
    request_id: str,
) -> pd.DataFrame:
    if not isinstance(expected, list) or not expected:
        raise HTTPException(status_code=500, detail="metadata missing features_expected")
    
    
    # 1) Missing columns in DF (no record had it)
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=422,
            detail={"msg": "missing required features", 
                    "missing": missing_cols, 
                    "request_id": request_id
                    },
        )
    
    # 2) Reorder and drop extras (only mantain expected)
    df2 = df[expected]
    
    # 3) Missing values in row (NaN) in expected columns
    nan_mask = df2[expected].isna()

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
        
    return df2