# src/data/load_and_profile.py
"""
Load the UCI Credit Card Default dataset and generate a lightweight profiling report.

Usage:
  python -m data.load_and_profile --input data/raw/credit_default.csv --out reports/profile_summary.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd


TARGET_COL = "default_payment_next_month"
ID_COL = "ID"

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]


@dataclass
class ProfileSummary:
    input_path: str
    n_rows: int
    n_cols: int
    columns: List[str]
    dtypes: Dict[str, str]
    missing_total: int
    missing_by_col: Dict[str, int]

    target_col: str
    target_rate_pos: float
    target_counts: Dict[str, int]
    baseline_majority_class: int
    baseline_majority_accuracy: float

    pay_unique_values: Dict[str, List[int]]

    skew_by_col: Dict[str, float]
    skew_bill_amt: Dict[str, float]
    skew_pay_amt: Dict[str, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load dataset and write a simple profiling report (JSON).")
    p.add_argument("--input", required=True, help="Path to raw CSV (e.g., data/raw/credit_default.xls)")
    p.add_argument("--out", required=True, help="Path to output JSON (e.g., reports/profile_summary.json)")
    return p.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    missing = []
    for col in [TARGET_COL, ID_COL, *PAY_COLS]:
        if col not in df.columns:
            missing.append(col)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Found columns: {list(df.columns)}")


def compute_target_summary(y: pd.Series) -> tuple[float, Dict[str, int], int, float]:
    counts = y.value_counts(dropna=False).to_dict()
    # Ensure keys are JSON-serializable strings
    counts_str = {str(k): int(v) for k, v in counts.items()}

    # assume binary 0/1
    pos_rate = float((y == 1).mean())

    majority_class = int(y.value_counts().idxmax())
    majority_acc = float((y == majority_class).mean())

    return pos_rate, counts_str, majority_class, majority_acc


def skew_dict(df: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    # Use pandas skew (Fisher-Pearson). Cast to float for JSON.
    s = df[cols].skew(numeric_only=True)
    return {c: float(s[c]) for c in s.index}


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.out)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(in_path, header=1)
    df.columns = (
    df.columns
      .str.strip()
      .str.replace(".", "", regex=False)
      .str.replace(" ", "_")
)
    validate_columns(df)

    missing_by_col = df.isnull().sum().to_dict()
    missing_by_col_int = {str(k): int(v) for k, v in missing_by_col.items()}
    missing_total = int(df.isnull().sum().sum())

    y = df[TARGET_COL]
    target_rate_pos, target_counts, majority_class, majority_acc = compute_target_summary(y)

    # PAY_X unique values (sorted)
    pay_uniques: Dict[str, List[int]] = {}
    for col in PAY_COLS:
        uniques = sorted(pd.Series(df[col].unique()).dropna().astype(int).tolist())
        pay_uniques[col] = uniques

    # Skew: for specific groups + global numeric skew
    bill_cols = [c for c in df.columns if c.startswith("BILL_AMT")]
    pay_amt_cols = [c for c in df.columns if c.startswith("PAY_AMT")]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    summary = ProfileSummary(
        input_path=str(in_path),
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        columns=[str(c) for c in df.columns.tolist()],
        dtypes={str(k): str(v) for k, v in df.dtypes.astype(str).to_dict().items()},
        missing_total=missing_total,
        missing_by_col=missing_by_col_int,
        target_col=TARGET_COL,
        target_rate_pos=float(target_rate_pos),
        target_counts=target_counts,
        baseline_majority_class=int(majority_class),
        baseline_majority_accuracy=float(majority_acc),
        pay_unique_values=pay_uniques,
        skew_by_col=skew_dict(df, numeric_cols),
        skew_bill_amt=skew_dict(df, bill_cols),
        skew_pay_amt=skew_dict(df, pay_amt_cols),
    )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote profile summary to: {out_path}")


if __name__ == "__main__":
    main()
