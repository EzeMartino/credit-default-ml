import json
from pathlib import Path
import argparse

import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="reports")
    return p.parse_args()

def brier_score(y_true, y_proba):
    brier = brier_score_loss(y_true, y_proba)
    return brier

def calibration_curve_calculation(y_true, y_proba):
    mean_predicted_value, fraction_of_positives = calibration_curve(y_true, y_proba, n_bins=10)
    return mean_predicted_value, fraction_of_positives

def main():
    args = parse_args()
    REPORTS = Path(args.output_dir)
    REPORTS.mkdir(parents=True, exist_ok=True)
    y_true = np.load(REPORTS / "oof_y_true.npy")

    models = {
        "logreg_log_payamt": np.load(REPORTS / "oof_logreg_log_payamt_proba.npy"),
        "random_forest": np.load(REPORTS / "oof_random_forest_proba.npy"),
    }

    out = {}

    # Brier & Calibration curve
    for model_name, y_proba in models.items():
        out[model_name] = {
        "brier score": None,
        "calibration_curve": {
            "mean predicted value": None,
            "fraction of positives": None
        }
    }
        
        brier_res = brier_score(y_true, y_proba)
        
        mean_predicted_value, fraction_of_positives  = calibration_curve_calculation(y_true, y_proba)
        
        out[model_name]["brier score"] = float(brier_res)
        out[model_name]["calibration_curve"]["mean predicted value"] = mean_predicted_value.tolist()
        out[model_name]["calibration_curve"]["fraction of positives"] = fraction_of_positives.tolist()
        
        
    (REPORTS / "calibration_analysis.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(f"[OK] Wrote {REPORTS}\\calibration_analysis.json")


if __name__ == "__main__":
    main()