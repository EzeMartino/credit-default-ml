import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix

REPORTS = Path("reports")


def _metrics_at_threshold(y_true, y_proba, threshold: float):
    y_pred = (y_proba >= threshold).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    flagged_rate = float(np.mean(y_pred))

    return {
        "threshold": float(threshold),
        "precision": float(p),
        "recall": float(r),
        "flagged_rate": flagged_rate,
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }


def find_threshold_subject_to_constraint(
    y_true,
    y_proba,
    *,
    min_recall: float | None = None,
    min_precision: float | None = None,
    step: float = 0.001,
):
    """
    Devuelve el threshold 'óptimo' bajo un constraint.
    - Si min_recall está seteado: elegimos el threshold que MAXIMIZA precision con recall >= min_recall.
      (equivalentemente: el threshold más alto que todavía cumple recall >= target).
    - Si min_precision está seteado: elegimos el threshold que MAXIMIZA recall con precision >= min_precision.
    """
    if (min_recall is None) == (min_precision is None):
        raise ValueError("Seteá exactamente uno: min_recall o min_precision")

    thresholds = np.arange(1.0, -1e-12, -step)  # de 1.0 a 0.0 inclusive

    best = None
    best_key = -1.0

    for t in thresholds:
        m = _metrics_at_threshold(y_true, y_proba, float(t))

        if min_recall is not None:
            # constraint: recall >= target ; objetivo: maximizar precision
            if m["recall"] >= min_recall:
                key = m["precision"]
            else:
                continue

        else:
            # constraint: precision >= target ; objetivo: maximizar recall
            if m["precision"] >= min_precision:
                key = m["recall"]
            else:
                continue

        if key > best_key:
            best_key = key
            best = m

    return best


def main():
    y_true = np.load(REPORTS / "oof_y_true.npy")

    models = {
        "logreg_log_payamt": np.load(REPORTS / "oof_logreg_log_payamt_proba.npy"),
        "random_forest": np.load(REPORTS / "oof_random_forest_proba.npy"),
    }

    out = {}

    recall_targets = [0.60, 0.70]
    precision_targets = [0.50, 0.60]

    for model_name, y_proba in models.items():
        out[model_name] = {"constraints": {}}

        # Constraints de recall
        for mr in recall_targets:
            res = find_threshold_subject_to_constraint(
                y_true, y_proba, min_recall=mr
            )
            out[model_name]["constraints"][f"min_recall_{mr:.2f}"] = res

        # Constraints de precision
        for mp in precision_targets:
            res = find_threshold_subject_to_constraint(
                y_true, y_proba, min_precision=mp
            )
            out[model_name]["constraints"][f"min_precision_{mp:.2f}"] = res

    (REPORTS / "threshold_analysis.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("[OK] Wrote reports/threshold_analysis.json")


if __name__ == "__main__":
    main()