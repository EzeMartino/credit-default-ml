import json
from pathlib import Path

def test_threshold_analysis_json_schema():
    path = Path("reports/UtilizationPAY_0_v1/threshold_analysis.json")
    assert path.exists()

    data = json.loads(path.read_text(encoding="utf-8"))

    for model_name in ["logreg_log_payamt", "random_forest"]:
        assert model_name in data
        assert "constraints" in data[model_name]

        # check an expected key
        k = "min_recall_0.60"
        assert k in data[model_name]["constraints"]
        res = data[model_name]["constraints"][k]
        assert res is None or all(x in res for x in ["threshold", "precision", "recall", "flagged_rate", "confusion_matrix"])