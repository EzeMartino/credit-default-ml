# Model Comparison

This project evaluated three model families for credit default prediction.

| Model | ROC-AUC | PR-AUC | Brier Score | Notes |
|------|------|------|------|------|
| Logistic Regression | 0.756 | 0.516 | 0.142 | Interpretable linear baseline |
| Random Forest | 0.780 | ~0.53 | ~0.134 | Captures non-linear structure |
| Torch MLP | 0.787 | 0.552 | 0.134 | Best ranking performance |

## Interpretation

The Torch MLP achieved the best ranking performance while maintaining calibration comparable to Random Forest.

However, Logistic Regression remains the deployed model due to:

- interpretability
- regulatory explainability
- simpler deployment footprint

This reflects a common production trade-off between model complexity and operational constraints.