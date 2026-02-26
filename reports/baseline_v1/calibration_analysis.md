# Calibration Analysis

## Brier Score
- Logistic: 0.14250123439842702
- Random Forest: 0.13463498643805738

Lower Brier score indicates better probabilistic calibration.
Random Forest shows slightly better overall calibration.

## Observations
Logistic Regression tends to:
* Slightly overestimate risk in lower probability bins.
* Underestimate risk in higher probability bins.
* Produce more conservative probability estimates.

Random Forest:
* Displays closer alignment between predicted probabilities and observed default rates.
* Shows stronger calibration in high-risk regions.
* Provides more reliable probabilities in the operational decision range.

## Implications
Since threshold-based policies depend on probability stability, better calibration improves:
* Threshold robustness
* Policy consistency
* Operational predictability
Random Forest not only improves ranking performance but also demonstrates competitive probabilistic calibration, strengthening its case under operational constraints.