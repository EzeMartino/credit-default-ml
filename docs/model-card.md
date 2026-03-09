# Model Card — Credit Default Logistic Regression


## Model Overview

This model predicts the probability that a credit card client will default on the next payment cycle.

The currently deployed model is a Logistic Regression pipeline served through a FastAPI batch inference API.


## Intended Use

Intended for batch risk scoring and threshold-based operational review workflows.

Not intended for fully automated high-stakes approval/denial decisions without human review.


## Training Data

The model was trained on the Taiwan credit card default dataset (30,000 rows, 25 columns, ~22.1% positive class rate).


## Features

- credit_utilization
- utilization_x_pay0


## Metrics

- ROC-AUC: `0.7482154292379494`
- PR-AUC: `0.5063888746525542`
- Threshold: `0.23`
- Operational criteria: Recall >= 60%


## Threshold Policy

The deployed threshold is `0.23`.

It was selected through validation-based threshold analysis under explicit operational constraints, balancing recall and review workload.


## Limitations

- The deployed model prioritizes interpretability over maximum ranking performance.
- Threshold selection depends on operational assumptions and may need recalibration under new data distributions.
- The current API is inference-only and does not include automated retraining.
- The model was trained on a historical dataset from a specific domain and geography.
- It should not be used outside similar credit risk scoring contexts without revalidation.


## Operational Risks

- Distribution drift
- Threshold obsolescence
- Calibration degradation
- Regulatory changes or business policy changes


## Monitoring Considerations

- Monitor positive rate
- Score distribution
- Flagged rate
- Performance degradation
- Threshold stability

