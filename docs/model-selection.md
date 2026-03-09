# Model Selection

## Baseline Comparison (this section corresponds to previous scripts in transition)

To run a baseline comparison between the Logistic Regression, Random Forest and a Dummy model

python -m models.train\_baseline --output_dir reports/

This script:
* Runs 5-fold stratified CV
* Evaluates Dummy, Logistic and Random Forest
* Computes ROC-AUC, PR-AUC and F1
* Stores results in reports/baseline_results.json

### Run threshold analysis 
python -m evaluations.threshold_analysis --output_dir reports/

This script:
* Generates out-of-fold probabilities
* Searches thresholds under operational constraints:
    * Maximize precision subject to Recall ≥ target
    * Maximize recall subject to Precision ≥ target
* Reports precision, recall, flagged_rate and confusion matrix
* Saves results to reports/threshold_analysis.json

### Key Findings
- Logistic ROC-AUC: ~0.747
- Random Forest ROC-AUC: ~0.779
- Under Recall ≥ 0.60 → RF reduces operational workload.
- Under Precision ≥ 0.50 → trade-off depends on FN cost.

### Executive Summary

- Logistic Regression provides strong baseline performance and interpretability.
- Random Forest improves ranking quality and operational efficiency under recall constraints.
- Final model choice depends on regulatory interpretability requirements vs operational cost priorities.


## Model Comparison Summary

|Model|CV ROC-AUC|CV std|Precision@20%|Recall@20%|
|-|-|-|-|-|
|Logistic (log-transformed)|0.747|0.005|0.55|0.497|
|Random Forest|0.780|0.005|0.565|0.511|

The Random Forest model demonstrates superior ranking performance, suggesting non-linear structure in the problem. However, the Logistic model remains more interpretable and production-ready at this stage.

> For operational constraint-based analysis, see:
> reports/model_comparison_v2.md


## Feature Engineering

An explicit interaction feature was introduced:

utilization_x_pay0 = credit_utilization × PAY_0

This interaction models financial stress under delayed payment conditions.

Impact on Logistic Regression:
- ROC-AUC: 0.747 → 0.756
- PR-AUC: 0.507 → 0.516
- Reduced flagged_rate under Recall ≥ 0.60
- Increased recall under Precision ≥ 0.50

This demonstrates the importance of explicit interaction terms for linear models.


## Calibration Analysis

Calibration was evaluated using the Brier Score and calibration curves.

Command used during experimentation:

python -m evaluations.calibration_analysis --output_dir reports/

This analysis:

- Generates predicted probabilities using cross-validation
- Computes Brier Score
- Plots calibration curves comparing predicted vs observed probabilities

### Results

Brier Score:

- Logistic Regression: 0.1425
- Random Forest: 0.1346

Random Forest showed slightly better probability calibration, especially in higher-risk regions.

### Interpretation

Calibration quality is important because operational policies rely on probability thresholds to trigger manual review.

Even though Random Forest showed slightly better calibration, the difference was not large enough to offset the interpretability and operational advantages of Logistic Regression.


## Experiment Reproduction
From a clean environment:

1. Install dependencies
2. Run data profiling
3. Run baseline comparison
4. Run threshold analysis
5. Run Calibration analysis
6. Review reports/ folder