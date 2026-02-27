# Credit Default ML


## Problem

Binary classification task:
Predict whether a client will default on their credit card payment next month.

Target:
`default_payment_next_month`

* 1 = default
* 0 = no default


## Dataset

Taiwan credit card clients dataset.

Shape:

* 30,000 rows
* 25 columns

No missing values detected.

Target rate:

* Positive class (default): 22.12%
* Majority class baseline accuracy: 0.7788


## Evaluation Philosophy

Primary (ranking) metric:
- ROC-AUC

Operational constraint:
- Threshold-based policy
- Either Recall ≥ X
- Or Precision ≥ X

Performance is never discussed without:
- ROC-AUC (ranking quality)
- Operational metric under explicit threshold


## Project Structure

```
credit-default-ml/
├── data/
├── notebooks/
├── reports/
├── src/
├── tests/
```


## Setup

Create virtual environment:
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt


## Data Profiling Script

Generate automated dataset report:
python src/data/load\_and\_profile.py
--input data/raw/credit\_default.xls
--out reports/profile\_summary.json

This script validates:

* column structure
* target distribution
* baseline trivial model
* skewness statistics
* PAY\_X unique values


## Training the Model

To train the Logistic Regression model with log-transformed features:

python -m src.models.train\_logreg --input data/raw/credit\_default.xls

This will:

* Load the dataset
* Perform 5-fold stratified cross-validation
* Report mean and standard deviation of ROC-AUC
* Train a holdout model (80/20 split)
* Report holdout ROC-AUC


## Model Comparison Summary

|Model|CV ROC-AUC|CV std|Precision@20%|Recall@20%|
|-|-|-|-|-|
|Logistic (log-transformed)|0.747|0.005|0.55|0.497|
|Random Forest|0.780|0.005|0.565|0.511|

The Random Forest model demonstrates superior ranking performance, suggesting non-linear structure in the problem. However, the Logistic model remains more interpretable and production-ready at this stage.

> For operational constraint-based analysis, see:
> reports/model_comparison_v2.md


## Baseline Comparison

To run a baseline comparison between the Logistic Regression, Random Forest and a Dummy model

python -m src.models.train\_baseline --output_dir reports/

This script:
* Runs 5-fold stratified CV
* Evaluates Dummy, Logistic and Random Forest
* Computes ROC-AUC, PR-AUC and F1
* Stores results in reports/baseline_results.json

### Run threshold analysis
python -m src.evaluations.threshold_analysis --output_dir reports/

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

## Executive Summary

- Logistic Regression provides strong baseline performance and interpretability.
- Random Forest improves ranking quality and operational efficiency under recall constraints.
- Final model choice depends on regulatory interpretability requirements vs operational cost priorities.

## Feature Engineering Improvements

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
python -m src.evaluations.calibration_analysis --output_dir reports/

Brier Score:
- Logistic: 0.1425
- Random Forest: 0.1346

Random Forest showed slightly better probability calibration, especially in high-risk regions.

Calibration matters because threshold-based operational policies rely on probability stability.

## Reproducibility

From a clean environment:

1. Install dependencies
2. Run data profiling
3. Run baseline comparison
4. Run threshold analysis
5. Run Calibration analysis
6. Review reports/ folder

## Testing

The project includes a minimal pytest suite to ensure pipeline integrity.

Run tests with:

pytest -q

The test suite validates:

- Data loading and target consistency
- Logistic pipeline training on a small subset
- Stratified cross-validation preserves class distribution
- Threshold analysis output schema

Tests are designed to be lightweight (< 15 seconds) and prevent silent pipeline regressions. All tests must pass before pushing new experimental changes.