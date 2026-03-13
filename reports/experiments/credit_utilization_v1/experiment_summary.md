# Experiment: Credit Utilization Feature

## Change Introduced

Added new feature:

credit_utilization = BILL_AMT1 / LIMIT_BAL

Clipped to avoid extreme outliers.

Included in numeric_no_log pipeline.

####

## Motivation

Credit utilization is a well-known risk indicator in credit scoring.
High balance usage relative to credit limit may indicate financial stress.

####

## Results

### Logistic Regression
- ROC-AUC: 0.74704 → 0.74648
- PR-AUC: 0.50670 → 0.50628

No improvement. Slight degradation.

### Random Forest
- ROC-AUC: 0.77878 → 0.78107
- PR-AUC: 0.55598 → 0.55663

Small but consistent improvement.

####

## Interpretation

The feature does not add new linear information for Logistic Regression.
However, it slightly improves Random Forest performance, likely by simplifying a non-linear interaction between balance and limit.

####

## Decision

Keep the feature in the pipeline, as it:

- Does not harm performance
- Slightly improves Random Forest
- Is business-interpretable