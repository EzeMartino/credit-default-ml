# Credit Default ML

## Problem

Binary classification task:
Predict whether a client will default on their credit card payment next month.

Target:
`default\\\\\\\_payment\\\\\\\_next\\\\\\\_month`

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

## Primary Metrics

North Star metric:

* ROC-AUC

Operational metric:

* Recall (class 1)

## Project Structure

credit-default-ml/
│
├── data/
├── notebooks/
├── reports/
├── src/
├── tests/



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



## Baseline Comparison

To run a baseline comparison between the Logistic Regression, Random Forest and a Dummy model

python -m src.models.train\_baseline

This will:

* Load the dataset
* Perform 5-fold stratified cross-validation for all models with different scorings
* Create a table in the results variable for reporting
* Create a report in reports/baseline\_results.json
* The report will include mean and std deviation of ROC-AUC, PR-AUC and F1 score for the models



