# Credit Default ML

## Problem

Binary classification task:
Predict whether a client will default on their credit card payment next month.

Target:
`default_payment_next_month`
- 1 = default
- 0 = no default

## Dataset

Taiwan credit card clients dataset.

Shape:
- 30,000 rows
- 25 columns

No missing values detected.

Target rate:
- Positive class (default): 22.12%
- Majority class baseline accuracy: 0.7788

## Primary Metrics

North Star metric:
- ROC-AUC

Operational metric:
- Recall (class 1)

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
venv\Scripts\activate
pip install -r requirements.txt

## Data Profiling Script

Generate automated dataset report:
python src/data/load_and_profile.py
--input data/raw/credit_default.xls
--out reports/profile_summary.json

This script validates:
- column structure
- target distribution
- baseline trivial model
- skewness statistics
- PAY_X unique values

