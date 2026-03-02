from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


from credit_ml.features.build import add_features


ModelName = Literal["log_reg", "rf"]


@dataclass(frozen=True)
class FeatureGroups:
    descriptive_no_numeric: list[str]
    numeric: list[str]
    pay: list[str]
    bill_amt: list[str]
    pay_amt: list[str]


def get_feature_groups() -> FeatureGroups:
    return FeatureGroups(
        descriptive_no_numeric=["SEX", "EDUCATION", "MARRIAGE"],
        numeric=["AGE", "LIMIT_BAL"],
        pay=["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
        bill_amt=["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"],
        pay_amt=["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"],
    )


def build_preprocessor() -> ColumnTransformer:
    groups = get_feature_groups()

    numeric_no_log = (
        groups.numeric
        + groups.pay
        + groups.bill_amt
        + ["credit_utilization", "util_x_pay0"]
    )

    numeric_log = groups.pay_amt
    categorical_features = groups.descriptive_no_numeric

    log_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
    ])

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("payamt_log", log_transformer, numeric_log),
            ("num", num_transformer, numeric_no_log),
            ("cat", cat_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def build_pipeline(model_name: ModelName) -> Pipeline:
    preprocessor = build_preprocessor()

    if model_name == "log_reg":
        estimator = LogisticRegression(max_iter=2000)  # Add final params when choosen (C, class_weight, etc)
    elif model_name == "rf":
        estimator = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    pipe = Pipeline(steps=[
        ("feat", FunctionTransformer(add_features, validate=False)),
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])

    return pipe