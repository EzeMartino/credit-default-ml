import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import argparse

TARGET = "default payment next month"
RANDOM_STATE = 42

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/credit_default.xls")
    return p.parse_args()

def load_data(path):
    df = pd.read_excel(path, header=1)
    
    # Rename if necessary
    if "Y" in df.columns:
        df = df.rename(columns={"Y": TARGET})
    
    X = df.drop(columns=[TARGET, "ID"], errors="ignore")
    y = df[TARGET]

    return (X,y)

def build_model(X, model):
    groups = {    
        "descriptive_no_numeric": ["SEX", "EDUCATION", "MARRIAGE"],

        "numeric": ["AGE", "LIMIT_BAL"],
    
        "pay": ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
    
        "bill_amt": ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"],
    
        "pay_amt": ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"],
    }

    numeric_no_log = groups["numeric"]+groups["pay"]+groups["bill_amt"]

    numeric_log = groups["pay_amt"]

    categorical_features = groups["descriptive_no_numeric"]

    log_transformer = Pipeline(steps=[
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scaler", StandardScaler())
    ])
    
    num_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    
    log_model_preprocessor = ColumnTransformer(
        transformers=[
            ("payamt_log", log_transformer, numeric_log),
            ("num", num_transformer, numeric_no_log),
            ("cat", OneHotEncoder(drop="first"), categorical_features),
        ],
        remainder="drop"
    )
    if (model == 'log_reg'):
        log_model = Pipeline(steps=[
            ("preprocessor", log_model_preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ])
        return log_model
    elif (model == 'rf'):
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )

        rf_model = Pipeline([
            ("preprocessor", log_model_preprocessor),
            ("classifier", rf)
        ])
        return rf_model
    else: return(ValueError)
    
    

def main():
    args = parse_args()
    X,y = load_data(args.input)

    log_model = build_model(X)

    # Cross-Validation (CV)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(
        log_model,
        X,
        y,
        cv=cv,
        scoring="roc_auc"
    )
    print("CV ROC-AUC mean:", scores.mean())
    print("CV std:", scores.std())

    # Holdout evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    log_model.fit(X_train, y_train)
    y_proba_log_model = log_model.predict_proba(X_test)[:, 1]

    print("Holdout ROC-AUC:", roc_auc_score(y_test, y_proba_log_model))


if __name__ == "__main__":
    main()