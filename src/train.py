"""
Training script: trains Logistic Regression, Random Forest, XGBoost, and
LightGBM, each with an unweighted baseline plus class-imbalance-handling
variant(s) (class_weight='balanced' and/or scale_pos_weight, depending on
what each model natively supports).

All metrics below are computed at the default 0.5 probability threshold.
Threshold tuning is handled separately in a later step.

Run:
    python src/train.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

TARGET_COL = "SeriousDlqin2yrs"
DATA_DIR = "data/processed"
MODELS_DIR = "models"
RANDOM_STATE = 42


def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    return X_train, y_train, X_test, y_test


def evaluate(model, X_test, y_test, sample_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "model": sample_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    X_train, y_train, X_test, y_test = load_data()
    feature_columns = list(X_train.columns)

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    balanced_sample_weight = compute_sample_weight("balanced", y_train)

    print(f"Train negatives: {neg_count}, positives: {pos_count}")
    print(f"scale_pos_weight (neg/pos ratio): {scale_pos_weight:.4f}")
    print()

    # Logistic Regression needs scaled features; tree-based models do not.
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    results = []
    trained_models = {}

    # ---------------- Logistic Regression ----------------
    lr_baseline = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_baseline.fit(X_train_scaled, y_train)
    results.append(evaluate(lr_baseline, X_test_scaled, y_test, "LogisticRegression_baseline"))
    trained_models["LogisticRegression_baseline"] = lr_baseline

    lr_balanced = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    )
    lr_balanced.fit(X_train_scaled, y_train)
    results.append(evaluate(lr_balanced, X_test_scaled, y_test, "LogisticRegression_balanced"))
    trained_models["LogisticRegression_balanced"] = lr_balanced

    # ---------------- Random Forest ----------------
    rf_baseline = RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_baseline.fit(X_train, y_train)
    results.append(evaluate(rf_baseline, X_test, y_test, "RandomForest_baseline"))
    trained_models["RandomForest_baseline"] = rf_baseline

    rf_balanced = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_balanced.fit(X_train, y_train)
    results.append(evaluate(rf_balanced, X_test, y_test, "RandomForest_balanced"))
    trained_models["RandomForest_balanced"] = rf_balanced

    # ---------------- XGBoost ----------------
    xgb_baseline = XGBClassifier(
        n_estimators=300, random_state=RANDOM_STATE, eval_metric="logloss"
    )
    xgb_baseline.fit(X_train, y_train)
    results.append(evaluate(xgb_baseline, X_test, y_test, "XGBoost_baseline"))
    trained_models["XGBoost_baseline"] = xgb_baseline

    xgb_scale_pos_weight = XGBClassifier(
        n_estimators=300,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    xgb_scale_pos_weight.fit(X_train, y_train)
    results.append(evaluate(xgb_scale_pos_weight, X_test, y_test, "XGBoost_scale_pos_weight"))
    trained_models["XGBoost_scale_pos_weight"] = xgb_scale_pos_weight

    xgb_balanced_sw = XGBClassifier(
        n_estimators=300, random_state=RANDOM_STATE, eval_metric="logloss"
    )
    xgb_balanced_sw.fit(X_train, y_train, sample_weight=balanced_sample_weight)
    results.append(evaluate(xgb_balanced_sw, X_test, y_test, "XGBoost_balanced_sample_weight"))
    trained_models["XGBoost_balanced_sample_weight"] = xgb_balanced_sw

    # ---------------- LightGBM ----------------
    lgbm_baseline = LGBMClassifier(
        n_estimators=300, random_state=RANDOM_STATE, verbose=-1
    )
    lgbm_baseline.fit(X_train, y_train)
    results.append(evaluate(lgbm_baseline, X_test, y_test, "LightGBM_baseline"))
    trained_models["LightGBM_baseline"] = lgbm_baseline

    lgbm_class_weight = LGBMClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, verbose=-1
    )
    lgbm_class_weight.fit(X_train, y_train)
    results.append(evaluate(lgbm_class_weight, X_test, y_test, "LightGBM_class_weight_balanced"))
    trained_models["LightGBM_class_weight_balanced"] = lgbm_class_weight

    lgbm_scale_pos_weight = LGBMClassifier(
        n_estimators=300,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    lgbm_scale_pos_weight.fit(X_train, y_train)
    results.append(evaluate(lgbm_scale_pos_weight, X_test, y_test, "LightGBM_scale_pos_weight"))
    trained_models["LightGBM_scale_pos_weight"] = lgbm_scale_pos_weight

    # ---------------- Results ----------------
    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    results_df = results_df.round(4)

    print()
    print("=" * 90)
    print("MODEL COMPARISON (threshold = 0.5)")
    print("=" * 90)
    print(results_df.to_string(index=False))
    print("=" * 90)

    results_path = os.path.join(MODELS_DIR, "model_comparison.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nComparison table saved to {results_path}")

    # Save every trained model so evaluate.py can load any of them later
    for name, model in trained_models.items():
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.pkl"))
    print(f"All {len(trained_models)} trained models saved to {MODELS_DIR}/")

    # Save feature column order - the API must build request features in
    # this exact order/name before calling model.predict_proba
    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f, indent=2)
    print(f"Feature column order saved to {MODELS_DIR}/feature_columns.json")


if __name__ == "__main__":
    main()