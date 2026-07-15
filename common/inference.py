"""
Shared inference logic - the single source of truth for turning raw
applicant data into a prediction (+ optional SHAP explanation).

Used by:
    - app/main.py (FastAPI) - wraps these functions as HTTP endpoints
    - dashboard/app.py - calls these functions directly, in-process, when
      the FastAPI backend is unreachable (e.g. when deployed standalone on
      Streamlit Community Cloud, which can't run a second backend process)

Keeping this logic in one place means the API and the dashboard's fallback
can NEVER silently diverge - same model, same preprocessing, same SHAP
explainer, regardless of which path served the request.
"""

import json
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
import shap

from common.preprocessing import RAW_FEATURE_COLUMNS, apply_preprocessing

MODELS_DIR = "models"
LOGS_DIR = "logs"
REQUEST_LOG_PATH = os.path.join(LOGS_DIR, "requests.jsonl")


def load_artifacts(models_dir: str = MODELS_DIR) -> dict:
    """Load the champion model, preprocessing artifacts, decision threshold,
    and build the SHAP explainer. Called once at FastAPI startup, and once
    per Streamlit session (cached) for the local fallback path."""
    with open(os.path.join(models_dir, "decision_threshold.json")) as f:
        decision_config = json.load(f)

    model_name = decision_config["champion_model"]
    model = joblib.load(os.path.join(models_dir, f"{model_name}.pkl"))

    with open(os.path.join(models_dir, "preprocessing_artifacts.json")) as f:
        preprocessing_artifacts = json.load(f)

    with open(os.path.join(models_dir, "feature_columns.json")) as f:
        feature_columns = json.load(f)

    explainer = shap.TreeExplainer(model)

    return {
        "model": model,
        "model_name": model_name,
        "threshold": decision_config["selected_threshold"],
        "preprocessing_artifacts": preprocessing_artifacts,
        "feature_columns": feature_columns,
        "shap_explainer": explainer,
    }


def log_request(raw_values: dict, window: str = None) -> None:
    os.makedirs(LOGS_DIR, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "window": window,
        **raw_values,
    }
    with open(REQUEST_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _build_processed_df(raw_dict: dict, bundle: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw_dict])[RAW_FEATURE_COLUMNS]
    processed_df = apply_preprocessing(df, bundle["preprocessing_artifacts"])
    return processed_df[bundle["feature_columns"]]


def predict_one(raw_dict: dict, bundle: dict, window: str = None, log: bool = True) -> dict:
    processed_df = _build_processed_df(raw_dict, bundle)
    probability = float(bundle["model"].predict_proba(processed_df)[0, 1])
    is_high_risk = probability >= bundle["threshold"]

    if log:
        log_request(raw_dict, window=window)

    return {
        "default_probability": round(probability, 4),
        "is_high_risk": is_high_risk,
        "decision_threshold": bundle["threshold"],
        "model_name": bundle["model_name"],
    }


def explain_one(raw_dict: dict, bundle: dict, window: str = None, log: bool = True) -> dict:
    processed_df = _build_processed_df(raw_dict, bundle)
    probability = float(bundle["model"].predict_proba(processed_df)[0, 1])
    is_high_risk = probability >= bundle["threshold"]

    if log:
        log_request(raw_dict, window=window)

    explainer = bundle["shap_explainer"]
    raw_shap_values = explainer.shap_values(processed_df)
    raw_expected_value = explainer.expected_value

    # shap's return shape varies by version - handle both list and array forms
    if isinstance(raw_shap_values, list):
        instance_shap_values = raw_shap_values[1][0]
        base_value = float(raw_expected_value[1])
    else:
        instance_shap_values = raw_shap_values[0]
        base_value = float(raw_expected_value)

    shap_contributions = {
        col: float(val) for col, val in zip(bundle["feature_columns"], instance_shap_values)
    }

    return {
        "default_probability": round(probability, 4),
        "is_high_risk": is_high_risk,
        "decision_threshold": bundle["threshold"],
        "model_name": bundle["model_name"],
        "base_value": round(base_value, 4),
        "shap_contributions": shap_contributions,
    }