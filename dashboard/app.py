"""
Streamlit dashboard for the Credit Risk Prediction API.

Tabs:
    1. Predict - custom input OR load a real example from the held-out
       test set (for people unfamiliar with what realistic values look
       like). Calls the live /explain endpoint, shows a risk gauge and a
       SHAP feature contribution chart explaining the prediction.
    2. Drift Monitoring - placeholder until src/drift_monitor.py exists.

Model source (API vs local fallback):
    This dashboard tries the FastAPI backend first (POST /explain). If that
    call fails for any reason (backend not running, unreachable, timed out -
    e.g. because you deployed only this Streamlit app to a free-tier host
    that can't also run a second FastAPI process), it transparently falls
    back to running the same model in-process using common/inference.py.
    Same model file, same preprocessing, same SHAP explainer either way -
    see common/inference.py's module docstring for why that's guaranteed.
    A small badge under the input form always shows which path served the
    last prediction.

Run:
    streamlit run dashboard/app.py

The FastAPI app (app/main.py) is optional. Run it separately for the "live
API" path (default: http://localhost:8000); if you don't run it, or it's
unreachable, the dashboard just uses the local fallback automatically.
"""

import json
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Allow "from common.inference import ..." when this script is launched as
# `streamlit run dashboard/app.py` from the project root (Python only adds
# this script's own directory to sys.path, not the project root, so we add
# the parent directory ourselves).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.inference import explain_one, load_artifacts  # noqa: E402

st.set_page_config(page_title="Credit Risk Prediction", layout="wide", page_icon=None)

DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_TIMEOUT_SECONDS = 3  # keep short so the fallback kicks in quickly, not after a long hang
LOG_PATH = "logs/requests.jsonl"
TEST_DATA_PATH = "data/processed/test.csv"

RAW_FEATURE_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

FIELD_KEYS = {
    "RevolvingUtilizationOfUnsecuredLines": "f_revolving_utilization",
    "age": "f_age",
    "NumberOfTime30-59DaysPastDueNotWorse": "f_late_30_59",
    "DebtRatio": "f_debt_ratio",
    "MonthlyIncome": "f_monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "f_open_credit_lines",
    "NumberOfTimes90DaysLate": "f_late_90",
    "NumberRealEstateLoansOrLines": "f_real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "f_late_60_89",
    "NumberOfDependents": "f_dependents",
}

DEFAULT_VALUES = {
    "RevolvingUtilizationOfUnsecuredLines": 0.3,
    "age": 42,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.35,
    "MonthlyIncome": 5500,
    "NumberOfOpenCreditLinesAndLoans": 6,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2,
}

FEATURE_DISPLAY_NAMES = {
    "RevolvingUtilizationOfUnsecuredLines": "Credit Utilization",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "Late Payments (30-59 days)",
    "DebtRatio": "Debt Ratio",
    "MonthlyIncome": "Monthly Income",
    "MonthlyIncome_was_missing": "Income Was Unreported",
    "NumberOfOpenCreditLinesAndLoans": "Open Credit Lines/Loans",
    "NumberOfTimes90DaysLate": "Late Payments (90+ days)",
    "NumberRealEstateLoansOrLines": "Real Estate Loans/Lines",
    "NumberOfTime60-89DaysPastDueNotWorse": "Late Payments (60-89 days)",
    "NumberOfDependents": "Number of Dependents",
    "NumberOfDependents_was_missing": "Dependents Was Unreported",
}


@st.cache_data
def load_test_data():
    if not os.path.exists(TEST_DATA_PATH):
        return None
    return pd.read_csv(TEST_DATA_PATH)


@st.cache_resource(show_spinner="Loading model for local fallback...")
def get_local_bundle():
    """Loaded once per Streamlit session (cached) and reused for every
    fallback prediction, so the fallback path is fast after the first hit."""
    models_dir = os.path.join(PROJECT_ROOT, "models")
    return load_artifacts(models_dir=models_dir)


def get_api_url() -> str:
    return st.session_state.get("api_url", DEFAULT_API_URL)


def init_field_state():
    for col in RAW_FEATURE_COLUMNS:
        key = FIELD_KEYS[col]
        if key not in st.session_state:
            st.session_state[key] = DEFAULT_VALUES[col]


def load_example_into_state(row: pd.Series):
    for col in RAW_FEATURE_COLUMNS:
        key = FIELD_KEYS[col]
        value = row[col]
        if col in ("MonthlyIncome", "NumberOfDependents") and pd.isna(value):
            value = DEFAULT_VALUES[col]
        if col in ("age", "NumberOfTime30-59DaysPastDueNotWorse", "NumberOfOpenCreditLinesAndLoans",
                    "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines",
                    "NumberOfTime60-89DaysPastDueNotWorse"):
            value = int(value)
        st.session_state[key] = value


def call_api(endpoint: str, payload: dict, api_url: str) -> dict:
    response = requests.post(f"{api_url}/{endpoint}", json=payload, timeout=API_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def get_explanation(payload: dict, api_url: str) -> tuple[dict, str, str | None]:
    """Try the live FastAPI /explain endpoint first; if it's unreachable,
    times out, or errors, fall back to running the model in-process via
    common/inference.py. Returns (result, source, api_error) where source
    is "api" or "local" and api_error is the reason the API path was
    skipped (None if the API call succeeded)."""
    try:
        result = call_api("explain", payload, api_url)
        return result, "api", None
    except requests.exceptions.RequestException as e:
        try:
            bundle = get_local_bundle()
        except Exception as load_err:
            raise RuntimeError(
                f"API unreachable ({e}) and local fallback model failed to load "
                f"({load_err}). Make sure the models/ directory is present."
            ) from load_err
        result = explain_one(payload, bundle)
        return result, "local", str(e)


def render_risk_gauge(probability: float, threshold: float) -> go.Figure:
    color = "#C44E52" if probability >= threshold else "#55A868"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Predicted Default Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, threshold * 100], "color": "#E8F5E9"},
                    {"range": [threshold * 100, 100], "color": "#FFEBEE"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(t=50, b=10, l=20, r=20))
    return fig


def render_shap_chart(shap_contributions: dict, top_n: int = 8) -> go.Figure:
    items = sorted(shap_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    items = items[::-1]  # largest at top when plotted horizontally

    labels = [FEATURE_DISPLAY_NAMES.get(k, k) for k, _ in items]
    values = [v for _, v in items]
    colors = ["#C44E52" if v > 0 else "#55A868" for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="What drove this prediction (SHAP contribution, log-odds)",
        xaxis_title="Contribution to risk score (red = increases risk, green = decreases risk)",
        height=380,
        margin=dict(t=50, b=40, l=10, r=10),
    )
    return fig


def predict_tab():
    init_field_state()
    test_df = load_test_data()

    with st.sidebar:
        st.text_input("API URL", value=DEFAULT_API_URL, key="api_url")
        st.caption(
            "If this API isn't reachable, predictions automatically fall back "
            "to running the model locally inside this Streamlit app."
        )

    st.subheader("Applicant Risk Assessment")

    mode = st.radio(
        "Input mode",
        ["Custom input", "Load example from dataset"],
        horizontal=True,
    )

    if mode == "Load example from dataset":
        if test_df is None:
            st.warning(f"Could not find {TEST_DATA_PATH}. Run src/preprocess.py first.")
        else:
            idx = st.slider("Pick an applicant from the test set", 0, len(test_df) - 1, 0)
            example_row = test_df.iloc[idx]
            if st.button("Load this applicant"):
                load_example_into_state(example_row)
                st.rerun()
            actual_label = "Defaulted" if example_row["SeriousDlqin2yrs"] == 1 else "Did not default"
            st.caption(f"Actual historical outcome for this applicant: **{actual_label}** (ground truth, for comparison only - not shown to the model)")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.slider("Credit Utilization Ratio", 0.0, 2.0, step=0.01, key=FIELD_KEYS["RevolvingUtilizationOfUnsecuredLines"])
        st.number_input("Age", min_value=18, max_value=110, key=FIELD_KEYS["age"])
        st.number_input("Late Payments 30-59 Days (last 2 yrs)", min_value=0, key=FIELD_KEYS["NumberOfTime30-59DaysPastDueNotWorse"])
        st.slider("Debt Ratio", 0.0, 3.0, step=0.01, key=FIELD_KEYS["DebtRatio"])
        st.number_input("Monthly Income ($)", min_value=0, step=100, key=FIELD_KEYS["MonthlyIncome"])

    with col2:
        st.number_input("Open Credit Lines / Loans", min_value=0, key=FIELD_KEYS["NumberOfOpenCreditLinesAndLoans"])
        st.number_input("Late Payments 90+ Days", min_value=0, key=FIELD_KEYS["NumberOfTimes90DaysLate"])
        st.number_input("Real Estate Loans / Lines", min_value=0, key=FIELD_KEYS["NumberRealEstateLoansOrLines"])
        st.number_input("Late Payments 60-89 Days (last 2 yrs)", min_value=0, key=FIELD_KEYS["NumberOfTime60-89DaysPastDueNotWorse"])
        st.number_input("Number of Dependents", min_value=0, key=FIELD_KEYS["NumberOfDependents"])

    if st.button("Predict Risk", type="primary"):
        payload = {col: st.session_state[FIELD_KEYS[col]] for col in RAW_FEATURE_COLUMNS}

        try:
            result, source, api_error = get_explanation(payload, get_api_url())
        except RuntimeError as e:
            st.error(str(e))
            return

        if source == "api":
            st.success(f"Served by live API at {get_api_url()}")
        else:
            st.info(
                f"API unreachable ({api_error}) - served by the model running "
                "locally inside this app instead."
                
            )

        probability = result["default_probability"]
        threshold = result["decision_threshold"]
        is_high_risk = result["is_high_risk"]

        gauge_col, shap_col = st.columns([1, 1.4])

        with gauge_col:
            st.plotly_chart(render_risk_gauge(probability, threshold), use_container_width=True)
            if is_high_risk:
                st.error(f"HIGH RISK - flagged for review\n\n{probability:.1%} >= threshold {threshold:.1%}")
            else:
                st.success(f"LOW RISK\n\n{probability:.1%} < threshold {threshold:.1%}")
            st.caption(f"Model: {result['model_name']}")

        with shap_col:
            st.plotly_chart(render_shap_chart(result["shap_contributions"]), use_container_width=True)
            st.caption(
                "Contributions are in log-odds space (the model's raw score before converting "
                "to a probability), computed with SHAP TreeExplainer. Positive bars pushed the "
                "prediction toward higher risk; negative bars pushed it toward lower risk."
            )


def render_psi_heatmap(feature_level: list, windows: list) -> go.Figure:
    df = pd.DataFrame(feature_level)
    pivot = df.pivot(index="feature", columns="window", values="psi").reindex(columns=windows)
    pivot.index = [FEATURE_DISPLAY_NAMES.get(f, f) for f in pivot.index]

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="RdYlGn_r",
            zmin=0,
            zmax=max(0.5, pivot.values.max()),
            text=pivot.values.round(3),
            texttemplate="%{text}",
            colorbar={"title": "PSI"},
        )
    )
    fig.update_layout(
        title="Drift Heatmap - PSI per Feature per Window",
        height=450,
        margin=dict(t=50, b=40, l=10, r=10),
    )
    return fig


def render_psi_trend(window_summary: list) -> go.Figure:
    df = pd.DataFrame(window_summary)
    colors = [
        "#55A868" if sev == "stable" else "#DD8452" if sev == "moderate_drift" else "#C44E52"
        for sev in df["overall_severity"]
    ]

    fig = go.Figure(
        go.Bar(x=df["window"], y=df["mean"], marker_color=colors, text=df["overall_severity"], textposition="outside")
    )
    fig.add_hline(y=0.10, line_dash="dash", line_color="gray", annotation_text="Moderate drift (0.10)")
    fig.add_hline(y=0.25, line_dash="dash", line_color="black", annotation_text="Significant drift (0.25)")
    fig.update_layout(
        title="Mean PSI Across Features, Per Simulated Time Window",
        yaxis_title="Mean PSI",
        height=420,
        margin=dict(t=50, b=40, l=10, r=10),
    )
    return fig


def drift_tab():
    st.subheader("Drift Monitoring")
    st.caption(
        "Traffic shown here is simulated (src/simulate_traffic.py) to validate the drift "
        "detection mechanism, not real production usage - see README for details."
    )

    report_path = os.path.join("models", "drift_report.json")
    if not os.path.exists(report_path):
        st.warning(
            "No drift report found yet. Run src/simulate_traffic.py then src/drift_monitor.py "
            "from the project root to generate one."
        )
        return

    with open(report_path) as f:
        report = json.load(f)

    windows = report["windows"]
    window_summary = report["window_summary"]

    flagged = [w for w in window_summary if w["overall_severity"] == "significant_drift"]
    stable = [w for w in window_summary if w["overall_severity"] == "stable"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Windows analyzed", len(windows))
    col2.metric("Flagged as significant drift", len(flagged))
    col3.metric("Stable windows", len(stable))

    st.plotly_chart(render_psi_trend(window_summary), use_container_width=True)
    st.plotly_chart(render_psi_heatmap(report["feature_level"], windows), use_container_width=True)

    st.caption(
        f"PSI thresholds: stable below {report['thresholds']['stable_below']}, "
        f"significant drift at or above {report['thresholds']['significant_at_or_above']} "
        "(standard industry convention for population stability monitoring)."
    )

    with st.expander("Raw feature-level PSI table"):
        st.dataframe(pd.DataFrame(report["feature_level"]), use_container_width=True)


def main():
    st.title("Credit Risk Prediction Dashboard")
    tab1, tab2 = st.tabs(["Predict", "Drift Monitoring"])
    with tab1:
        predict_tab()
    with tab2:
        drift_tab()


if __name__ == "__main__":
    main()