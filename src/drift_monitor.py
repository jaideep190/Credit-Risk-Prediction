"""
Drift monitoring: computes the Population Stability Index (PSI) for each
model feature, comparing incoming request windows against the training
distribution.

Reference distribution: data/processed/train.csv (already preprocessed -
this is the actual distribution the model was trained on).

Incoming data: logs/requests.jsonl, written by app/main.py on every
/predict or /explain call, grouped by the 'window' tag set by
src/simulate_traffic.py. Raw incoming values are passed through the same
apply_preprocessing() used at training time before computing PSI, so the
comparison is apples-to-apples (processed vs processed).

PSI interpretation (standard industry thresholds):
    PSI < 0.10             -> stable, no meaningful drift
    0.10 <= PSI < 0.25      -> moderate drift, worth monitoring
    PSI >= 0.25             -> significant drift, investigate/retrain

Outputs:
    models/drift_report.json   - full results, consumed by the dashboard
    models/drift_report.csv    - same, flat table
    images/psi_heatmap.png     - window x feature PSI heatmap
    images/psi_by_window.png   - mean PSI per window, in chronological order

Run:
    python src/drift_monitor.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.preprocessing import apply_preprocessing

TARGET_COL = "SeriousDlqin2yrs"
TRAIN_PATH = "data/processed/train.csv"
LOG_PATH = "logs/requests.jsonl"
ARTIFACTS_PATH = "models/preprocessing_artifacts.json"
FEATURE_COLUMNS_PATH = "models/feature_columns.json"
IMAGES_DIR = "images"
MODELS_DIR = "models"

STABLE_THRESHOLD = 0.10
SIGNIFICANT_THRESHOLD = 0.25
N_BINS = 10


def calculate_psi(reference: pd.Series, actual: pd.Series, bins: int = N_BINS) -> float:
    """Population Stability Index between two 1-D distributions, using
    quantile bins derived from the reference distribution."""
    reference = reference.dropna()
    actual = actual.dropna()

    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.unique(reference.quantile(quantiles).values)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    if len(bin_edges) < 3:
        # Reference column has too little variation to bin meaningfully
        # (e.g. a near-constant flag column) - PSI is not meaningful here.
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    act_pct = act_counts / max(act_counts.sum(), 1)

    epsilon = 1e-4
    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    act_pct = np.where(act_pct == 0, epsilon, act_pct)

    psi = float(np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct)))
    return psi


def classify_psi(psi_value: float) -> str:
    if psi_value < STABLE_THRESHOLD:
        return "stable"
    elif psi_value < SIGNIFICANT_THRESHOLD:
        return "moderate_drift"
    else:
        return "significant_drift"


def load_reference() -> pd.DataFrame:
    return pd.read_csv(TRAIN_PATH)


def load_logged_requests() -> pd.DataFrame:
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(
            f"{LOG_PATH} not found. Run the API and src/simulate_traffic.py first."
        )
    rows = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    with open(ARTIFACTS_PATH) as f:
        artifacts = json.load(f)
    with open(FEATURE_COLUMNS_PATH) as f:
        feature_columns = json.load(f)

    reference_df = load_reference()
    print(f"Reference distribution (train set): {reference_df.shape}")

    logs_df = load_logged_requests()
    print(f"Loaded {len(logs_df)} logged requests")

    logs_df = logs_df[logs_df["window"].notna()]
    print(f"Of those, {len(logs_df)} are tagged with a simulated window label")

    if logs_df.empty:
        print("No windowed traffic found. Run src/simulate_traffic.py first.")
        return

    windows = sorted(logs_df["window"].unique())
    print(f"Windows found: {windows}")

    results = []
    for window in windows:
        window_df_raw = logs_df[logs_df["window"] == window].copy()
        window_df_processed = apply_preprocessing(window_df_raw, artifacts)

        for feature in feature_columns:
            psi_value = calculate_psi(reference_df[feature], window_df_processed[feature])
            results.append({
                "window": window,
                "feature": feature,
                "psi": round(psi_value, 4),
                "severity": classify_psi(psi_value),
            })

    results_df = pd.DataFrame(results)

    print()
    print("=" * 90)
    print("PSI DRIFT REPORT (feature-level, per window)")
    print("=" * 90)
    print(results_df.to_string(index=False))

    window_summary = (
        results_df.groupby("window")["psi"]
        .agg(["mean", "max"])
        .reindex(windows)
        .round(4)
    )
    window_summary["overall_severity"] = window_summary["max"].apply(classify_psi)

    print()
    print("=" * 90)
    print("WINDOW-LEVEL SUMMARY (mean/max PSI across all features)")
    print("=" * 90)
    print(window_summary.to_string())
    print("=" * 90)

    # ---------------- Save machine-readable report ----------------
    report = {
        "windows": windows,
        "feature_level": results,
        "window_summary": window_summary.reset_index().to_dict(orient="records"),
        "thresholds": {"stable_below": STABLE_THRESHOLD, "significant_at_or_above": SIGNIFICANT_THRESHOLD},
    }
    with open(os.path.join(MODELS_DIR, "drift_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    results_df.to_csv(os.path.join(MODELS_DIR, "drift_report.csv"), index=False)
    print(f"\nSaved drift_report.json and drift_report.csv to {MODELS_DIR}/")

    # ---------------- Heatmap: window x feature ----------------
    pivot = results_df.pivot(index="feature", columns="window", values="psi").reindex(columns=windows)
    fig, ax = plt.subplots(figsize=(max(8, len(windows) * 1.8), 7))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdYlGn_r", center=STABLE_THRESHOLD,
        vmin=0, vmax=max(0.5, pivot.values.max()), ax=ax, cbar_kws={"label": "PSI"}
    )
    ax.set_title("Drift Heatmap - PSI per Feature per Window")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "psi_heatmap.png"), dpi=150)
    plt.close()
    print("Saved psi_heatmap.png")

    # ---------------- Bar chart: mean PSI per window, chronological ----------------
    fig, ax = plt.subplots(figsize=(max(7, len(windows) * 1.5), 5))
    colors = [
        "#55A868" if classify_psi(v) == "stable"
        else "#DD8452" if classify_psi(v) == "moderate_drift"
        else "#C44E52"
        for v in window_summary["mean"]
    ]
    ax.bar(window_summary.index, window_summary["mean"], color=colors)
    ax.axhline(STABLE_THRESHOLD, color="gray", linestyle="--", linewidth=1, label="Moderate drift threshold (0.10)")
    ax.axhline(SIGNIFICANT_THRESHOLD, color="black", linestyle="--", linewidth=1, label="Significant drift threshold (0.25)")
    ax.set_ylabel("Mean PSI across features")
    ax.set_title("Drift Over Simulated Time Windows")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "psi_by_window.png"), dpi=150)
    plt.close()
    print("Saved psi_by_window.png")


if __name__ == "__main__":
    main()