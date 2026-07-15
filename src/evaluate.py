"""
Evaluation script for the champion model (LightGBM_baseline - chosen for
having the best ROC AUC across all trained variants; see model_comparison.csv).

What this does:
1. Loads the champion model and the held-out test set.
2. Plots and saves the ROC curve (with AUC) and Precision-Recall curve.
3. Computes two candidate decision thresholds:
   - Youden's J statistic (maximizes TPR - FPR) - the threshold actually
     selected for deployment. Standard, cost-agnostic technique for
     choosing an operating point on the ROC curve.
   - F1-optimal threshold - shown for comparison only, not selected.
4. Prints a comparison table of accuracy/precision/recall/f1/specificity
   at 0.5, the Youden's J threshold, and the F1-optimal threshold, so the
   trade-off is visible rather than hidden behind one chosen number.
5. Saves confusion matrices (at 0.5 and at the selected threshold) as images.
6. Saves the selected threshold + champion model name to
   models/decision_threshold.json for the API to load.

Run:
    python src/evaluate.py
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

TARGET_COL = "SeriousDlqin2yrs"
CHAMPION_MODEL_NAME = "LightGBM_baseline"
MODELS_DIR = "models"
IMAGES_DIR = "images"
DATA_DIR = "data/processed"


def load_test_data():
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    return X_test, y_test


def metrics_at_threshold(y_test, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    return {
        "threshold": round(threshold, 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "specificity": round(specificity, 4),
        "f1": round(f1_score(y_test, y_pred), 4),
    }


def plot_roc_curve(y_test, y_proba, youden_threshold, youden_idx, fpr, tpr, roc_auc_value):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#4C72B0", label=f"ROC curve (AUC = {roc_auc_value:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random classifier")
    ax.scatter(
        fpr[youden_idx],
        tpr[youden_idx],
        color="#C44E52",
        zorder=5,
        label=f"Youden's J threshold = {youden_threshold:.4f}",
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {CHAMPION_MODEL_NAME}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "roc_curve.png"), dpi=150)
    plt.close()
    print("Saved roc_curve.png")


def plot_pr_curve(precision, recall, f1_threshold, f1_idx):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#4C72B0", label="Precision-Recall curve")
    ax.scatter(
        recall[f1_idx],
        precision[f1_idx],
        color="#C44E52",
        zorder=5,
        label=f"F1-optimal threshold = {f1_threshold:.4f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve - {CHAMPION_MODEL_NAME}")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "precision_recall_curve.png"), dpi=150)
    plt.close()
    print("Saved precision_recall_curve.png")


def plot_confusion_matrices(y_test, y_proba, selected_threshold):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, threshold, title in [
        (axes[0], 0.5, "Threshold = 0.5 (default)"),
        (axes[1], selected_threshold, f"Threshold = {selected_threshold:.4f} (Youden's J, selected)"),
    ]:
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Default", "Default"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "confusion_matrices.png"), dpi=150)
    plt.close()
    print("Saved confusion_matrices.png")


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, f"{CHAMPION_MODEL_NAME}.pkl")
    model = joblib.load(model_path)
    print(f"Loaded champion model: {CHAMPION_MODEL_NAME}")

    X_test, y_test = load_test_data()
    y_proba = model.predict_proba(X_test)[:, 1]

    # ---------------- ROC curve + Youden's J ----------------
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
    roc_auc_value = roc_auc_score(y_test, y_proba)

    youden_j = tpr - fpr
    youden_idx = np.argmax(youden_j)
    youden_threshold = roc_thresholds[youden_idx]

    plot_roc_curve(y_test, y_proba, youden_threshold, youden_idx, fpr, tpr, roc_auc_value)

    # ---------------- Precision-Recall curve + F1-optimal ----------------
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
    # precision_recall_curve returns len(thresholds) = len(precision) - 1
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    f1_idx = np.argmax(f1_scores)
    f1_threshold = pr_thresholds[f1_idx]

    plot_pr_curve(precision, recall, f1_threshold, f1_idx)

    # ---------------- Threshold comparison table ----------------
    candidate_thresholds = {
        "default_0.5": 0.5,
        "youden_j": float(youden_threshold),
        "f1_optimal": float(f1_threshold),
    }

    rows = []
    for label, threshold in candidate_thresholds.items():
        row = metrics_at_threshold(y_test, y_proba, threshold)
        row["strategy"] = label
        rows.append(row)

    comparison_df = pd.DataFrame(rows)[
        ["strategy", "threshold", "accuracy", "precision", "recall", "specificity", "f1"]
    ]

    print()
    print("=" * 90)
    print(f"THRESHOLD COMPARISON - {CHAMPION_MODEL_NAME} (test set, n={len(y_test)})")
    print("=" * 90)
    print(comparison_df.to_string(index=False))
    print("=" * 90)
    print(f"\nROC AUC (threshold-independent): {roc_auc_value:.4f}")

    # ---------------- Selected threshold: Youden's J ----------------
    selected_threshold = float(youden_threshold)
    plot_confusion_matrices(y_test, y_proba, selected_threshold)

    selected_metrics = metrics_at_threshold(y_test, y_proba, selected_threshold)
    print(f"\nSELECTED THRESHOLD (Youden's J): {selected_threshold:.4f}")
    print(f"Metrics at selected threshold: {selected_metrics}")

    decision_config = {
        "champion_model": CHAMPION_MODEL_NAME,
        "selected_threshold": selected_threshold,
        "selection_method": "youdens_j",
        "test_set_metrics_at_selected_threshold": selected_metrics,
        "test_set_roc_auc": round(roc_auc_value, 4),
    }
    threshold_path = os.path.join(MODELS_DIR, "decision_threshold.json")
    with open(threshold_path, "w") as f:
        json.dump(decision_config, f, indent=2)
    print(f"\nSaved decision config to {threshold_path}")

    comparison_df.to_csv(os.path.join(MODELS_DIR, "threshold_comparison.csv"), index=False)
    print(f"Saved threshold comparison table to {MODELS_DIR}/threshold_comparison.csv")


if __name__ == "__main__":
    main()