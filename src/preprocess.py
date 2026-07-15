"""
Preprocessing script for the "Give Me Some Credit" dataset.

Revised after raw EDA (src/eda_raw.py). Every threshold below is derived
from percentiles/observed values rather than picked by convention - see
the EDA output and README for the reasoning behind each one.

Leakage-safe by construction: the raw row-level cleaning (dropping age==0)
happens before the split, since it's a fixed rule, not a statistic. But all
statistics used for imputation and capping (medians, percentile caps,
per-column delinquency maxima) are computed on the TRAIN split only, then
applied identically to the test split and saved to disk so the exact same
transform can be applied to live API requests later.

Run:
    python src/preprocess.py --input data/raw/cs-training.csv
"""

import argparse
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COL = "SeriousDlqin2yrs"

DELINQUENCY_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
SENTINEL_CODES = [96, 98]

# Percentile choices below are justified by the raw EDA output:
# - Utilization's tail breaks between the 99.5th and 99.9th percentile
#   (1.37 vs 1571) - the 99.9th is dominated by a handful of broken values,
#   so 99.5th is the defensible cutoff.
# - DebtRatio's tail is even noisier (income-near-zero artifact), so we use
#   the tighter 99th percentile.
# - MonthlyIncome gets the same 99th percentile treatment for the same
#   "extreme outliers likely data errors" reasoning.
UTILIZATION_CAP_PERCENTILE = 0.995
DEBT_RATIO_CAP_PERCENTILE = 0.99
MONTHLY_INCOME_CAP_PERCENTILE = 0.99


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df


def drop_impossible_rows(df: pd.DataFrame) -> pd.DataFrame:
    # age == 0 is a fixed, unambiguous data-entry error - not a statistic,
    # so this is safe to do before the split.
    before = len(df)
    df = df[df["age"] != 0].reset_index(drop=True)
    print(f"Dropped {before - len(df)} row(s) with age == 0.")
    return df


def fit_preprocessing_artifacts(train_df: pd.DataFrame) -> dict:
    """Compute every imputation/capping statistic from the TRAIN split only."""
    artifacts = {}

    artifacts["MonthlyIncome_median"] = float(train_df["MonthlyIncome"].median())
    artifacts["NumberOfDependents_median"] = float(train_df["NumberOfDependents"].median())

    artifacts["monthly_income_cap"] = float(
        train_df["MonthlyIncome"].quantile(MONTHLY_INCOME_CAP_PERCENTILE)
    )
    artifacts["revolving_utilization_cap"] = float(
        train_df["RevolvingUtilizationOfUnsecuredLines"].quantile(UTILIZATION_CAP_PERCENTILE)
    )
    artifacts["debt_ratio_cap"] = float(
        train_df["DebtRatio"].quantile(DEBT_RATIO_CAP_PERCENTILE)
    )

    delinquency_caps = {}
    for col in DELINQUENCY_COLS:
        non_sentinel = train_df.loc[~train_df[col].isin(SENTINEL_CODES), col]
        delinquency_caps[col] = float(non_sentinel.max())
    artifacts["delinquency_caps"] = delinquency_caps

    return artifacts


def apply_preprocessing(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Apply a fitted set of artifacts to any dataframe (train, test, or a live request)."""
    df = df.copy()

    # Missingness flags, computed before imputation
    df["MonthlyIncome_was_missing"] = df["MonthlyIncome"].isnull().astype(int)
    df["NumberOfDependents_was_missing"] = df["NumberOfDependents"].isnull().astype(int)

    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(artifacts["MonthlyIncome_median"])
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(artifacts["NumberOfDependents_median"])

    # Percentile-based caps
    df["MonthlyIncome"] = df["MonthlyIncome"].clip(upper=artifacts["monthly_income_cap"])
    df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(
        upper=artifacts["revolving_utilization_cap"]
    )
    df["DebtRatio"] = df["DebtRatio"].clip(upper=artifacts["debt_ratio_cap"])

    # Delinquency sentinel codes capped at each column's own true observed max
    for col in DELINQUENCY_COLS:
        df[col] = df[col].clip(upper=artifacts["delinquency_caps"][col])

    return df


def main():
    parser = argparse.ArgumentParser(description="Preprocess Give Me Some Credit dataset")
    parser.add_argument("--input", type=str, default="data/raw/cs-training.csv")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--artifacts-path", type=str, default="models/preprocessing_artifacts.json")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = load_data(args.input)
    print(f"Loaded raw data: {df.shape}")

    df = drop_impossible_rows(df)

    # Split BEFORE computing any imputation/capping statistics, to avoid leakage
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    train_df_raw = X_train.copy()
    train_df_raw[TARGET_COL] = y_train
    test_df_raw = X_test.copy()
    test_df_raw[TARGET_COL] = y_test

    artifacts = fit_preprocessing_artifacts(train_df_raw)

    print("\nFitted preprocessing artifacts (from train split only):")
    print(json.dumps(artifacts, indent=2))

    train_df = apply_preprocessing(train_df_raw, artifacts)
    test_df = apply_preprocessing(test_df_raw, artifacts)

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.csv")
    test_path = os.path.join(args.output_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nTrain set: {train_df.shape} saved to {train_path}")
    print(f"Test set:  {test_df.shape} saved to {test_path}")
    print(f"Train default rate: {y_train.mean():.4f}")
    print(f"Test default rate:  {y_test.mean():.4f}")

    os.makedirs(os.path.dirname(args.artifacts_path), exist_ok=True)
    with open(args.artifacts_path, "w") as f:
        json.dump(artifacts, f, indent=2)
    print(f"\nPreprocessing artifacts saved to {args.artifacts_path}")


if __name__ == "__main__":
    main()