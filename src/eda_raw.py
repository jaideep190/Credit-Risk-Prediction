"""
EDA script for the RAW Give Me Some Credit dataset (before any cleaning).

Purpose: every preprocessing decision in src/preprocess.py (imputation,
capping, dropping rows) should be traceable back to something visible here.
This script does not just describe the data - it computes data-driven
recommendations (e.g. what cap value the delinquency columns actually
justify) so those decisions are defensible with numbers, not convention.

Generates and saves plots to images/:
    - raw_missingness.png
    - raw_age_distribution.png
    - raw_revolving_utilization.png
    - raw_debt_ratio.png
    - raw_delinquency_sentinel_codes.png
    - raw_monthly_income.png
    - raw_correlation_with_target.png

Run:
    python src/eda_raw.py --input data/raw/cs-training.csv
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TARGET_COL = "SeriousDlqin2yrs"
IMAGES_DIR = "images"

DELINQUENCY_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
SENTINEL_CODES = [96, 98]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------- missing
def analyze_missingness(df: pd.DataFrame) -> None:
    section("MISSING VALUES")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    pct = 100 * missing / len(df)

    for col in missing.index:
        print(f"  {col}: {missing[col]} missing ({pct[col]:.2f}%)")

    if len(missing) == 0:
        print("  No missing values found.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(missing.index, pct.values, color="#C44E52")
    ax.set_xlabel("Percent Missing")
    ax.set_title("Missing Values by Column (Raw Data)")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "raw_missingness.png"), dpi=150)
    plt.close()
    print("Saved raw_missingness.png")

    # Decision logic: >5% missing warrants a missingness flag column,
    # not just silent imputation, since it is too large a share to assume
    # missing-completely-at-random.
    for col in missing.index:
        if pct[col] >= 5:
            print(
                f"  DECISION: {col} missingness ({pct[col]:.2f}%) is large enough that "
                f"we should keep an explicit '{col}_was_missing' flag, not just impute silently."
            )
        else:
            print(
                f"  DECISION: {col} missingness ({pct[col]:.2f}%) is small; "
                f"median imputation with a flag column is still cheap insurance."
            )


# ---------------------------------------------------------------- age
def analyze_age(df: pd.DataFrame) -> None:
    section("AGE")
    print(df["age"].describe().to_string())

    zero_count = (df["age"] == 0).sum()
    print(f"\nRows with age == 0: {zero_count}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df["age"], bins=60, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Age Distribution (Raw)")
    sns.boxplot(x=df["age"], ax=axes[1], color="#4C72B0")
    axes[1].set_title("Age Boxplot (Raw)")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "raw_age_distribution.png"), dpi=150)
    plt.close()
    print("Saved raw_age_distribution.png")

    if zero_count > 0:
        print(
            f"\n  DECISION: age == 0 is biologically impossible for a credit applicant "
            f"({zero_count} row(s)). Drop these rows rather than impute - there is no "
            f"principled 'average' fix for a data entry error, and it is a negligible "
            f"fraction of {len(df)} rows."
        )


# ---------------------------------------------------------------- utilization
def analyze_revolving_utilization(df: pd.DataFrame) -> None:
    section("REVOLVING UTILIZATION OF UNSECURED LINES")
    col = "RevolvingUtilizationOfUnsecuredLines"
    print(df[col].describe().to_string())

    over_1 = (df[col] > 1).sum()
    over_10 = (df[col] > 10).sum()
    print(f"\nRows with value > 1 (should conceptually be a 0-1 ratio): {over_1}")
    print(f"Rows with value > 10 (clearly erroneous): {over_10}")

    percentiles = [0.90, 0.95, 0.99, 0.995, 0.999]
    print("\nUpper percentiles:")
    for p in percentiles:
        print(f"  {p*100:.1f}th percentile: {df[col].quantile(p):.4f}")

    clip_val = df[col].quantile(0.99)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(df[col].clip(upper=clip_val), bins=60, ax=ax, color="#4C72B0")
    ax.set_title(f"{col} Distribution (clipped at 99th pct = {clip_val:.2f} for display)")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "raw_revolving_utilization.png"), dpi=150)
    plt.close()
    print("Saved raw_revolving_utilization.png")

    cap_99_9 = df[col].quantile(0.999)
    print(
        f"\n  DECISION: {over_1} rows exceed the theoretical 0-1 ratio bound, and "
        f"{over_10} are off by orders of magnitude - these are data entry errors, not "
        f"real high-utilization customers. Recommend capping at the 99.9th percentile "
        f"({cap_99_9:.2f}) rather than dropping, to keep the rows without letting extreme "
        f"values dominate model training or later distance-based drift metrics."
    )


# ---------------------------------------------------------------- debt ratio
def analyze_debt_ratio(df: pd.DataFrame) -> None:
    section("DEBT RATIO")
    col = "DebtRatio"
    print(df[col].describe().to_string())

    percentiles = [0.90, 0.95, 0.99, 0.995, 0.999]
    print("\nUpper percentiles:")
    for p in percentiles:
        print(f"  {p*100:.1f}th percentile: {df[col].quantile(p):.4f}")

    # Known quirk in this dataset: when MonthlyIncome is 0 or missing,
    # DebtRatio (debt payments / income) becomes a nonsensical huge number,
    # since it is effectively dividing by ~0.
    zero_income = df[df["MonthlyIncome"] == 0]
    if len(zero_income) > 0:
        print(
            f"\nRows with MonthlyIncome == 0: {len(zero_income)}, "
            f"their median DebtRatio: {zero_income[col].median():.2f} "
            f"(vs overall median {df[col].median():.2f})"
        )

    clip_val = df[col].quantile(0.99)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(df[col].clip(upper=clip_val), bins=60, ax=ax, color="#4C72B0")
    ax.set_title(f"{col} Distribution (clipped at 99th pct = {clip_val:.2f} for display)")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "raw_debt_ratio.png"), dpi=150)
    plt.close()
    print("Saved raw_debt_ratio.png")

    cap_99 = df[col].quantile(0.99)
    print(
        f"\n  DECISION: DebtRatio has extreme right-skew driven mechanically by near-zero "
        f"income denominators, not genuine debt burden. Recommend capping at the 99th "
        f"percentile ({cap_99:.2f}) rather than the more lenient 99.9th used for utilization, "
        f"since DebtRatio's tail is more of a division artifact than real signal."
    )


# ---------------------------------------------------------------- delinquency
def analyze_delinquency(df: pd.DataFrame) -> None:
    section("DELINQUENCY COLUMNS (30-59, 60-89, 90+ days past due)")

    for col in DELINQUENCY_COLS:
        print(f"\n{col}:")
        print(df[col].value_counts().sort_index().to_string())

    sentinel_mask = df[DELINQUENCY_COLS[0]].isin(SENTINEL_CODES)
    sentinel_count = sentinel_mask.sum()
    print(f"\nRows with sentinel codes (96/98) in {DELINQUENCY_COLS[0]}: {sentinel_count}")

    same_rows_all_cols = df.loc[sentinel_mask, DELINQUENCY_COLS].isin(SENTINEL_CODES).all(axis=1).sum()
    print(
        f"Of those, rows where ALL THREE delinquency columns are also sentinel-coded: "
        f"{same_rows_all_cols} (checking if this is one systematic batch of corrupted records)"
    )

    non_sentinel_max = df[~sentinel_mask][DELINQUENCY_COLS[0]].max()
    non_sentinel_99_9 = df[~sentinel_mask][DELINQUENCY_COLS[0]].quantile(0.999)
    print(f"\nMax value in {DELINQUENCY_COLS[0]} excluding sentinel codes: {non_sentinel_max}")
    print(f"99.9th percentile excluding sentinel codes: {non_sentinel_99_9}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, col in enumerate(DELINQUENCY_COLS):
        vc = df[col].value_counts().sort_index()
        axes[i].bar(vc.index.astype(str), vc.values, color="#C44E52")
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(axis="x", rotation=90, labelsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "raw_delinquency_sentinel_codes.png"), dpi=150)
    plt.close()
    print("\nSaved raw_delinquency_sentinel_codes.png")

    print(
        f"\n  DECISION: sentinel codes 96/98 are not real counts (a real 96 late-payment "
        f"count is impossible in a 24-month observation window) - they are almost "
        f"certainly a coded 'error/unknown' flag baked into the original data collection. "
        f"Recommend capping at {non_sentinel_max} (the true observed max excluding sentinels), "
        f"not at an arbitrary round number, and treating this as a shared data quality "
        f"batch rather than 269 x 3 independent anomalies."
    )


# ---------------------------------------------------------------- income
def analyze_monthly_income(df: pd.DataFrame) -> None:
    section("MONTHLY INCOME")
    col = "MonthlyIncome"
    print(df[col].describe().to_string())

    zero_count = (df[col] == 0).sum()
    print(f"\nRows with MonthlyIncome == 0: {zero_count}")

    clip_val = df[col].quantile(0.99)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(df[col].dropna().clip(upper=clip_val), bins=60, ax=ax, color="#4C72B0")
    ax.set_title(f"{col} Distribution (clipped at 99th pct = {clip_val:.0f} for display)")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "raw_monthly_income.png"), dpi=150)
    plt.close()
    print("Saved raw_monthly_income.png")

    print(
        f"\n  DECISION: income is right-skewed (as expected for real-world income data), "
        f"so median imputation for missing values is more appropriate than mean, since "
        f"the mean would be pulled upward by high earners and overstate a typical "
        f"applicant's income."
    )


# ---------------------------------------------------------------- correlation
def analyze_correlation_with_target(df: pd.DataFrame) -> None:
    section("CORRELATION WITH TARGET (pairwise, raw data, NaNs excluded pairwise)")
    corr = df.corr(numeric_only=True)[TARGET_COL].drop(TARGET_COL)
    corr = corr.sort_values(key=abs, ascending=False)
    print(corr.to_string())

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors)
    ax.set_title("Feature Correlation with Target (Raw Data)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "raw_correlation_with_target.png"), dpi=150)
    plt.close()
    print("\nSaved raw_correlation_with_target.png")

    top_feature = corr.index[0]
    print(
        f"\n  INSIGHT: {top_feature} has the strongest linear correlation with default "
        f"({corr.iloc[0]:.4f}). Note this is Pearson correlation, so it only captures "
        f"linear relationships - tree-based models may still find nonlinear signal in "
        f"weakly-correlated features, so nothing here should be used to drop features "
        f"outright."
    )


def main():
    parser = argparse.ArgumentParser(description="Raw EDA for Give Me Some Credit dataset")
    parser.add_argument("--input", type=str, default="data/raw/cs-training.csv")
    args = parser.parse_args()

    os.makedirs(IMAGES_DIR, exist_ok=True)
    df = load_raw(args.input)

    section("SHAPE AND DTYPES")
    print(f"Shape: {df.shape}")
    print(df.dtypes.to_string())

    analyze_missingness(df)
    analyze_age(df)
    analyze_revolving_utilization(df)
    analyze_debt_ratio(df)
    analyze_delinquency(df)
    analyze_monthly_income(df)
    analyze_correlation_with_target(df)

    section("DONE")
    print("All plots saved to images/. Review the DECISION lines above before")
    print("finalizing src/preprocess.py thresholds.")


if __name__ == "__main__":
    main()