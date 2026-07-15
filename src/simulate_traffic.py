"""
Simulates incoming API traffic across several labeled time windows, some
matching the training distribution (should show no drift) and some
deliberately shifted (should trigger drift detection). Calls the real
/predict endpoint for every record, so this exercises the actual serving
pipeline - it is not a shortcut around the API.

IMPORTANT: this is simulated traffic for demonstrating the drift monitoring
system, not real user activity. Said explicitly here and in the README -
the point is to validate the PSI detector correctly stays quiet on stable
windows and correctly fires on shifted ones.

Windows generated:
    week1_baseline, week2_baseline  - sampled from the real test set,
        same distribution as training. Drift detector should NOT fire.
    week3_drift_younger_segment      - applicant age shifted younger
        (simulating a new marketing channel attracting younger customers)
    week4_drift_income_drop          - MonthlyIncome shifted down
        (simulating a macroeconomic downturn)
    week5_drift_utilization_spike    - RevolvingUtilizationOfUnsecuredLines
        shifted up (simulating customers under increasing credit stress)

Run:
    python src/simulate_traffic.py --api-url http://localhost:8000
"""

import argparse
import time

import numpy as np
import pandas as pd
import requests

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


def sample_baseline(test_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return test_df.sample(n=n, random_state=seed)[RAW_FEATURE_COLUMNS].reset_index(drop=True)


def sample_drift_younger(test_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    df = test_df.sample(n=n, random_state=seed)[RAW_FEATURE_COLUMNS].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    # Shift age distribution down by ~15 years on average, floor at 18
    df["age"] = (df["age"] - rng.integers(10, 20, size=n)).clip(lower=18)
    return df


def sample_drift_income_drop(test_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    df = test_df.sample(n=n, random_state=seed)[RAW_FEATURE_COLUMNS].reset_index(drop=True)
    # Simulate a macroeconomic income shock - incomes drop 30-50%
    rng = np.random.default_rng(seed)
    shock_factor = rng.uniform(0.5, 0.7, size=n)
    df["MonthlyIncome"] = (df["MonthlyIncome"] * shock_factor).round(2)
    return df


def sample_drift_utilization_spike(test_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    df = test_df.sample(n=n, random_state=seed)[RAW_FEATURE_COLUMNS].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    spike_factor = rng.uniform(1.8, 2.5, size=n)
    df["RevolvingUtilizationOfUnsecuredLines"] = (
        df["RevolvingUtilizationOfUnsecuredLines"] * spike_factor
    ).clip(upper=2.0).round(4)
    return df


WINDOWS = {
    "week1_baseline": sample_baseline,
    "week2_baseline": sample_baseline,
    "week3_drift_younger_segment": sample_drift_younger,
    "week4_drift_income_drop": sample_drift_income_drop,
    "week5_drift_utilization_spike": sample_drift_utilization_spike,
}


def send_window(df: pd.DataFrame, window_label: str, api_url: str) -> int:
    success_count = 0
    for _, row in df.iterrows():
        payload = row.where(pd.notnull(row), None).to_dict()
        try:
            resp = requests.post(f"{api_url}/predict", json=payload, params={"window": window_label}, timeout=5)
            if resp.status_code == 200:
                success_count += 1
            else:
                print(f"  Request failed ({resp.status_code}): {resp.text[:200]}")
        except requests.exceptions.RequestException as e:
            print(f"  Request error: {e}")
    return success_count


def main():
    parser = argparse.ArgumentParser(description="Simulate incoming API traffic for drift testing")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000")
    parser.add_argument("--test-path", type=str, default="data/processed/test.csv")
    parser.add_argument("--n-per-window", type=int, default=200)
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_path)
    print(f"Loaded test set for sampling: {test_df.shape}")

    for i, (window_label, sample_fn) in enumerate(WINDOWS.items()):
        df = sample_fn(test_df, args.n_per_window, seed=100 + i)
        print(f"\nSending window '{window_label}' ({len(df)} requests)...")
        start = time.time()
        success_count = send_window(df, window_label, args.api_url)
        elapsed = time.time() - start
        print(f"  {success_count}/{len(df)} succeeded in {elapsed:.1f}s")

    print("\nDone. Simulated traffic logged to logs/requests.jsonl")


if __name__ == "__main__":
    main()