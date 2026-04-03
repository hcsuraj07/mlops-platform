# =============================================================================
# monitor.py
# PURPOSE: Detect when incoming prediction data drifts away from training data.
#
# WHY DOES DRIFT MATTER?
#   Your model was trained on 2005 Taiwan credit data. Six months later,
#   an economic crisis hits — people's payment behavior changes completely.
#   Your model still runs fine technically, but its predictions are now wrong
#   because the real world no longer looks like your training data.
#   This is called DATA DRIFT — and it's the #1 silent killer of production
#   ML models. Most companies don't detect it until business metrics tank.
#
# TWO DRIFT DETECTION METHODS:
#
#   1. PSI (Population Stability Index)
#      Compares the distribution of a feature between training and new data.
#      Think of it like comparing two histograms — if they look the same,
#      PSI is low. If they look very different, PSI is high.
#      PSI < 0.1  → no drift (stable)
#      PSI 0.1-0.2 → moderate drift (monitor closely)
#      PSI > 0.2  → severe drift (retrain immediately)
#
#   2. KS Test (Kolmogorov-Smirnov Test)
#      A statistical test that asks: "could these two samples have come
#      from the same distribution?" Returns a p-value. If p < 0.05,
#      the distributions are statistically significantly different.
#      This is more rigorous than PSI — it has a mathematical guarantee.
#
# FLOW:
#   Load training data (reference) → load new incoming data (current)
#   → compute PSI for each feature → run KS test for each feature
#   → flag drifted features → log results to MLflow → save report
# =============================================================================

import pandas as pd
import numpy as np
import mlflow
import json
import os
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("mlruns")


# -----------------------------------------------------------------------------
# FUNCTION 1: compute_psi
#
# PSI measures how much a feature's distribution has shifted.
#
# HOW IT WORKS:
#   1. Bin the reference data into 10 buckets (like a histogram)
#   2. Count what % of reference data falls in each bucket
#   3. Count what % of current data falls in each bucket
#   4. For each bucket: PSI += (current% - reference%) * ln(current%/reference%)
#
# The formula penalizes large shifts in any bucket. If 40% of training
# data had credit_limit between $10k-$20k, but only 5% of new data does,
# that bucket contributes a large PSI value.
#
# WHY 10 BINS?
#   Standard practice. Fewer bins miss subtle shifts, more bins are noisy
#   with small samples. 10 is the industry default for PSI.
# -----------------------------------------------------------------------------
def compute_psi(reference: np.ndarray, current: np.ndarray,
                n_bins: int = 10) -> float:
    # create bins based on reference data distribution
    # we use reference percentiles so bins are evenly populated
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)  # remove duplicates for constant features

    if len(breakpoints) < 2:
        return 0.0  # constant feature — no drift possible

    # count % of data in each bin
    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    # convert to proportions (must sum to 1)
    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    # avoid division by zero or log(0) — replace zeros with small epsilon
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)

    # PSI formula
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


# -----------------------------------------------------------------------------
# FUNCTION 2: compute_ks_test
#
# KS test asks: "are these two samples from the same distribution?"
#
# HOW IT WORKS:
#   Computes the maximum difference between the two empirical CDFs
#   (cumulative distribution functions). If the max difference is large
#   relative to sample size, the distributions are significantly different.
#
# RETURNS:
#   statistic — the maximum CDF difference (0=identical, 1=completely different)
#   p_value   — probability of seeing this difference by chance
#               p < 0.05 means drift is statistically significant
# -----------------------------------------------------------------------------
def compute_ks_test(reference: np.ndarray,
                    current: np.ndarray) -> tuple[float, float]:
    statistic, p_value = stats.ks_2samp(reference, current)
    return float(statistic), float(p_value)


# -----------------------------------------------------------------------------
# FUNCTION 3: simulate_drifted_data
#
# In production, "current" data comes from your prediction logs.
# For this demo, we simulate drift by shifting key features.
#
# We simulate an economic downturn scenario:
#   - Payment delays increase (people struggling to pay)
#   - Bill amounts increase (more spending / inflation)
#   - Payment amounts decrease (less money to pay bills)
#
# WHY SIMULATE? We don't have real production data yet. But the monitoring
# code works identically whether data is simulated or real. In production
# you'd replace simulate_drifted_data() with a function that reads from
# your prediction logs database.
# -----------------------------------------------------------------------------
def simulate_drifted_data(reference_df: pd.DataFrame,
                           n_samples: int = 1000) -> pd.DataFrame:
    # sample from reference data as a starting point
    current = reference_df.sample(n=n_samples, random_state=99).copy()

    # simulate economic downturn drift
    # payment delays increase by 1-2 months on average
    payment_cols = ["payment_status_sep", "payment_status_aug",
                    "payment_status_jul", "payment_status_jun",
                    "payment_status_may", "payment_status_apr"]
    for col in payment_cols:
        current[col] = current[col] + np.random.randint(1, 3, size=len(current))

    # bill amounts increase 20% (inflation)
    bill_cols = ["bill_amt_sep", "bill_amt_aug", "bill_amt_jul",
                 "bill_amt_jun", "bill_amt_may", "bill_amt_apr"]
    for col in bill_cols:
        current[col] = current[col] * 1.2

    # payment amounts decrease 30% (less disposable income)
    payment_amt_cols = ["payment_amt_sep", "payment_amt_aug", "payment_amt_jul",
                        "payment_amt_jun", "payment_amt_may", "payment_amt_apr"]
    for col in payment_amt_cols:
        current[col] = current[col] * 0.7

    return current


# -----------------------------------------------------------------------------
# FUNCTION 4: run_drift_detection
#
# Main function — runs PSI and KS test on all features and logs to MLflow.
#
# WHAT GETS LOGGED:
#   - PSI score per feature
#   - KS statistic + p-value per feature
#   - Number of drifted features
#   - Overall drift status (stable / warning / critical)
#   - Full drift report as JSON artifact
# -----------------------------------------------------------------------------
def run_drift_detection(reference_path: str = "data/raw/credit_default.csv"):
    print("Running drift detection...")

    # load reference (training) data
    ref_df = pd.read_csv(reference_path)

    # rename columns to match train.py
    column_map = {
        "X1": "credit_limit", "X2": "gender", "X3": "education",
        "X4": "marriage", "X5": "age",
        "X6": "payment_status_sep", "X7": "payment_status_aug",
        "X8": "payment_status_jul", "X9": "payment_status_jun",
        "X10": "payment_status_may", "X11": "payment_status_apr",
        "X12": "bill_amt_sep", "X13": "bill_amt_aug",
        "X14": "bill_amt_jul", "X15": "bill_amt_jun",
        "X16": "bill_amt_may", "X17": "bill_amt_apr",
        "X18": "payment_amt_sep", "X19": "payment_amt_aug",
        "X20": "payment_amt_jul", "X21": "payment_amt_jun",
        "X22": "payment_amt_may", "X23": "payment_amt_apr",
    }
    ref_df = ref_df.rename(columns=column_map)

    # drop target column — we only monitor features, not labels
    feature_cols = [c for c in ref_df.columns if c != "default"]
    ref_features = ref_df[feature_cols]

    # simulate incoming production data with drift
    current_features = simulate_drifted_data(ref_features)

    print(f"Reference samples: {len(ref_features)}")
    print(f"Current samples:   {len(current_features)}")

    # run drift detection on each feature
    results = []
    drifted_features = []

    for col in feature_cols:
        ref_vals = ref_features[col].values.astype(float)
        cur_vals = current_features[col].values.astype(float)

        psi = compute_psi(ref_vals, cur_vals)
        ks_stat, ks_pval = compute_ks_test(ref_vals, cur_vals)

        # flag as drifted if PSI > 0.2 OR KS p-value < 0.05
        is_drifted = psi > 0.2 or ks_pval < 0.05

        if is_drifted:
            drifted_features.append(col)

        results.append({
            "feature": col,
            "psi": round(psi, 4),
            "ks_statistic": round(ks_stat, 4),
            "ks_pvalue": round(ks_pval, 4),
            "drifted": is_drifted,
            "psi_status": "critical" if psi > 0.2 else
                          "warning" if psi > 0.1 else "stable"
        })

    # determine overall drift status
    n_drifted = len(drifted_features)
    if n_drifted == 0:
        overall_status = "stable"
    elif n_drifted <= 3:
        overall_status = "warning"
    else:
        overall_status = "critical — retrain recommended"

    # print summary
    print(f"\n{'='*50}")
    print("DRIFT DETECTION RESULTS")
    print(f"{'='*50}")
    print(f"Features monitored: {len(feature_cols)}")
    print(f"Drifted features:   {n_drifted}")
    print(f"Overall status:     {overall_status}")
    print(f"\nDrifted features:")
    for r in results:
        if r["drifted"]:
            print(f"  {r['feature']:30} PSI={r['psi']:.3f} "
                  f"({r['psi_status']}) | "
                  f"KS p={r['ks_pvalue']:.4f}")

    # log to MLflow
    with mlflow.start_run(run_name="drift-detection"):
        mlflow.set_tag("run_type", "drift_detection")
        mlflow.log_metric("n_drifted_features", n_drifted)
        mlflow.log_metric("drift_rate",
                          round(n_drifted / len(feature_cols), 3))

        # log PSI for each feature
        for r in results:
            mlflow.log_metric(f"psi_{r['feature']}", r["psi"])
            mlflow.log_metric(f"ks_pval_{r['feature']}", r["ks_pvalue"])

        mlflow.set_tag("overall_status", overall_status)

        # save full report as artifact
        os.makedirs("notebooks", exist_ok=True)
        report_path = "notebooks/drift_report.json"
        with open(report_path, "w") as f:
            json.dump({
                "overall_status": overall_status,
                "n_drifted": n_drifted,
                "drifted_features": drifted_features,
                "feature_results": results
            }, f, indent=2)

        mlflow.log_artifact(report_path)
        print(f"\nDrift report saved → {report_path}")
        print("Results logged to MLflow")

    return results, overall_status


if __name__ == "__main__":
    results, status = run_drift_detection()
    print(f"\nFinal status: {status}")