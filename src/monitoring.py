"""
Module 6 — Operational Automation & Monitoring.

Two things a model needs once it's in production and nobody's watching
a notebook anymore: (1) a single command that re-runs the whole
pipeline end-to-end (data -> features -> models -> artifacts), and
(2) a drift check that flags when incoming data has quietly drifted
away from what the model was trained on, so retraining gets triggered
before accuracy silently degrades.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

DRIFT_FEATURES = [
    "distance_km", "Delivery_person_Ratings", "Delivery_person_Age",
    "prep_time_min", "traffic_ordinal",
]


def check_drift(reference_df: pd.DataFrame, incoming_df: pd.DataFrame, alpha=0.01):
    """Kolmogorov-Smirnov two-sample test per feature. Returns a report
    flagging any feature whose distribution has shifted significantly
    vs. the training reference — the trigger condition a retraining
    job would watch for."""
    report = []
    for col in DRIFT_FEATURES:
        ref = reference_df[col].dropna()
        inc = incoming_df[col].dropna()
        stat, p_value = ks_2samp(ref, inc)
        report.append({
            "feature": col,
            "ks_statistic": stat,
            "p_value": p_value,
            "drift_detected": p_value < alpha,
            "reference_mean": ref.mean(),
            "incoming_mean": inc.mean(),
        })
    report_df = pd.DataFrame(report)
    any_drift = report_df["drift_detected"].any()
    return report_df, any_drift


if __name__ == "__main__":
    train = pd.read_parquet("data/processed/train_clean.parquet")
    test = pd.read_parquet("data/processed/test_clean.parquet")

    report, any_drift = check_drift(train, test)
    print(report.to_string(index=False))
    print(f"\nRetraining trigger: {'YES — drift detected' if any_drift else 'no drift detected'}")
    report.to_csv("models/drift_report.csv", index=False)
