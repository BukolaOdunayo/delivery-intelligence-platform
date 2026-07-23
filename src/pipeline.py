"""
Single-command pipeline: raw CSVs -> clean data -> features -> every
trained model artifact. This is the "operational automation" layer —
the same steps a CI job would run on a schedule or on a data-drift
trigger from monitoring.py.

Usage:
    python src/pipeline.py
"""
import time
import pandas as pd

from data_prep import clean_raw, add_core_features
from eta_model import train_eta_models
from forecasting import build_daily_demand, train_forecast_model
from recommendation import build_partner_profiles
from fraud_detection import detect_anomalies
from monitoring import check_drift


def run(train_raw="data/train_raw.csv", test_raw="data/test_raw.csv", models_dir="models"):
    t0 = time.time()
    log = []

    def step(msg):
        elapsed = time.time() - t0
        log.append(f"[{elapsed:6.1f}s] {msg}")
        print(log[-1])

    step("Cleaning raw data...")
    train = add_core_features(clean_raw(train_raw, is_train=True))
    test = add_core_features(clean_raw(test_raw, is_train=False))
    train.to_parquet("data/processed/train_clean.parquet")
    test.to_parquet("data/processed/test_clean.parquet")

    step("Training ETA prediction model (module 1/5)...")
    eta_results, best_name, _, _ = train_eta_models(train, models_dir)
    step(f"  -> best model: {best_name}, MAE={eta_results.iloc[0]['MAE']:.2f}")

    step("Training demand forecasting model (module 2/5)...")
    daily = build_daily_demand(train)
    fc_out = train_forecast_model(daily, models_dir=models_dir)
    step(f"  -> forecast MAE={fc_out['mae']:.1f} orders/day vs naive={fc_out['naive_mae']:.1f}")

    step("Building partner profiles for recommendation engine (module 3/5)...")
    profiles = build_partner_profiles(train)
    profiles.to_csv(f"{models_dir}/partner_profiles.csv", index=False)
    step(f"  -> {len(profiles)} partner profiles built")

    step("Scoring fraud/anomaly detection model (module 4/5)...")
    scored, _ = detect_anomalies(train, models_dir=models_dir)
    step(f"  -> flagged {scored['is_anomaly'].sum()} anomalous orders ({scored['is_anomaly'].mean()*100:.1f}%)")

    step("Running data drift check against holdout test set (module 5/5, monitoring)...")
    report, any_drift = check_drift(train, test)
    report.to_csv(f"{models_dir}/drift_report.csv", index=False)
    step(f"  -> retraining trigger: {'YES' if any_drift else 'no'}")

    step("Pipeline complete. All artifacts written to models/ and data/processed/.")
    return log


if __name__ == "__main__":
    run()
