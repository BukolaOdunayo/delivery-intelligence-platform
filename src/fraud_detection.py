"""
Module 4 — Fraud / Anomaly Detection.

This dataset has no labeled fraud, so this module frames the realistic
operational version of the problem: unsupervised anomaly detection over
engineered "implausibility" signals, the same pattern used for delivery
fraud (fake GPS, impossible timings, rating manipulation) in production
logistics systems. Isolation Forest scores every order; a rule-based
layer explains *why* the top anomalies were flagged (important for any
fraud system — a black-box score alone isn't actionable for an ops team).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


ANOMALY_FEATURES = [
    "implied_speed_kmh", "distance_km", "prep_time_min",
    "Delivery_person_Ratings", "Time_taken(min)", "multiple_deliveries",
]


def engineer_fraud_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # implied average speed for the trip (km per hour) — the core "is this physically plausible" signal
    df["implied_speed_kmh"] = df["distance_km"] / (df["Time_taken(min)"] / 60)
    return df


def detect_anomalies(df: pd.DataFrame, contamination=0.02, models_dir="models"):
    df = engineer_fraud_signals(df)
    X = df[ANOMALY_FEATURES].fillna(df[ANOMALY_FEATURES].median())

    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=42, n_jobs=-1)
    model.fit(X)
    df["anomaly_score"] = -model.decision_function(X)  # higher = more anomalous
    df["is_anomaly"] = model.predict(X) == -1

    # rule-based explanations for interpretability
    def explain(row):
        reasons = []
        if row["implied_speed_kmh"] > 45:
            reasons.append(f"implausible speed ({row['implied_speed_kmh']:.0f} km/h)")
        if row["prep_time_min"] <= 5 and row["distance_km"] > 15:
            reasons.append("near-zero prep time on a long-distance order")
        if row["Delivery_person_Ratings"] >= 4.9 and row["Time_taken(min)"] > df["Time_taken(min)"].quantile(0.95):
            reasons.append("top-rated partner with abnormally long delivery time")
        if row["multiple_deliveries"] >= 3 and row["Time_taken(min)"] < df["Time_taken(min)"].quantile(0.10):
            reasons.append("many simultaneous deliveries but unusually fast completion")
        return "; ".join(reasons) if reasons else "flagged by model (no single rule matched)"

    df["flag_reason"] = df.apply(explain, axis=1)

    import joblib, os
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, f"{models_dir}/fraud_isolation_forest.pkl")

    return df, model


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/train_clean.parquet")
    scored, model = detect_anomalies(df)
    top_anomalies = scored[scored["is_anomaly"]].sort_values("anomaly_score", ascending=False)
    print(f"Flagged {scored['is_anomaly'].sum()} / {len(scored)} orders ({scored['is_anomaly'].mean()*100:.1f}%)")
    print(top_anomalies[["ID", "implied_speed_kmh", "distance_km", "Time_taken(min)", "flag_reason"]].head(10).to_string())
    scored.to_parquet("data/processed/train_with_fraud_scores.parquet")
