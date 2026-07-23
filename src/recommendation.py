"""
Module 3 — Delivery-Partner Recommendation / Matching Engine.

Given a new order's context (distance, traffic, weather, time of day),
rank a pool of candidate delivery partners by *predicted* performance
using the ETA model as a scoring function, then blend in a reliability
prior from each partner's track record. This is the recommendation-
systems angle applied to dispatch: "who should take this order?"
rather than "what should this user watch?" — same ranking mechanics.
"""
import numpy as np
import pandas as pd


def build_partner_profiles(df: pd.DataFrame) -> pd.DataFrame:
    profiles = (
        df.groupby("Delivery_person_ID")
        .agg(
            avg_age=("Delivery_person_Age", "mean"),
            avg_rating=("Delivery_person_Ratings", "mean"),
            avg_vehicle_condition=("Vehicle_condition", "mean"),
            primary_vehicle=("Type_of_vehicle", lambda s: s.mode().iloc[0]),
            n_deliveries=("ID", "count"),
            avg_time_taken=("Time_taken(min)", "mean"),
        )
        .reset_index()
    )
    # experience-adjusted reliability score (0-1): blends rating + volume
    max_n = profiles["n_deliveries"].max()
    profiles["experience_score"] = (profiles["n_deliveries"] / max_n).clip(upper=1)
    profiles["reliability_score"] = (
        0.7 * (profiles["avg_rating"] / 5) + 0.3 * profiles["experience_score"]
    )
    return profiles


def recommend_partners(order_context: dict, candidate_partners: pd.DataFrame,
                        eta_model, feature_columns, top_k=5):
    """order_context: dict with distance_km, traffic_ordinal, order_hour, order_dow,
    is_weekend, is_rush_hour, prep_time_min, distance_traffic_interaction,
    bad_weather, is_festival, Type_of_order, City (order-level, fixed).
    candidate_partners: rows from partner profiles (avg_age, avg_rating,
    avg_vehicle_condition, primary_vehicle, reliability_score).
    Returns candidates ranked by a blended score (lower predicted ETA + higher reliability).
    """
    rows = []
    for _, p in candidate_partners.iterrows():
        row = dict(order_context)
        row["Delivery_person_Age"] = p["avg_age"]
        row["Delivery_person_Ratings"] = p["avg_rating"]
        row["Vehicle_condition"] = p["avg_vehicle_condition"]
        row["Type_of_vehicle"] = p["primary_vehicle"]
        rows.append(row)
    cand_df = pd.DataFrame(rows)

    cat_cols = ["Type_of_order", "Type_of_vehicle", "City"]
    X = pd.get_dummies(cand_df, columns=cat_cols, drop_first=True)
    X = X.reindex(columns=feature_columns, fill_value=0)

    predicted_eta = eta_model.predict(X)

    result = candidate_partners.copy().reset_index(drop=True)
    result["predicted_eta_min"] = predicted_eta
    # blended dispatch score: normalize ETA (lower=better) and reliability (higher=better)
    eta_norm = 1 - (result["predicted_eta_min"] - result["predicted_eta_min"].min()) / (
        result["predicted_eta_min"].max() - result["predicted_eta_min"].min() + 1e-9
    )
    result["dispatch_score"] = 0.6 * eta_norm + 0.4 * result["reliability_score"]
    result = result.sort_values("dispatch_score", ascending=False).reset_index(drop=True)
    return result.head(top_k)


if __name__ == "__main__":
    import joblib, json
    df = pd.read_parquet("data/processed/train_clean.parquet")
    profiles = build_partner_profiles(df)
    profiles.to_csv("models/partner_profiles.csv", index=False)

    eta_model = joblib.load("models/eta_model.pkl")
    feature_columns = json.load(open("models/eta_feature_columns.json"))

    sample_order = {
        "distance_km": 8.5, "order_hour": 19, "order_dow": 4, "is_weekend": 0,
        "is_rush_hour": 1, "prep_time_min": 10, "traffic_ordinal": 2,
        "distance_traffic_interaction": 17.0, "bad_weather": 0, "is_festival": 0,
        "multiple_deliveries": 1, "Type_of_order": "Snack", "City": "Metropolitian",
    }
    pool = profiles.sample(20, random_state=1)
    top = recommend_partners(sample_order, pool, eta_model, feature_columns, top_k=5)
    print(top[["Delivery_person_ID", "avg_rating", "n_deliveries", "predicted_eta_min", "dispatch_score"]])
