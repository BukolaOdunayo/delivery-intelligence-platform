"""
Module 5 — Logistics Optimization (batch order-to-partner assignment).

Given a batch of simultaneously-pending orders and a pool of available
delivery partners, use the ETA model to build a predicted-time cost
matrix (partner x order), then solve the assignment problem optimally
with the Hungarian algorithm (scipy.optimize.linear_sum_assignment) to
minimize total fleet delivery time. Compared against a naive
first-come-first-served / nearest-random baseline to quantify the
operational lift — the standard framing for a dispatch optimization
system.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def build_cost_matrix(orders: pd.DataFrame, partners: pd.DataFrame, eta_model, feature_columns):
    """orders: rows with order-level context columns.
    partners: rows with avg_age, avg_rating, avg_vehicle_condition, primary_vehicle.
    Returns an (n_partners x n_orders) predicted-ETA matrix.
    """
    n_o, n_p = len(orders), len(partners)
    records = []
    for _, partner in partners.iterrows():
        for _, order in orders.iterrows():
            row = order.to_dict()
            row["Delivery_person_Age"] = partner["avg_age"]
            row["Delivery_person_Ratings"] = partner["avg_rating"]
            row["Vehicle_condition"] = partner["avg_vehicle_condition"]
            row["Type_of_vehicle"] = partner["primary_vehicle"]
            records.append(row)

    cand_df = pd.DataFrame(records)
    cat_cols = ["Type_of_order", "Type_of_vehicle", "City"]
    X = pd.get_dummies(cand_df, columns=cat_cols, drop_first=True)
    X = X.reindex(columns=feature_columns, fill_value=0)

    preds = eta_model.predict(X)
    cost_matrix = preds.reshape(n_p, n_o)
    return cost_matrix


def optimize_assignment(cost_matrix: np.ndarray):
    """Returns (partner_idx, order_idx) optimal pairs and total predicted minutes."""
    n_p, n_o = cost_matrix.shape
    # pad to square if unequal (dummy partners/orders with 0 cost get ignored downstream)
    size = max(n_p, n_o)
    padded = np.full((size, size), cost_matrix.max() * 2)
    padded[:n_p, :n_o] = cost_matrix
    row_idx, col_idx = linear_sum_assignment(padded)

    pairs = [(r, c) for r, c in zip(row_idx, col_idx) if r < n_p and c < n_o]
    total_time = sum(cost_matrix[r, c] for r, c in pairs)
    return pairs, total_time


def naive_assignment(cost_matrix: np.ndarray, seed=0):
    """Baseline: assign orders to partners in the order they arrive
    (first available partner gets the next order) — mirrors a simple
    FCFS dispatch queue with no optimization."""
    n_p, n_o = cost_matrix.shape
    rng = np.random.default_rng(seed)
    partner_order = rng.permutation(n_p)
    pairs = []
    total_time = 0
    for i in range(min(n_p, n_o)):
        p, o = partner_order[i], i
        pairs.append((p, o))
        total_time += cost_matrix[p, o]
    return pairs, total_time


if __name__ == "__main__":
    import joblib, json
    from recommendation import build_partner_profiles

    df = pd.read_parquet("data/processed/train_clean.parquet")
    profiles = build_partner_profiles(df)
    eta_model = joblib.load("models/eta_model.pkl")
    feature_columns = json.load(open("models/eta_feature_columns.json"))

    order_cols = [
        "distance_km", "order_hour", "order_dow", "is_weekend", "is_rush_hour",
        "prep_time_min", "traffic_ordinal", "distance_traffic_interaction",
        "bad_weather", "is_festival", "multiple_deliveries", "Type_of_order", "City",
    ]

    improvements = []
    for trial in range(20):
        orders_batch = df[order_cols].sample(15, random_state=trial).reset_index(drop=True)
        partners_batch = profiles.sample(15, random_state=trial).reset_index(drop=True)
        cost = build_cost_matrix(orders_batch, partners_batch, eta_model, feature_columns)
        _, opt_total = optimize_assignment(cost)
        _, naive_total = naive_assignment(cost, seed=trial)
        improvements.append(float((naive_total - opt_total) / naive_total * 100))

    print(f"Avg improvement over 20 batches of 15 orders/15 partners: {np.mean(improvements):.1f}% "
          f"(range {np.min(improvements):.1f}%-{np.max(improvements):.1f}%)")

    import json as _json
    with open("models/logistics_optimization_results.json", "w") as f:
        _json.dump({"trials": improvements, "mean_improvement_pct": float(np.mean(improvements))}, f)
