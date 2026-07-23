"""
Berlin Food Delivery — ETA & Ops Intelligence Platform
Interactive demo tying together all 6 modules built on the Kaggle
Food Delivery Dataset: ETA prediction, demand forecasting, partner
recommendation, fraud detection, logistics optimization, and monitoring.

Run with:  streamlit run app/streamlit_app.py
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from recommendation import recommend_partners
from logistics_optimization import build_cost_matrix, optimize_assignment, naive_assignment
from monitoring import check_drift, DRIFT_FEATURES

ROOT = os.path.join(os.path.dirname(__file__), "..")

st.set_page_config(page_title="Delivery Intelligence Platform", layout="wide", page_icon="🛵")


@st.cache_resource
def load_artifacts():
    eta_model = joblib.load(f"{ROOT}/models/eta_model.pkl")
    feature_columns = json.load(open(f"{ROOT}/models/eta_feature_columns.json"))
    profiles = pd.read_csv(f"{ROOT}/models/partner_profiles.csv")
    eta_comparison = pd.read_csv(f"{ROOT}/models/eta_model_comparison.csv")
    return eta_model, feature_columns, profiles, eta_comparison


@st.cache_data
def load_data():
    train = pd.read_parquet(f"{ROOT}/data/processed/train_clean.parquet")
    test = pd.read_parquet(f"{ROOT}/data/processed/test_clean.parquet")
    fraud_scored = pd.read_parquet(f"{ROOT}/data/processed/train_with_fraud_scores.parquet")
    return train, test, fraud_scored


eta_model, feature_columns, profiles, eta_comparison = load_artifacts()
train, test, fraud_scored = load_data()

st.title("🛵 Delivery Intelligence Platform")
st.caption(
    "Built on the Kaggle Food Delivery Dataset (45,593 orders). "
    "One ETA model powers ETA prediction, partner recommendation, and logistics optimization; "
    "two additional models handle demand forecasting and fraud detection."
)

tab_eta, tab_forecast, tab_rec, tab_fraud, tab_opt, tab_monitor = st.tabs(
    ["📦 ETA Prediction", "📈 Demand Forecasting", "🎯 Partner Recommendation",
     "🚩 Fraud Detection", "🧮 Logistics Optimization", "🩺 Monitoring"]
)

# ---------------------------------------------------------------- ETA
with tab_eta:
    st.subheader("Predict delivery time for a new order")
    col1, col2 = st.columns([1, 1.3])
    with col1:
        distance_km = st.slider("Distance (km)", 1.0, 30.0, 8.5)
        traffic = st.selectbox("Traffic density", ["Low", "Medium", "High", "Jam"], index=2)
        weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Windy", "Fog", "Stormy", "Sandstorms"])
        order_hour = st.slider("Order hour (0-23)", 0, 23, 19)
        multiple_deliveries = st.slider("Concurrent deliveries for this partner", 0, 3, 1)
        vehicle = st.selectbox("Vehicle type", ["motorcycle", "scooter", "electric_scooter", "bicycle"])
        order_type = st.selectbox("Order type", ["Snack", "Meal", "Drinks", "Buffet"])
        city = st.selectbox("City type", ["Metropolitian", "Urban", "Semi-Urban"])
        partner_rating = st.slider("Delivery partner rating", 1.0, 5.0, 4.6, 0.1)
        partner_age = st.slider("Delivery partner age", 18, 50, 30)
        vehicle_condition = st.select_slider("Vehicle condition (0=worst, 2=best)", options=[0, 1, 2], value=1)

        traffic_map = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}
        row = {
            "Delivery_person_Age": partner_age, "Delivery_person_Ratings": partner_rating,
            "Vehicle_condition": vehicle_condition, "multiple_deliveries": multiple_deliveries,
            "distance_km": distance_km, "order_hour": order_hour, "order_dow": 3,
            "is_weekend": 0, "is_rush_hour": int(order_hour in [8, 9, 12, 13, 18, 19, 20]),
            "prep_time_min": 10, "traffic_ordinal": traffic_map[traffic],
            "distance_traffic_interaction": distance_km * traffic_map[traffic],
            "bad_weather": int(weather in ["Stormy", "Sandstorms", "Fog"]), "is_festival": 0,
            "Type_of_order": order_type, "Type_of_vehicle": vehicle, "City": city,
        }
        X = pd.get_dummies(pd.DataFrame([row]), columns=["Type_of_order", "Type_of_vehicle", "City"], drop_first=True)
        X = X.reindex(columns=feature_columns, fill_value=0)
        pred = eta_model.predict(X)[0]

        st.metric("Predicted delivery time", f"{pred:.1f} min")

    with col2:
        st.markdown("**Model comparison (held-out validation set)**")
        fig = px.bar(eta_comparison, x="model", y="MAE", color="model", text_auto=".2f",
                     title="MAE by model (lower is better)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Best model: **{eta_comparison.iloc[0]['model']}** — "
                   f"MAE {eta_comparison.iloc[0]['MAE']:.2f} min, R² {eta_comparison.iloc[0]['R2']:.3f}")

# ---------------------------------------------------------------- Forecasting
with tab_forecast:
    st.subheader("Daily demand forecast")
    daily = train.groupby(train["Order_Date"].dt.date).size().reset_index()
    daily.columns = ["date", "orders"]
    fig = px.line(daily, x="date", y="orders", markers=True, title="Daily order volume (Feb 11 – Apr 6, 2022)")
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "A gradient-boosted forecast (day-of-week + lag-1/2/3/7 + rolling means + festival/weather rate) "
        "achieves **MAE ≈ 26 orders/day (MAPE 2.5%)** on a 9-day held-out window, vs. **MAE ≈ 183** for a "
        "same-day-last-week naive baseline — see `notebooks/04_demand_forecasting.ipynb` for the full walk-forward evaluation."
    )

# ---------------------------------------------------------------- Recommendation
with tab_rec:
    st.subheader("Recommend the best delivery partner for an order")
    col1, col2 = st.columns([1, 1.3])
    with col1:
        n_candidates = st.slider("Candidate pool size", 5, 50, 20)
        speed_weight = st.slider("Weight on speed (vs. reliability)", 0.0, 1.0, 0.6)
        rec_order = dict(row)  # reuse the order built in the ETA tab
        pool = profiles.sample(n_candidates, random_state=int(n_candidates * 7))

        recs = []
        for _, p in pool.iterrows():
            r = dict(rec_order)
            r["Delivery_person_Age"] = p["avg_age"]
            r["Delivery_person_Ratings"] = p["avg_rating"]
            r["Vehicle_condition"] = p["avg_vehicle_condition"]
            r["Type_of_vehicle"] = p["primary_vehicle"]
            recs.append(r)
        cand_df = pd.DataFrame(recs)
        Xc = pd.get_dummies(cand_df, columns=["Type_of_order", "Type_of_vehicle", "City"], drop_first=True)
        Xc = Xc.reindex(columns=feature_columns, fill_value=0)
        predicted_eta = eta_model.predict(Xc)

        result = pool.copy().reset_index(drop=True)
        result["predicted_eta_min"] = predicted_eta
        eta_norm = 1 - (result["predicted_eta_min"] - result["predicted_eta_min"].min()) / (
            result["predicted_eta_min"].max() - result["predicted_eta_min"].min() + 1e-9
        )
        result["dispatch_score"] = speed_weight * eta_norm + (1 - speed_weight) * result["reliability_score"]
        result = result.sort_values("dispatch_score", ascending=False).reset_index(drop=True)

        st.markdown("**Top 5 recommended partners for the order configured in the ETA tab:**")
        st.dataframe(
            result[["Delivery_person_ID", "avg_rating", "n_deliveries", "predicted_eta_min", "dispatch_score"]].head(5),
            use_container_width=True,
        )
    with col2:
        fig = px.scatter(result, x="predicted_eta_min", y="reliability_score", size="dispatch_score",
                          color="dispatch_score", hover_data=["Delivery_person_ID"],
                          title="Candidate partners: predicted ETA vs. reliability")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- Fraud
with tab_fraud:
    st.subheader("Anomaly / fraud detection")
    n_flagged = fraud_scored["is_anomaly"].sum()
    st.metric("Orders flagged", f"{n_flagged} / {len(fraud_scored)} ({fraud_scored['is_anomaly'].mean()*100:.1f}%)")
    fig = px.histogram(fraud_scored, x="anomaly_score", nbins=60, title="Anomaly score distribution")
    threshold = fraud_scored[fraud_scored["is_anomaly"]]["anomaly_score"].min()
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="flag threshold")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Top flagged orders (with rule-based explanation):**")
    top = fraud_scored[fraud_scored["is_anomaly"]].sort_values("anomaly_score", ascending=False)
    st.dataframe(
        top[["ID", "implied_speed_kmh", "distance_km", "Time_taken(min)", "Delivery_person_Ratings", "flag_reason"]].head(15),
        use_container_width=True,
    )

# ---------------------------------------------------------------- Optimization
with tab_opt:
    st.subheader("Batch order-to-partner assignment optimization")
    batch_size = st.slider("Batch size (simultaneous orders & partners)", 5, 30, 15)
    order_cols = ["distance_km", "order_hour", "order_dow", "is_weekend", "is_rush_hour",
                  "prep_time_min", "traffic_ordinal", "distance_traffic_interaction",
                  "bad_weather", "is_festival", "multiple_deliveries", "Type_of_order", "City"]
    seed = st.number_input("Random seed (batch sample)", 0, 100, 7)
    orders_batch = train[order_cols].sample(batch_size, random_state=seed).reset_index(drop=True)
    partners_batch = profiles.sample(batch_size, random_state=seed).reset_index(drop=True)
    cost = build_cost_matrix(orders_batch, partners_batch, eta_model, feature_columns)
    opt_pairs, opt_total = optimize_assignment(cost)
    naive_pairs, naive_total = naive_assignment(cost, seed=seed)
    improvement = (naive_total - opt_total) / naive_total * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Optimized total time", f"{opt_total:.1f} min")
    c2.metric("Naive FCFS total time", f"{naive_total:.1f} min")
    c3.metric("Improvement", f"{improvement:.1f}%")

    fig = px.imshow(cost, labels=dict(x="Order #", y="Partner #", color="Predicted min"),
                     title="Partner × Order predicted-ETA cost matrix", color_continuous_scale="RdYlGn_r")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Solved optimally with the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`). "
               "Across 20 random batches this averages ~5-6% reduction in total fleet delivery time vs. naive FCFS dispatch.")

# ---------------------------------------------------------------- Monitoring
with tab_monitor:
    st.subheader("Data drift monitoring")
    report, any_drift = check_drift(train, test)
    st.dataframe(report, use_container_width=True)
    if any_drift:
        st.error("Drift detected — retraining pipeline would trigger.")
    else:
        st.success("No significant drift detected between training data and the holdout set — no retraining trigger.")
    st.caption(
        "Kolmogorov-Smirnov two-sample test per feature (α=0.01). In production this runs on a schedule "
        "against fresh incoming data (not a static holdout) and gates `src/pipeline.py` — see "
        "`notebooks/08_operational_automation.ipynb`."
    )
