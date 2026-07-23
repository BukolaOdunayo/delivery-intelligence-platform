"""Generates every static PNG used in the README and notebooks, from the
already-trained artifacts, so the writeup and notebooks stay in sync
with whatever the pipeline actually produced."""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110

OUT = "images"


def savefig(name):
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    plt.close()


def main():
    train = pd.read_parquet("data/processed/train_clean.parquet")

    # 1. target distribution
    plt.figure(figsize=(7, 4))
    sns.histplot(train["Time_taken(min)"], bins=40, kde=True, color="#4C72B0")
    plt.title("Delivery Time Distribution")
    plt.xlabel("Time taken (min)")
    savefig("target_distribution")

    # 2. distance vs time
    plt.figure(figsize=(7, 4))
    sns.scatterplot(data=train.sample(3000, random_state=1), x="distance_km", y="Time_taken(min)",
                     hue="Road_traffic_density", alpha=0.4, s=15)
    plt.title("Delivery Time vs. Distance (colored by traffic)")
    savefig("distance_vs_time")

    # 3. traffic vs time
    plt.figure(figsize=(6, 4))
    order = ["Low", "Medium", "High", "Jam"]
    sns.boxplot(data=train, x="Road_traffic_density", y="Time_taken(min)", order=order)
    plt.title("Delivery Time by Traffic Density")
    savefig("traffic_vs_time")

    # 4. ETA model comparison
    eta_results = pd.read_csv("models/eta_model_comparison.csv")
    plt.figure(figsize=(6, 4))
    sns.barplot(data=eta_results, x="model", y="MAE", color="#55A868")
    plt.title("ETA Model Comparison (lower MAE = better)")
    plt.xticks(rotation=20)
    savefig("eta_model_comparison")

    # 5. feature importance
    fi = pd.read_csv("models/eta_feature_importance.csv", index_col=0).iloc[:, 0].sort_values(ascending=True).tail(12)
    plt.figure(figsize=(7, 5))
    fi.plot(kind="barh", color="#C44E52")
    plt.title("Top Feature Importances — ETA Model")
    savefig("eta_feature_importance")

    # 6. demand forecast
    daily = train.groupby(train["Order_Date"].dt.date).size()
    plt.figure(figsize=(8, 4))
    daily.plot(marker="o", markersize=3)
    plt.title("Daily Order Volume (raw series)")
    plt.ylabel("orders")
    savefig("daily_demand_series")

    # 7. fraud anomaly scores
    scored = pd.read_parquet("data/processed/train_with_fraud_scores.parquet")
    plt.figure(figsize=(7, 4))
    sns.histplot(scored["anomaly_score"], bins=50, color="#8172B2")
    plt.axvline(scored[scored["is_anomaly"]]["anomaly_score"].min(), color="red", linestyle="--", label="flag threshold")
    plt.title("Fraud/Anomaly Score Distribution")
    plt.legend()
    savefig("fraud_anomaly_scores")

    # 8. logistics optimization improvement
    with open("models/logistics_optimization_results.json") as f:
        opt_res = json.load(f)
    plt.figure(figsize=(6, 4))
    plt.hist(opt_res["trials"], bins=10, color="#DD8452")
    plt.axvline(opt_res["mean_improvement_pct"], color="black", linestyle="--",
                label=f"mean {opt_res['mean_improvement_pct']:.1f}%")
    plt.title("Optimization Lift vs. Naive Assignment (20 batches)")
    plt.xlabel("% reduction in total predicted delivery time")
    plt.legend()
    savefig("logistics_optimization_lift")

    print("All plots saved to images/")


if __name__ == "__main__":
    main()
