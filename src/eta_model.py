"""Core ETA prediction model: trains & compares baselines vs ensembles,
picks the best, saves the artifact + the encoder column reference."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data_prep import build_model_matrix


def train_eta_models(train_df: pd.DataFrame, models_dir="models"):
    X = build_model_matrix(train_df)
    y = train_df["Time_taken(min)"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42),
        "xgboost": xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=-1,
        ),
    }

    results = []
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        results.append({
            "model": name,
            "MAE": mean_absolute_error(y_val, preds),
            "RMSE": np.sqrt(mean_squared_error(y_val, preds)),
            "R2": r2_score(y_val, preds),
        })
        fitted[name] = model

    results_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
    best_name = results_df.iloc[0]["model"]
    best_model = fitted[best_name]

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best_model, f"{models_dir}/eta_model.pkl")
    with open(f"{models_dir}/eta_feature_columns.json", "w") as f:
        json.dump(list(X.columns), f)
    results_df.to_csv(f"{models_dir}/eta_model_comparison.csv", index=False)

    # feature importance if available
    if hasattr(best_model, "feature_importances_"):
        importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        importance.to_csv(f"{models_dir}/eta_feature_importance.csv")

    return results_df, best_name, best_model, X.columns.tolist()


if __name__ == "__main__":
    train_df = pd.read_parquet("data/processed/train_clean.parquet")
    results_df, best_name, best_model, cols = train_eta_models(train_df)
    print(results_df)
    print("Best model:", best_name)
