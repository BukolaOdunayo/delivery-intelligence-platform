"""
Module 2 — Demand Forecasting.

Aggregates order volume to a daily/hourly grain and forecasts near-term
demand, the kind of signal an ops team would use for delivery-partner
staffing decisions. Uses a lag + calendar feature regression approach
(walk-forward, time-based split — never shuffled).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def build_daily_demand(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby(df["Order_Date"].dt.date)
        .agg(orders=("ID", "count"), pct_festival=("is_festival", "mean"),
             pct_bad_weather=("bad_weather", "mean"))
        .reset_index()
        .rename(columns={"Order_Date": "date"})
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["dow"] = daily["date"].dt.dayofweek
    daily["is_weekend"] = daily["dow"].isin([5, 6]).astype(int)

    for lag in [1, 2, 3, 7]:
        daily[f"lag_{lag}"] = daily["orders"].shift(lag)
    daily["rolling_mean_3"] = daily["orders"].shift(1).rolling(3).mean()
    daily["rolling_mean_7"] = daily["orders"].shift(1).rolling(7).mean()

    return daily


def train_forecast_model(daily: pd.DataFrame, test_days=9, models_dir="models"):
    feature_cols = [
        "dow", "is_weekend", "pct_festival", "pct_bad_weather",
        "lag_1", "lag_2", "lag_3", "lag_7", "rolling_mean_3", "rolling_mean_7",
    ]
    d = daily.dropna(subset=feature_cols).reset_index(drop=True)

    train = d.iloc[: -test_days]
    test = d.iloc[-test_days:]

    model = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(train[feature_cols], train["orders"])

    preds = model.predict(test[feature_cols])
    mae = mean_absolute_error(test["orders"], preds)
    mape = mean_absolute_percentage_error(test["orders"], preds)

    naive_preds = test["lag_7"].values  # "same day last week" baseline
    naive_mae = mean_absolute_error(test["orders"], naive_preds)

    import joblib, os
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, f"{models_dir}/demand_forecast_model.pkl")

    return {
        "model": model,
        "feature_cols": feature_cols,
        "test_dates": test["date"].tolist(),
        "y_true": test["orders"].tolist(),
        "y_pred": preds.tolist(),
        "mae": mae,
        "mape": mape,
        "naive_mae": naive_mae,
    }


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/train_clean.parquet")
    daily = build_daily_demand(df)
    out = train_forecast_model(daily)
    print(f"Model MAE: {out['mae']:.1f} orders/day (MAPE {out['mape']*100:.1f}%)")
    print(f"Naive (same-day-last-week) baseline MAE: {out['naive_mae']:.1f} orders/day")
