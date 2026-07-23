"""
Data cleaning utilities for the Berlin Food Delivery ETA & Ops Intelligence project.

Raw Kaggle source: https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset
"""
import numpy as np
import pandas as pd


STRING_COLS_TO_STRIP = [
    "ID", "Delivery_person_ID", "Road_traffic_density", "Type_of_order",
    "Type_of_vehicle", "Festival", "City", "Weatherconditions",
]


def _strip(df: pd.DataFrame) -> pd.DataFrame:
    for c in STRING_COLS_TO_STRIP:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def clean_raw(path: str, is_train: bool = True) -> pd.DataFrame:
    """Load and clean the raw Kaggle CSV export (handles the known quirks:
    stray whitespace, literal 'NaN' strings, malformed target column,
    numeric columns typed as text)."""
    df = pd.read_csv(path)
    df = _strip(df)

    # literal "NaN " strings -> real NaN
    df = df.replace(to_replace=r"^\s*NaN\s*$", value=np.nan, regex=True)

    # Weatherconditions comes as "conditions Sunny" etc.
    df["Weatherconditions"] = (
        df["Weatherconditions"].astype(str).str.replace("conditions", "", regex=False).str.strip()
    )
    df.loc[df["Weatherconditions"].isin(["nan", "NaN"]), "Weatherconditions"] = np.nan

    # numeric columns mistyped as strings
    df["Delivery_person_Age"] = pd.to_numeric(df["Delivery_person_Age"], errors="coerce")
    df["Delivery_person_Ratings"] = pd.to_numeric(df["Delivery_person_Ratings"], errors="coerce")
    df["multiple_deliveries"] = pd.to_numeric(df["multiple_deliveries"], errors="coerce")

    if is_train:
        df["Time_taken(min)"] = (
            df["Time_taken(min)"].astype(str).str.extract(r"(\d+)").astype(float)
        )

    # dates / times
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y", errors="coerce")
    for c in ["Time_Orderd", "Time_Order_picked"]:
        df[c] = pd.to_datetime(df[c], format="%H:%M:%S", errors="coerce").dt.time

    # impute
    num_cols = ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    cat_cols = ["Weatherconditions", "Road_traffic_density", "Festival", "City"]
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0])

    return df.reset_index(drop=True)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def add_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering shared across every downstream module
    (ETA model, forecasting, matching, fraud, optimization)."""
    df = df.copy()

    # distance, capped at 30km (corrupted coordinate outliers)
    df["distance_km"] = haversine_km(
        df["Restaurant_latitude"], df["Restaurant_longitude"],
        df["Delivery_location_latitude"], df["Delivery_location_longitude"],
    ).clip(upper=30)

    # datetime features
    df["order_hour"] = df["Time_Orderd"].apply(lambda t: t.hour if pd.notnull(t) else np.nan)
    df["order_dow"] = df["Order_Date"].dt.dayofweek
    df["is_weekend"] = df["order_dow"].isin([5, 6]).astype(int)
    df["is_rush_hour"] = df["order_hour"].isin([8, 9, 12, 13, 18, 19, 20]).astype(int)

    # prep time (minutes between order placed and picked up)
    def _prep_minutes(row):
        o, p = row["Time_Orderd"], row["Time_Order_picked"]
        if pd.isnull(o) or pd.isnull(p):
            return np.nan
        o_s = o.hour * 3600 + o.minute * 60 + o.second
        p_s = p.hour * 3600 + p.minute * 60 + p.second
        diff = p_s - o_s
        if diff < 0:
            diff += 24 * 3600  # order spans midnight
        return diff / 60

    df["prep_time_min"] = df.apply(_prep_minutes, axis=1)
    df["prep_time_min"] = df["prep_time_min"].fillna(df["prep_time_min"].median())
    df["order_hour"] = df["order_hour"].fillna(df["order_hour"].median())

    # traffic ordinal + interaction
    traffic_map = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}
    df["traffic_ordinal"] = df["Road_traffic_density"].map(traffic_map).fillna(1)
    df["distance_traffic_interaction"] = df["distance_km"] * df["traffic_ordinal"]

    df["bad_weather"] = df["Weatherconditions"].isin(["Stormy", "Sandstorms", "Fog"]).astype(int)
    df["is_festival"] = (df["Festival"] == "Yes").astype(int)

    return df


FEATURE_COLS_NUMERIC = [
    "Delivery_person_Age", "Delivery_person_Ratings", "Vehicle_condition",
    "multiple_deliveries", "distance_km", "order_hour", "order_dow",
    "is_weekend", "is_rush_hour", "prep_time_min", "traffic_ordinal",
    "distance_traffic_interaction", "bad_weather", "is_festival",
]
FEATURE_COLS_CATEGORICAL = ["Type_of_order", "Type_of_vehicle", "City"]


def build_model_matrix(df: pd.DataFrame, encoder_columns=None):
    """One-hot encode categoricals, align columns to a reference set if given."""
    X = pd.get_dummies(df[FEATURE_COLS_NUMERIC + FEATURE_COLS_CATEGORICAL],
                        columns=FEATURE_COLS_CATEGORICAL, drop_first=True)
    if encoder_columns is not None:
        X = X.reindex(columns=encoder_columns, fill_value=0)
    return X
