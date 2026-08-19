import pandas as pd
import joblib
import fastf1

# CIRCUIT MAPPING

RACE_TO_CIRCUIT = {
    "Australia": "albert_park", "China": "shanghai", "Japan": "suzuka",
    "Bahrain": "bahrain", "Saudi Arabia": "jeddah", "Miami": "miami",
    "Monaco": "monaco", "Spain": "catalunya", "Canada": "villeneuve",
    "Austria": "red_bull_ring", "Britain": "silverstone", "Belgium": "spa",
    "Hungary": "hungaroring", "Netherlands": "zandvoort", "Italy": "monza",
    "Azerbaijan": "baku", "Singapore": "singapore", "USA": "cota",
    "Mexico": "rodriguez", "Brazil": "interlagos", "Las Vegas": "las_vegas",
    "Qatar": "losail", "Abu Dhabi": "yas_marina",
}

# LOAD ARTIFACTS

class ModelBundle:
    """Hold everything loaded once, reused across requests."""
    def __init__(self):
        self.model = joblib.load("models/f1_xgboost_model.pkl")
        self.features = joblib.load("models/model_features.pkl")
        self.le_circuit = joblib.load("models/le_circuit.pkl")
        self.le_constructor = joblib.load("models/le_constructor.pkl")
        self.le_driver = joblib.load("models/le_driver.pkl")
        self.history_df = pd.read_csv("data/processed/f1_eda_ready.csv")

# QUALIFYING FETCHER

def load_qualifying(season: int, race_name: str) -> pd.DataFrame:
    session = fastf1.get_session(season, race_name, "Q")
    session.load(telemetry=False, weather=False, messages=False)
    results = session.results.copy()
    quali_df = pd.DataFrame({
        'driver_name': results['FullName'],
        'constructor_name': results['TeamName'],
        'grid_position': results['Position'],  # Grid position is usually position + 1
        'quali_position': results['Position'],
    })
    return quali_df

# FEATURE BUILDER

def build_race_features(quali_df, history_df, circuit_id, le_driver, le_constructor, le_circuit):
    rows = []
    for _, row in quali_df.iterrows():
        driver = row["driver_name"]
        constructor = row["constructor_name"]
        driver_hist = history_df[history_df["driver_name"] == driver]
        constructor_hist = history_df[history_df["constructor_name"] == constructor]

        avg_finish_last3 = driver_hist.sort_values(["year", "round"]).tail(3)["finish_position"].mean()
        avg_points_last3 = driver_hist.sort_values(["year", "round"]).tail(3)["points"].mean()
        dnf_rate = (driver_hist["finish_position"] > 15).mean()

        circuit_hist = driver_hist[driver_hist["circuit_id"] == circuit_id]
        win_rate = (circuit_hist["finish_position"] == 1).mean() if len(circuit_hist) > 0 else 0

        driver_points = driver_hist["points"].sum()
        constructor_points = constructor_hist["points"].sum()
        constructor_avg_points_last3 = constructor_hist.groupby("race_id")["points"].sum().tail(3).mean()

        try:
            driver_encoded = le_driver.transform([driver])[0]
        except Exception:
            driver_encoded = -1
        try:
            constructor_encoded = le_constructor.transform([constructor])[0]
        except Exception:
            constructor_encoded = -1
        try:
            circuit_encoded = le_circuit.transform([circuit_id])[0]
        except Exception:
            circuit_encoded = -1

        rows.append({
            "driver_name": driver,
            "constructor_name": constructor,
            "grid_position": row["grid_position"],
            "quali_position": row["quali_position"],
            "driver_avg_finish_last3": avg_finish_last3,
            "driver_avg_points_last3": avg_points_last3,
            "constructor_avg_points_last3": constructor_avg_points_last3,
            "driver_win_rate_at_circuit": win_rate,
            "driver_dnf_rate": dnf_rate,
            "driver_points_so_far": driver_points,
            "constructor_points_so_far": constructor_points,
            "driver_encoded": driver_encoded,
            "constructor_encoded": constructor_encoded,
            "circuit_encoded": circuit_encoded,
        })

    pred_df = pd.DataFrame(rows)
    pred_df["driver_avg_finish_last3"] = pred_df["driver_avg_finish_last3"].fillna(20)
    remaining_cols = [
        "driver_avg_points_last3", "constructor_avg_points_last3",
        "driver_win_rate_at_circuit", "driver_dnf_rate",
        "driver_points_so_far", "constructor_points_so_far",
    ]
    pred_df[remaining_cols] = pred_df[remaining_cols].fillna(0)
    return pred_df

# PREDICTOR CLASS

def predict_race(season: int, race_name: str, bundle: ModelBundle) -> pd.DataFrame:
    if race_name not in RACE_TO_CIRCUIT:
        raise ValueError(f"Unknown race: {race_name}")
    circuit_id = RACE_TO_CIRCUIT[race_name]
    quali_df = load_qualifying(season, race_name)
    feature_df = build_race_features(quali_df, bundle.history_df, circuit_id, bundle.le_driver, bundle.le_constructor, bundle.le_circuit)
    X_pred = feature_df[bundle.features]
    feature_df['win_probability'] = bundle.model.predict_proba(X_pred)[:,1]

    return feature_df[["driver_name", "constructor_name", "grid_position", "win_probability"]] \
           .sort_values(by="win_probability", ascending=False)