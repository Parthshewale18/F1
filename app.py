import streamlit as st
import pandas as pd
import numpy as np
import joblib
import fastf1
import plotly.express as px

st.set_page_config(
    page_title="F1 Race Winnner Predictor",
    page_icon="🏎️",
    layout="wide",
)

# LOAD ARTIFACTS

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/f1_xgboost_model.pkl")
    features = joblib.load("models/model_features.pkl")
    le_driver = joblib.load("models/le_driver.pkl")
    le_constructor = joblib.load("models/le_constructor.pkl")
    le_circuit = joblib.load("models/le_circuit.pkl")

    return (model, features, le_driver, le_constructor, le_circuit)

@st.cache_data
def load_history():
    return pd.read_csv("data/processed/f1_eda_ready.csv")

model, features, le_driver, le_constructor, le_circuit = load_artifacts()
history_df = load_history()

# CIRCUIT MAPPING

RACE_TO_CIRCUIT = {
    "Australia": "albert_park",
    "China": "shanghai",
    "Japan": "suzuka",
    "Bahrain": "bahrain",
    "Saudi Arabia": "jeddah",
    "Miami": "miami",
    "Monaco": "monaco",
    "Spain": "catalunya",
    "Canada": "villeneuve",
    "Austria": "red_bull_ring",
    "Britain": "silverstone",
    "Belgium": "spa",
    "Hungary": "hungaroring",
    "Netherlands": "zandvoort",
    "Italy": "monza",
    "Azerbaijan": "baku",
    "Singapore": "singapore",
    "USA": "cota",
    "Mexico": "rodriguez",
    "Brazil": "interlagos",
    "Las Vegas": "las_vegas",
    "Qatar": "losail",
    "Abu Dhabi": "yas_marina"
}

# QUALIFYING FETCHER

@st.cache_data(show_spinner=False)
def load_qualifing(season, race_name):
    session = fastf1.get_session(season, race_name, "Q")
    session.load(telemetry=False, weather=False, messages=False)  # Load session data (this may take a moment)
    results = session.results.copy()
    quali_df = pd.DataFrame({
        'driver_name': results['FullName'],
        'constructor_name': results['TeamName'],
        'grid_position': results['Position'],  # Grid position is usually position + 1
        'quali_position': results['Position'],
    })
    return quali_df

# FEATURE BUILDER

def build_race_features(quali_df, history_df, circuit_id):
    rows = []
    for _, row in quali_df.iterrows():
        driver = row['driver_name']
        constructor = row['constructor_name']
        driver_hist = history_df[ history_df['driver_name'] == driver]
        constructor_hist = history_df[ history_df['constructor_name'] == constructor]
        avg_finish_last3 = ( driver_hist.sort_values(['year', 'round'] ).tail(3) ['finish_position'].mean() )
        avg_points_last3 = ( driver_hist.sort_values(['year', 'round'] ).tail(3) ['points'].mean() )
        dnf_rate = ( (driver_hist["finish_position"] > 15).mean() )
        circuit_hist = driver_hist[ driver_hist['circuit_id'] == circuit_id ]
        if len(circuit_hist) > 0:
            win_rate = (circuit_hist['finish_position'] == 1).mean()
        else:
            win_rate = 0
        driver_points = driver_hist['points'].sum()
        constructor_points = constructor_hist['points'].sum()
        construtor_avg_points_last3 = (constructor_hist.groupby('race_id') ['points'].sum().tail(3).mean())
        try:
            driver_encoded = ( le_driver.transform([driver])[0] )
        except:
            driver_encoded = -1

        try:
            constructor_encoded = ( le_constructor.transform([constructor])[0] )
        except:
            constructor_encoded = -1

        try:
            circuit_encoded = ( le_circuit.transform([circuit_id])[0] )
        except:
            circuit_encoded = -1
        rows.append({
            'driver_name': driver,
            'constructor_name': constructor,
            'grid_position': row['grid_position'],
            'quali_position': row['quali_position'],
            'avg_finish_last3': avg_finish_last3,
            'avg_points_last3': avg_points_last3,
            'dnf_rate': dnf_rate,
            'win_rate': win_rate,
            'driver_points': driver_points,
            'constructor_points': constructor_points,
            'construtor_avg_points_last3': construtor_avg_points_last3,
            'driver_encoded': driver_encoded,
            'constructor_encoded': constructor_encoded,
            'circuit_encoded': circuit_encoded
        })
    pred_df = pd.DataFrame(rows)
    pred_df["driver_avg_finish_last3"] = ( pred_df["driver_avg_finish_last3"].fillna(20) )
    remaining_cols = [
        "driver_avg_points_last3",
        "constructor_avg_points_last3",
        "driver_win_rate_at_circuit",
        "driver_dnf_rate",
        "driver_points_so_far",
        "constructor_points_so_far"
    ]

    pred_df[remaining_cols] = ( pred_df[remaining_cols].fillna(0) )

    return pred_df


# UI
st.title("🏎️ F1 Race Winner Predictor")

st.markdown(
    """
    Predict the race winner using:

    - Qualifying position
    - Driver form
    - Constructor form
    - Circuit history
    - Reliability
    """
)

col1, col2 = st.columns(2)
with col1:
    season = st.number_input(
        "Select season",
         min_value = 2018,
         max_value = 2030,
         value = 2026
         )
with col2:
    race_name = st.selectbox(
        'Race',
        list(RACE_TO_CIRCUIT.keys())
        )
    
# LOAD QUALIFYING

if st.button("Load Qualifing Results"):
    try:
        with st.spinner("Loading qualifying data..."):
            quali_df = load_qualifing(season, race_name)
        
        st.success("Qualifying data loaded successfully!")
        st.dataframe(quali_df, use_container_width=True)
        st.session_state['quali_df'] = quali_df
    except Exception as e:
        st.error(f"Error loading qualifying data: {e}")