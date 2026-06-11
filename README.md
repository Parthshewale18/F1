# F1 Race Winner Predictor

A Streamlit web app that predicts Formula 1 race winner probabilities from qualifying results, driver form, constructor form, circuit history, reliability, and historical race performance.

The project uses historical F1 data for feature engineering, trains an XGBoost classification model, and serves predictions through an interactive Streamlit interface.

## Features

- Fetches qualifying results with FastF1.
- Builds race-day features for each driver.
- Uses driver, constructor, and circuit encodings created during training.
- Predicts each driver's win probability with an XGBoost model.
- Displays the predicted winner and full ranking table in Streamlit.
- Includes notebooks for data collection, exploration, feature engineering, training, and prediction.

## Project Structure

```text
F1/
|-- app.py
|-- requirements.txt
|-- README.md
|-- notebooks/
|   |-- data_collection.ipynb
|   |-- eda.ipynb
|   |-- feature_engineering.ipynb
|   |-- model_training.ipynb
|   `-- predict.ipynb
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- cache/
`-- models/
    |-- f1_xgboost_model.pkl
    |-- model_features.pkl
    |-- le_driver.pkl
    |-- le_constructor.pkl
    `-- le_circuit.pkl
```

> Note: `data/`, `models/`, and cache files are ignored by Git because they can be large or generated locally.

## Tech Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- XGBoost
- FastF1
- Plotly, Matplotlib, and Seaborn
- Joblib
- Jupyter Notebook

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Parthshewale18/F1.git
cd F1
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Required Files

The Streamlit app expects these files to exist before running:

```text
data/processed/f1_eda_ready.csv
models/f1_xgboost_model.pkl
models/model_features.pkl
models/le_driver.pkl
models/le_constructor.pkl
models/le_circuit.pkl
```

If these files are missing, generate them by running the notebooks in this order:

1. `notebooks/data_collection.ipynb`
2. `notebooks/eda.ipynb`
3. `notebooks/feature_engineering.ipynb`
4. `notebooks/model_training.ipynb`
5. `notebooks/predict.ipynb` optional, for notebook-based prediction testing

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How It Works

1. Select a season and race in the Streamlit app.
2. Click `Load Qualifing Results` to fetch qualifying data from FastF1.
3. The app builds model-ready features using:
   - qualifying position
   - grid position
   - recent driver form
   - recent constructor form
   - circuit-specific win rate
   - DNF rate
   - accumulated driver and constructor points
4. Click `Predict Race Winner`.
5. The model returns win probabilities and ranks drivers from most likely to least likely winner.

## Model Overview

The model is trained as a binary classifier where each driver entry is evaluated for race-winning probability. The training pipeline uses:

- historical race results
- qualifying and grid data
- rolling driver performance
- constructor performance
- circuit history
- encoded categorical features

The trained model and preprocessing encoders are saved with Joblib in the `models/` directory.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `data_collection.ipynb` | Loads historical F1 datasets and collects recent FastF1 data. |
| `eda.ipynb` | Explores the historical dataset and class imbalance. |
| `feature_engineering.ipynb` | Creates rolling performance, circuit, reliability, and encoded features. |
| `model_training.ipynb` | Trains the XGBoost model and saves model artifacts. |
| `predict.ipynb` | Tests prediction logic outside the Streamlit app. |

## Troubleshooting

### Missing model or data files

If Streamlit raises a `FileNotFoundError`, make sure the required files listed above exist. Run the notebooks in order to regenerate them.

### FastF1 data does not load

FastF1 needs internet access for new sessions and may take time the first time it downloads data. Cached sessions are stored under `data/cache/`.

### Race has no qualifying data

Predictions require qualifying results. If qualifying has not happened yet, FastF1 may not return usable results for that race.

## Future Improvements

- Add manual qualifying input when FastF1 data is unavailable.
- Improve handling for new drivers, teams, and circuits.
- Add model evaluation metrics to the app.
- Add confidence charts and feature importance visuals.
- Save prediction results for comparison after each race.

## License

This project is for learning and portfolio use. Add a license file if you plan to distribute or publish it publicly.
