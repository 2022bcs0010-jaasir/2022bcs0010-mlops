# run5_train.py
# Run 5: Dataset v2, RandomForestRegressor (Model B), Feature selection

import pandas as pd
import json
import os
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("2022bcs0010_experiment")

# Load v2 data
data = pd.read_csv("data/housing.csv")
data = data.fillna(data.mean(numeric_only=True))

# --- Feature Selection (same subset as Run 4 for fair comparison) ---
SELECTED_FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "median_income",
    "ocean_proximity"
]

y = data["median_house_value"]
X = data[SELECTED_FEATURES]
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# RandomForest hyperparameters
N_ESTIMATORS = 100
MAX_DEPTH = 15
MIN_SAMPLES_SPLIT = 5
RANDOM_STATE = 42

with mlflow.start_run(run_name="Run5_v2_RandomForest_FeatureSelection"):
    mlflow.log_param("run_id_label", "Run 5")
    mlflow.log_param("dataset_version", "v2")
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("features", "selected_subset")
    mlflow.log_param("selected_features", str(SELECTED_FEATURES))
    mlflow.log_param("num_features", len(X.columns))
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_param("min_samples_split", MIN_SAMPLES_SPLIT)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("student_name", "Jaasir")
    mlflow.log_param("roll_no", "2022bcs0010")

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, "model")

    metrics = {
        "run": "Run 5",
        "dataset_version": "v2",
        "model": "RandomForestRegressor",
        "features": "selected_subset",
        "selected_features": SELECTED_FEATURES,
        "num_features_used": len(X.columns),
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "dataset_size": len(data),
        "rmse": rmse,
        "r2": r2,
        "name": "Jaasir",
        "roll_no": "2022bcs0010"
    }

    print(metrics)
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)