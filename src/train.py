# run4_train.py
# Run 4: Dataset v2, LinearRegression, Feature selection (reduced features)
# Selected features: longitude, latitude, housing_median_age, total_rooms,
#                    median_income, ocean_proximity

import pandas as pd
import json
import os
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("2022bcs0010_experiment")

# Load v2 data
data = pd.read_csv("data/housing.csv")
data = data.fillna(data.mean(numeric_only=True))

# --- Feature Selection ---
SELECTED_FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "median_income",
    "ocean_proximity"   # categorical — will be one-hot encoded
]

y = data["median_house_value"]
X = data[SELECTED_FEATURES]
X = pd.get_dummies(X)  # encodes ocean_proximity

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="Run4_v2_LinearReg_FeatureSelection"):
    mlflow.log_param("run_id_label", "Run 4")
    mlflow.log_param("dataset_version", "v2")
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_param("features", "selected_subset")
    mlflow.log_param("selected_features", str(SELECTED_FEATURES))
    mlflow.log_param("num_features", len(X.columns))
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("student_name", "Jaasir")
    mlflow.log_param("roll_no", "2022bcs0010")

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, "model")

    metrics = {
        "run": "Run 4",
        "dataset_version": "v2",
        "model": "LinearRegression",
        "features": "selected_subset",
        "selected_features": SELECTED_FEATURES,
        "num_features_used": len(X.columns),
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