# run2_train.py
# Run 2: Dataset v1, LinearRegression, Hyperparameter change (fit_intercept=False)

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

# Load v1 data
data = pd.read_csv("data/housing.csv")
data = data.fillna(data.mean(numeric_only=True))

y = data["median_house_value"]
X = data.drop("median_house_value", axis=1)
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameter change: fit_intercept=False, normalize via manual scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

FIT_INTERCEPT = False

with mlflow.start_run(run_name="Run2_v1_LinearReg_HyperparamChange"):
    mlflow.log_param("run_id_label", "Run 2")
    mlflow.log_param("dataset_version", "v1")
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_param("features", "all")
    mlflow.log_param("fit_intercept", FIT_INTERCEPT)
    mlflow.log_param("scaling", "StandardScaler")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("student_name", "Jaasir")
    mlflow.log_param("roll_no", "2022bcs0010")

    model = LinearRegression(fit_intercept=FIT_INTERCEPT)
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, "model")

    metrics = {
        "run": "Run 2",
        "dataset_version": "v1",
        "model": "LinearRegression",
        "fit_intercept": FIT_INTERCEPT,
        "scaling": "StandardScaler",
        "features": "all",
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