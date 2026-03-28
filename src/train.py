import pandas as pd
import json
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("data/housing.csv")

data = data.fillna(data.mean(numeric_only=True))

y = data["median_house_value"]

X = data.drop("median_house_value", axis=1)

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

metrics = {
    "dataset_size": len(data),
    "rmse": float(rmse),
    "r2": float(r2)
}

print(metrics)

os.makedirs("metrics", exist_ok=True)

with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)