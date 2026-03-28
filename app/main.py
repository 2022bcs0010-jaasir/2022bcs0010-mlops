from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def health():
    return {
        "name": "Mohamed Jaasir Subair",
        "roll": "2022BCS0010"
    }

@app.post("/predict")
def predict():
    return {
        "prediction": 123,
        "name": "Mohamed Jaasir Subair",
        "roll": "2022BCS0010"
    }