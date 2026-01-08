from fastapi import FastAPI
import joblib
import pandas as pd
import os

app = FastAPI(title="Customer Churn Prediction API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "logistic_regression.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "..", "model", "feature_names.pkl")

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict_churn(data: dict):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data])

    # Reorder columns to match training
    input_df = input_df[feature_names]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "confidence": round(float(probability), 3)
    }
