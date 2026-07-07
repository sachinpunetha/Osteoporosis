from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# -----------------------------
# Load Saved Model
# -----------------------------
saved = joblib.load("best_model_pipeline.joblib")

model = saved["model"]
scaler = saved["scaler"]
features = saved["features"]
model_name = saved["model_name"]

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="Osteoporosis Prediction API",
    version="1.0"
)

# -----------------------------
# Input Schema
# -----------------------------
class Patient(BaseModel):
    fnt: float
    calcitriol: float
    uric: float
    alt: float
    bun: float
    crea: float
    fbg: float
    ldl_c: float
    l1_4t: float
    age: float
    hdl_c: float
    bmi: float
    ca: float
    p: float
    height: float
    ast: float
    weight: float
    calsium: float
    calcitonin: float
    as_: float


@app.get("/")
def home():
    return {
        "message": "Osteoporosis Prediction API is Running",
        "model": model_name
    }


@app.post("/predict")
def predict(patient: Patient):

    data = {
        "fnt": patient.fnt,
        "calcitriol": patient.calcitriol,
        "uric": patient.uric,
        "alt": patient.alt,
        "bun": patient.bun,
        "crea": patient.crea,
        "fbg": patient.fbg,
        "ldl_c": patient.ldl_c,
        "l1_4t": patient.l1_4t,
        "age": patient.age,
        "hdl_c": patient.hdl_c,
        "bmi": patient.bmi,
        "ca": patient.ca,
        "p": patient.p,
        "height": patient.height,
        "ast": patient.ast,
        "weight": patient.weight,
        "calsium": patient.calsium,
        "calcitonin": patient.calcitonin,
        "as": patient.as_
    }

    df = pd.DataFrame([data])

    df = df[features]

    df_scaled = scaler.transform(df)

    prediction = int(model.predict(df_scaled)[0])

    probability = model.predict_proba(df_scaled)[0]

    return {
        "prediction": prediction,
        "probability_no": round(float(probability[0]), 4),
        "probability_yes": round(float(probability[1]), 4),
        "risk": "Osteoporosis" if prediction == 1 else "Healthy"
    }