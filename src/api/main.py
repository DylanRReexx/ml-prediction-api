import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import lightgbm as lgb
import pandas as pd
import pickle
import mlflow
from utils.logger import get_logger

logger = get_logger("api")

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts heart disease probability using LightGBM",
    version="1.0.0"
)

MODEL_PATH = "src/models/model.pkl"
SCALER_PATH = "src/models/scaler.pkl"

# Cargar modelo y scaler al iniciar
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    logger.info("Modelo y scaler cargados correctamente")
except Exception as e:
    logger.error(f"Error cargando modelo: {e}")
    model = None
    scaler = None


class PatientData(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0=Female, 1=Male)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=80, le=220, description="Resting blood pressure")
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol in mg/dl")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results")
    thalach: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored")
    thal: int = Field(..., ge=0, le=3, description="Thal (0=Normal, 1=Fixed defect, 2=Reversible defect)")


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    diagnosis: str
    risk_level: str


@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API", "status": "running"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Crear features
        data = pd.DataFrame([patient.model_dump()])
        data["thalach_age_ratio"] = data["thalach"] / data["age"]
        data["high_bp"] = (data["trestbps"] > 140).astype(int)
        data["high_chol"] = (data["chol"] > 240).astype(int)

        # Escalar
        data_scaled = scaler.transform(data)

        # Predecir
        prediction = int(model.predict(data_scaled)[0])
        probability = round(float(model.predict_proba(data_scaled)[0][1]), 4)

        # Nivel de riesgo
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"

        diagnosis = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

        logger.info(f"Prediction: {diagnosis} | Probability: {probability} | Risk: {risk_level}")

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            diagnosis=diagnosis,
            risk_level=risk_level
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))