import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from utils.logger import get_logger

logger = get_logger("features")

DATA_PATH = "data/heart.csv"
SCALER_PATH = "src/models/scaler.pkl"


def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    return df


def validar_datos(df: pd.DataFrame):
    assert df.isnull().sum().sum() == 0, "Hay valores nulos en el dataset"
    assert (df["target"].isin([0, 1])).all(), "Target contiene valores inválidos"
    assert len(df) > 0, "El dataset está vacío"
    logger.info("Validaciones de datos pasadas ✓")


def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Feature: ratio frecuencia cardiaca / edad
    df["thalach_age_ratio"] = df["thalach"] / df["age"]

    # Feature: presión arterial alta
    df["high_bp"] = (df["trestbps"] > 140).astype(int)

    # Feature: colesterol alto
    df["high_chol"] = (df["chol"] > 240).astype(int)

    logger.info(f"Features creados: {df.shape[1]} columnas totales")
    return df


def preparar_split(df: pd.DataFrame):
    features = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope",
        "ca", "thal", "thalach_age_ratio", "high_bp", "high_chol"
    ]

    X = df[features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Escalar features numéricos
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=features
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=features
    )

    # Guardar scaler para usarlo en la API
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler guardado en {SCALER_PATH}")

    logger.info(f"Train: {len(X_train_scaled)} filas | Test: {len(X_test_scaled)} filas")
    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    df = cargar_datos()
    validar_datos(df)
    df = crear_features(df)
    X_train, X_test, y_train, y_test = preparar_split(df)
    logger.info("Feature engineering completado")