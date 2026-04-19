import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from utils.logger import get_logger
from src.features.build_features import cargar_datos, validar_datos, crear_features, preparar_split

logger = get_logger("training")

MLFLOW_EXPERIMENT = "heart-disease-prediction"
MODEL_PATH = "src/models/model.pkl"


def entrenar(params: dict, X_train, y_train, X_test, y_test):
    """Entrena un modelo LightGBM y loguea en MLflow."""

    with mlflow.start_run():
        # Loguear parámetros
        mlflow.log_params(params)

        # Entrenar
        model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
        model.fit(X_train, y_train)

        # Evaluar
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # Loguear métricas
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        # Loguear modelo
        mlflow.lightgbm.log_model(model, "model")

        logger.info(f"Accuracy: {round(acc * 100, 2)}%")
        logger.info(f"F1 Score: {round(f1, 4)}")
        logger.info(f"ROC AUC:  {round(auc, 4)}")
        logger.info(f"Run ID:   {mlflow.active_run().info.run_id}")

        print("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred,
              target_names=["No Disease", "Disease"]))

        return model, mlflow.active_run().info.run_id


if __name__ == "__main__":
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Preparar datos
    df = cargar_datos()
    validar_datos(df)
    df = crear_features(df)
    X_train, X_test, y_train, y_test = preparar_split(df)

    # Experimento 1 — parámetros base
    logger.info("Experimento 1 — parámetros base")
    params1 = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.05,
        "num_leaves": 31
    }
    model1, run1 = entrenar(params1, X_train, y_train, X_test, y_test)

    # Experimento 2 — más árboles, menos learning rate
    logger.info("Experimento 2 — más árboles, menos learning rate")
    params2 = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.01,
        "num_leaves": 50
    }
    model2, run2 = entrenar(params2, X_train, y_train, X_test, y_test)

    # Experimento 3 — más regularización
    logger.info("Experimento 3 — más regularización")
    params3 = {
        "n_estimators": 150,
        "max_depth": 3,
        "learning_rate": 0.05,
        "num_leaves": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1
    }
    model3, run3 = entrenar(params3, X_train, y_train, X_test, y_test)

    logger.info("Entrenamiento completado — revisá MLflow UI para comparar experimentos")
    logger.info("Corré: mlflow ui")