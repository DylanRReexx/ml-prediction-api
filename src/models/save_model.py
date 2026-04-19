import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import lightgbm as lgb
from utils.logger import get_logger
from src.features.build_features import cargar_datos, validar_datos, crear_features, preparar_split

logger = get_logger("save_model")

MODEL_PATH = "src/models/model.pkl"


if __name__ == "__main__":
    df = cargar_datos()
    validar_datos(df)
    df = crear_features(df)
    X_train, X_test, y_train, y_test = preparar_split(df)

    # Entrenamos con los mejores parámetros (Experimento 1)
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Modelo guardado en {MODEL_PATH}")