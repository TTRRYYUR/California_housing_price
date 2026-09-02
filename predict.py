import joblib
import pandas as pd
from functools import lru_cache

from features import get_raw_feature_names


@lru_cache(maxsize=1)
def load_model(model_path = "best_bank_model.pkl"):
    """"Загружает модель ровно один раз"""
    model = joblib.load(model_path)
    return model


def predict_client(features , model_path = "best_bank_model.pkl"):
    """Загружает модель и делает предсказание для одного клиента"""
    pipeline = load_model(model_path)
    num_cols, cat_cols = get_raw_feature_names()
    order_cols = num_cols + cat_cols
    df = pd.DataFrame([features])[order_cols]

    proba = pipeline.predict_proba(df)[0, 1]
    prediction = pipeline.predict(df)[0]

    return {
        "prediction": int(prediction),
        "probability":float(proba)
    }
