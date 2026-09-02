import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer


RAW_NUM_COLS = [
    "age",
    "balance",
    "campaign",
    "previous",
    "pdays"
]

CAT_COLS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "poutcome"
]

ENGINEERED_NUM_COLS = [
    "multi_loans",
    "balance_per_age",
    "is_young_single",
    "has_partner",
    "married_with_housing"
]
ENGINEERED_CAT_COLS = ["balance_category"]


def get_raw_feature_names():
    """Возвращает списки исходных числовых и категориальных колонок до генерации признаков"""
    return RAW_NUM_COLS, CAT_COLS


def get_all_feature_names():
    """Возвращает списки всех числовых и категориальных колонок после генерации признаков"""
    return RAW_NUM_COLS + ENGINEERED_NUM_COLS, CAT_COLS + ENGINEERED_CAT_COLS


def add_features(df):
    """Генерирует новые признаковые колонки на основе исходных данных"""
    df = df.copy()

    df["balance_category"] = pd.cut(
        df["balance"],
        bins=[-np.inf, 0, 500, 1500, np.inf],
        labels=["low_neg", "low", "medium", "high"]
    ).astype(str)

    df["multi_loans"] = ((df["loan"] == "yes") & (df["housing"] == "yes")).astype(int)
    df["balance_per_age"] = df["balance"] / df["age"]
    df["is_young_single"] = ((df["marital"] == "single") & (df["age"] < 30)).astype(int)
    df["has_partner"] = (df["marital"] == "married").astype(int)
    df["married_with_housing"] = ((df["marital"] == "married") & (df["housing"] == "yes")).astype(int)

    return df


def get_feature_engineering_transformer():
    """Возвращает трансформер для применения генерации признаков в пайплайне"""
    return FunctionTransformer(add_features)
