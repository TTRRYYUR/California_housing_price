from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from features import get_all_feature_names, get_feature_engineering_transformer


def create_preprocessor() -> ColumnTransformer:
    """Создает препроцессор для масштабирования числовых и кодирования категориальных признаков"""
    num_cols, cat_cols = get_all_feature_names()

    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols)
    ])


def create_model_pipeline(model) -> Pipeline:
    """Собирает полный пайплайн с генерацией признаков, предобработкой и оценкой модели"""
    fe_transformer = get_feature_engineering_transformer()
    preprocessor = create_preprocessor()

    return Pipeline([
        ("feature_engineering", fe_transformer),
        ("preprocessor", preprocessor),
        ("model", model)
    ])
