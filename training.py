from typing import Any
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score


def get_model_config():
    """Возвращает конфигурацию моделей и пространства их гиперпараметров для оптимизации"""
    return {
        "Logistic_Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {"model__C": loguniform(1e-3, 1e2)}
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "model__n_estimators": randint(50, 300),
                "model__max_depth": randint(5, 25),
                "model__min_samples_split": randint(2, 6)
            }
        },
        "LightGBM": {
            "model": LGBMClassifier(random_state=42, verbose=-1),
            "params": {
                "model__n_estimators": randint(100, 400),
                "model__max_depth": randint(3, 15),
                "model__num_leaves": randint(10, 70),
                "model__learning_rate": uniform(0.01, 0.2),
                "model__subsample": uniform(0.7, 0.3)
            }
        }
    }


def train_model(pipeline, X_train, y_train, param_grid):
    """Обучает пайплайн с подбором параметров или оценкой через кросс-валидацию"""
    if param_grid:
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=15,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42
        )
        search.fit(X_train, y_train)

        return {
            "pipeline": search.best_estimator_,
            "params": search.best_params_,
            "cv_score": search.best_score_
        }
    else:
        pipeline.fit(X_train, y_train)
        cv_score = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='roc_auc')
        return {
            "pipeline": pipeline,
            "params": "default",
            "cv_score": cv_score.mean()
        }


def evaluate_model(pipeline, X_test, y_test):
    """Вычисляет основные метрики качества модели на тестовой выборке"""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "predictions": y_pred,
        "probabilities": y_proba
    }
