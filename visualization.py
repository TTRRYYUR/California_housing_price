import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve


def plot_roc_curves(eval_results) :
    """Строит и отображает ROC-кривые для всех переданных моделей"""
    plt.figure(figsize=(8, 6))

    for model_name, res in eval_results.items():
        y_test = res["y_test"]
        y_proba = res["eval"]["probabilities"]

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = res["eval"]["roc_auc"]

        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label="Случайное угадывание (AUC = 0.500)")
    plt.title("Сравнение ROC-кривых", fontsize=14, fontweight="bold")
    plt.xlabel("Доля ложных исходов", fontsize=12)
    plt.ylabel("Доля положительных исходов", fontsize=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_test, y_pred, model_name) -> None:
    """Строит и отображает матрицу ошибок для выбранной модели"""
    plt.figure(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["No Deposit", "Deposit"],
        cmap="Blues",
        colorbar=False
    )

    plt.title(f"Матрица ошибок({model_name})", fontsize=14, fontweight="bold")
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(pipeline, X_test, y_test, top_n = 10):
    """Оценивает и отображает график относительной важности признаков"""
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
        scoring="roc_auc"
    )

    importances = np.maximum(0, result.importances_mean)
    total_imp = importances.sum()

    if total_imp > 0:
        importances_pct = (importances / total_imp) * 100
    else:
        importances_pct = importances

    importance_df = pd.DataFrame({
        "Признак": X_test.columns,
        "Важность_%": importances_pct
    }).sort_values("Важность_%", ascending=False).head(top_n)

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=importance_df,
        x="Важность_%",
        y="Признак",
        palette="viridis"
    )

    plt.title(f"Top-{top_n} Feature Importances (Permutation)", fontsize=14, fontweight="bold")
    plt.xlabel("Важность (%)", fontsize=11)
    plt.ylabel("Признак", fontsize=11)
    plt.tight_layout()
    plt.show()

    return importance_df
