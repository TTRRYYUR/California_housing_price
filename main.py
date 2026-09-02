import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from data import clean_data, load_data
from pipelines import create_model_pipeline
from training import evaluate_model, get_model_config, train_model
from visualization import plot_confusion_matrix, plot_feature_importance, plot_roc_curves


def main():
    raw_df = load_data("bank.csv")
    df = clean_data(raw_df)

    X = df.drop("deposit", axis=1)
    y = df["deposit"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_config = get_model_config()
    results = {}
    eval_results = {}

    for name, config in model_config.items():
        print(f"Обучение {name}...")

        pipeline = create_model_pipeline(config["model"])
        train_res = train_model(pipeline, X_train, y_train, config["params"])
        best_pipe = train_res["pipeline"]

        eval_res = evaluate_model(best_pipe, X_test, y_test)

        results[name] = {
            "best_params": train_res["params"],
            "cv_roc_auc": train_res["cv_score"],
            "test_roc_auc": eval_res["roc_auc"],
            "f1": eval_res["f1"],
            "accuracy": eval_res["accuracy"]
        }

        eval_results[name] = {
            "pipeline": best_pipe,
            "eval": eval_res,
            "y_test": y_test
        }

    summary_df = pd.DataFrame(results).T.sort_values(by="test_roc_auc", ascending=False)
    print("\nИтоговое сравнение моделей")
    print(summary_df[["cv_roc_auc", "test_roc_auc", "f1", "accuracy"]])

    best_model_name = summary_df.index[0]
    best_pipeline = eval_results[best_model_name]["pipeline"]
    print(f"\nЛучшая модель: {best_model_name}")

    joblib.dump(best_pipeline, "best_bank_model.pkl")

    plot_roc_curves(eval_results)

    best_pred = eval_results[best_model_name]["eval"]["predictions"]
    plot_confusion_matrix(y_test, best_pred, best_model_name)

    plot_feature_importance(best_pipeline, X_test, y_test)

if __name__ == "__main__":
    main()
