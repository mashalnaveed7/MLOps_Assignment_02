import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
import mlflow

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]


def main():
    with open(ROOT / "params.yaml", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    data_path = ROOT / params["data"]["path"]
    df = pd.read_csv(data_path)

    target_column = params["data"]["target_column"]
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["training"]["test_size"],
        random_state=params["training"]["random_state"],
        stratify=y,
    )

    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=params["models"]["logistic_regression"]["max_iter"]
            )),
        ]),
        "random_forest": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=params["models"]["random_forest"]["n_estimators"],
                max_depth=params["models"]["random_forest"]["max_depth"],
                random_state=params["training"]["random_state"],
            )),
        ]),
    }

    cv = StratifiedKFold(
        n_splits=params["training"]["cv_folds"],
        shuffle=True,
        random_state=params["training"]["random_state"],
    )
    scoring = {
        "malignant_recall": "recall",
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
    }

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("Breast Cancer Triage")

    results = {}
    best_name = None
    best_recall = -1.0

    for name, pipeline in models.items():
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        mean_recall = float(scores["test_malignant_recall"].mean())
        mean_accuracy = float(scores["test_accuracy"].mean())
        mean_auc = float(scores["test_roc_auc"].mean())

        results[name] = {
            "mean_cv_malignant_recall": mean_recall,
            "mean_cv_accuracy": mean_accuracy,
            "mean_cv_roc_auc": mean_auc,
        }

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_family", name)
            mlflow.log_param("cv_folds", params["training"]["cv_folds"])
            mlflow.log_param("test_size", params["training"]["test_size"])
            mlflow.log_param("random_state", params["training"]["random_state"])
            mlflow.log_metrics({
                "mean_cv_malignant_recall": mean_recall,
                "mean_cv_accuracy": mean_accuracy,
                "mean_cv_roc_auc": mean_auc,
            })

        print(
            f"{name}: CV malignant recall={mean_recall:.4f}, "
            f"accuracy={mean_accuracy:.4f}, ROC AUC={mean_auc:.4f}"
        )

        if mean_recall > best_recall:
            best_recall = mean_recall
            best_name = name

    best_model = clone(models[best_name])
    best_model.fit(X_train, y_train)

    predictions = best_model.predict(X_test)
    malignant_label = 0  # sklearn breast cancer dataset: 0 = malignant
    malignant_recall = recall_score(
        y_test, predictions, pos_label=malignant_label, zero_division=0
    )
    accuracy = accuracy_score(y_test, predictions)

    probabilities = best_model.predict_proba(X_test)
    malignant_index = list(best_model.classes_).index(malignant_label)
    malignant_scores = probabilities[:, malignant_index]
    malignant_truth = (y_test == malignant_label).astype(int)
    roc_auc = roc_auc_score(malignant_truth, malignant_scores)

    print(f"\nSelected model: {best_name}")
    print(f"Test malignant recall: {malignant_recall:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test ROC AUC: {roc_auc:.4f}")

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    joblib.dump(best_model, artifacts / "model.joblib")

    report = {
        "selected_model": best_name,
        "dataset_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "cv_results": results,
        "test_malignant_recall": float(malignant_recall),
        "test_accuracy": float(accuracy),
        "test_roc_auc": float(roc_auc),
    }
    (artifacts / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    if malignant_recall < params["training"]["minimum_malignant_recall"]:
        raise SystemExit(
            "ERROR: Test malignant recall is below the required threshold "
            f"{params['training']['minimum_malignant_recall']:.2f}"
        )


if __name__ == "__main__":
    main()
