from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from health_predict_ai.plots import compute_roc_auc_summary, plot_roc_curves
from health_predict_ai.preprocessing import preprocess_data


@dataclass
class ModelResult:
    name: str
    pipeline: ImbPipeline
    metrics: dict[str, float]


@dataclass
class ComparisonMetrics:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    cv_mean: float


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
    }


def evaluate_model(
    model: Any,
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    model_name: str,
) -> tuple[Any, ComparisonMetrics]:
    """
    Train a model, evaluate it, and return both the fitted model and metrics.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

    result = ComparisonMetrics(
        model_name=model_name,
        accuracy=round(float(accuracy_score(y_test, y_pred)), 4),
        precision=round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        recall=round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        f1_score=round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        roc_auc=round(float(roc_auc_score(y_test, y_prob)), 4),
        cv_mean=round(float(np.mean(cv_scores)), 4),
    )

    return model, result


def compare_models(
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Compare Logistic Regression and Random Forest on the same dataset.
    """
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight="balanced",
        ),
    }

    fitted_models: dict[str, Any] = {}
    results: dict[str, dict[str, float | str]] = {}

    print("\nModel Comparison Results")
    print("-" * 80)
    print(
        f"{'Model':<22} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} "
        f"{'F1-score':<10} {'ROC-AUC':<10} {'CV Mean':<10}"
    )
    print("-" * 80)

    for model_name, model in models.items():
        fitted_model, metrics_result = evaluate_model(
            model=clone(model),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
        )
        fitted_models[model_name] = fitted_model
        results[model_name] = asdict(metrics_result)

        print(
            f"{model_name:<22} "
            f"{metrics_result.accuracy:<10.3f} "
            f"{metrics_result.precision:<10.3f} "
            f"{metrics_result.recall:<10.3f} "
            f"{metrics_result.f1_score:<10.3f} "
            f"{metrics_result.roc_auc:<10.3f} "
            f"{metrics_result.cv_mean:<10.3f}"
        )

    best_model_name = max(results, key=lambda name: float(results[name]["roc_auc"]))
    print("\nBest model based on ROC-AUC:", best_model_name)

    return {
        "fitted_models": fitted_models,
        "results": results,
        "best_model_name": best_model_name,
    }


def train_candidate_models(
    df: pd.DataFrame,
    target_column: str,
    random_state: int,
    dataset_name: str | None = None,
) -> tuple[list[ModelResult], pd.DataFrame, pd.Series]:
    # Step 1: Split dataset into train and test sets
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=random_state,
        stratify=df[target_column],
    )

    # Step 2: Detect numeric and categorical features
    # Step 3: Handle missing values
    # Step 4: Encode categorical variables
    # Step 5: Scale numeric features
    # Step 6: Apply SMOTE to balance the training data
    artifacts = preprocess_data(
        train_df=train_df,
        test_df=test_df,
        target_column=target_column,
        use_smote=True,
        random_state=random_state,
    )

    # Step 7: Train machine learning models
    comparison_output = compare_models(
        X_train=artifacts.X_train_resampled,
        y_train=artifacts.y_train_resampled,
        X_test=artifacts.X_test_processed,
        y_test=artifacts.y_test,
        random_state=random_state,
    )
    plot_roc_curves(
        fitted_models=comparison_output["fitted_models"],
        X_test=artifacts.X_test_processed,
        y_test=artifacts.y_test,
        dataset_name=(dataset_name or target_column).replace("_", " ").title(),
    )
    roc_scores = compute_roc_auc_summary(
        fitted_models=comparison_output["fitted_models"],
        X_test=artifacts.X_test_processed,
        y_test=artifacts.y_test,
    )
    print("ROC-AUC Summary:", roc_scores)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    results: list[ModelResult] = []
    for name, fitted_estimator in comparison_output["fitted_models"].items():
        predictions = fitted_estimator.predict(artifacts.X_test_processed)
        probabilities = fitted_estimator.predict_proba(artifacts.X_test_processed)[:, 1]
        metrics = evaluate_predictions(artifacts.y_test, predictions, probabilities)

        if name == "logistic_regression":
            estimator = LogisticRegression(
                max_iter=2000,
                random_state=random_state,
                class_weight="balanced",
            )
        else:
            estimator = RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=2,
                random_state=random_state,
                class_weight="balanced",
            )
        evaluation_pipeline = ImbPipeline(
            steps=[
                ("preprocessor", clone(artifacts.preprocessor)),
                ("smote", SMOTE(random_state=random_state)),
                ("classifier", clone(estimator)),
            ]
        )
        cv_scores = cross_val_score(
            evaluation_pipeline,
            artifacts.X_train,
            artifacts.y_train,
            cv=cv,
            scoring="roc_auc",
        )
        metrics["cv_roc_auc_mean"] = round(float(cv_scores.mean()), 4)
        metrics["cv_roc_auc_std"] = round(float(cv_scores.std()), 4)
        metrics["comparison_cv_accuracy"] = float(
            comparison_output["results"][name]["cv_mean"]
        )

        deployment_pipeline = ImbPipeline(
            steps=[
                ("preprocessor", clone(artifacts.preprocessor)),
                ("smote", SMOTE(random_state=random_state)),
                ("classifier", clone(estimator)),
            ]
        )
        deployment_pipeline.fit(artifacts.X_train, artifacts.y_train)
        results.append(ModelResult(name=name, pipeline=deployment_pipeline, metrics=metrics))

    return results, artifacts.X_test, artifacts.y_test
