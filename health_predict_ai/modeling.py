from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ModelResult:
    name: str
    pipeline: ImbPipeline
    metrics: dict[str, float]


def split_features_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[target_column]), df[target_column]


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
    }


def train_candidate_models(
    df: pd.DataFrame,
    target_column: str,
    random_state: int,
) -> tuple[list[ModelResult], pd.DataFrame, pd.Series]:
    features, target = split_features_target(df, target_column)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
        stratify=target,
    )
    preprocessor = build_preprocessor(features)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight="balanced",
        ),
    }

    results: list[ModelResult] = []
    for name, estimator in candidates.items():
        pipeline = ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=random_state)),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate_predictions(y_test, predictions, probabilities)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        metrics["cv_roc_auc_mean"] = round(float(cv_scores.mean()), 4)
        metrics["cv_roc_auc_std"] = round(float(cv_scores.std()), 4)
        results.append(ModelResult(name=name, pipeline=pipeline, metrics=metrics))

    return results, X_test, y_test
