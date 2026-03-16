from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import joblib
import pandas as pd

from health_predict_ai.config import ARTIFACTS_DIR, DatasetConfig, REPORTS_DIR
from health_predict_ai.data import load_dataset
from health_predict_ai.features import build_feature_summary
from health_predict_ai.modeling import ModelResult, train_candidate_models


@dataclass
class TrainingBundle:
    dataset_name: str
    target_column: str
    best_model_name: str
    metrics: dict[str, dict[str, float]]
    feature_summary: dict[str, object]
    feature_names: list[str]
    estimator: object


def _select_best_model(results: list[ModelResult]) -> ModelResult:
    return max(results, key=lambda result: result.metrics["roc_auc"])


def train_for_config(config: DatasetConfig) -> TrainingBundle:
    df = load_dataset(config)
    results, X_test, _ = train_candidate_models(df, config.target_column, config.random_state)
    best_result = _select_best_model(results)
    feature_summary = build_feature_summary(df, config.target_column)

    bundle = TrainingBundle(
        dataset_name=config.name,
        target_column=config.target_column,
        best_model_name=best_result.name,
        metrics={result.name: result.metrics for result in results},
        feature_summary=feature_summary,
        feature_names=X_test.columns.tolist(),
        estimator=best_result.pipeline,
    )
    return bundle


def save_bundle(bundle: TrainingBundle) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"{bundle.dataset_name}_bundle.joblib"
    joblib.dump(asdict(bundle), path)


def save_metrics_report(bundles: list[TrainingBundle]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        bundle.dataset_name: {
            "best_model": bundle.best_model_name,
            "metrics": bundle.metrics,
            "feature_summary": bundle.feature_summary,
        }
        for bundle in bundles
    }
    (REPORTS_DIR / "metrics_summary.json").write_text(json.dumps(payload, indent=2))


def load_bundle(dataset_name: str) -> dict[str, object]:
    path = ARTIFACTS_DIR / f"{dataset_name}_bundle.joblib"
    return joblib.load(path)
