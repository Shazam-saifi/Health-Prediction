from __future__ import annotations

import os
from typing import Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None

from sklearn.metrics import RocCurveDisplay, roc_auc_score


def plot_roc_curves(
    fitted_models: dict[str, Any],
    X_test: Any,
    y_test: Any,
    dataset_name: str,
    output_dir: str = "reports/plots",
) -> None:
    """
    Plot ROC curves for multiple fitted models and save the figure.
    """
    if plt is None:
        raise ModuleNotFoundError(
            "matplotlib is required to generate ROC curve plots."
        )

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))

    for model_name, model in fitted_models.items():
        RocCurveDisplay.from_estimator(
            model,
            X_test,
            y_test,
            name=model_name,
        )

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"ROC Curve Comparison - {dataset_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    filename = f"{dataset_name.lower().replace(' ', '_')}_roc_curve.png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.close()


def compute_roc_auc_summary(
    fitted_models: dict[str, Any],
    X_test: Any,
    y_test: Any,
) -> dict[str, float]:
    """
    Return ROC-AUC scores for all fitted models.
    """
    scores: dict[str, float] = {}

    for model_name, model in fitted_models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict(X_test)

        scores[model_name] = round(float(roc_auc_score(y_test, y_prob)), 4)

    return scores
