from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
except ModuleNotFoundError:  # pragma: no cover
    plt = None
    shap = None


def load_bundle(bundle_path: str) -> dict[str, Any]:
    """
    Load a saved training bundle.
    Expected bundle structure can include:
    - model
    - preprocessor
    - feature_names
    """
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")
    return joblib.load(bundle_path)


def get_transformed_feature_names(preprocessor: Any, input_df: pd.DataFrame) -> list[str]:
    """
    Extract transformed feature names from a fitted ColumnTransformer.
    Falls back safely if names are not available.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"feature_{i}" for i in range(preprocessor.transform(input_df).shape[1])]


def prepare_explainer(model: Any, X_background: Any) -> Any:
    """
    Build a SHAP explainer for the trained model.
    TreeExplainer is preferred for tree-based models.
    """
    if shap is None:
        raise ModuleNotFoundError(
            "SHAP is not installed. Run `pip install -r requirements.txt` to enable explainability."
        )

    model_name = model.__class__.__name__.lower()

    if "forest" in model_name or "tree" in model_name or "xgb" in model_name:
        return shap.TreeExplainer(model)

    return shap.Explainer(model, X_background)


def explain_global(
    model: Any,
    X_sample: Any,
    feature_names: list[str],
    output_dir: str = "reports/shap",
    max_display: int = 10,
) -> None:
    """
    Generate and save a SHAP summary plot for global feature importance.
    """
    if shap is None or plt is None:
        raise ModuleNotFoundError(
            "SHAP and matplotlib are required to generate explainability plots."
        )

    os.makedirs(output_dir, exist_ok=True)

    explainer = prepare_explainer(model, X_sample)
    shap_values = explainer(X_sample)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
        max_display=max_display,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"), bbox_inches="tight")
    plt.close()


def explain_local(
    model: Any,
    X_sample: Any,
    feature_names: list[str],
    row_index: int = 0,
    output_dir: str = "reports/shap",
) -> None:
    """
    Generate and save a SHAP waterfall plot for one prediction.
    """
    if shap is None or plt is None:
        raise ModuleNotFoundError(
            "SHAP and matplotlib are required to generate explainability plots."
        )

    os.makedirs(output_dir, exist_ok=True)

    explainer = prepare_explainer(model, X_sample)
    shap_values = explainer(X_sample)

    plt.figure()
    shap.plots.waterfall(shap_values[row_index], show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"shap_waterfall_row_{row_index}.png"), bbox_inches="tight")
    plt.close()


def explain_from_bundle(
    bundle_path: str,
    raw_input_df: pd.DataFrame,
    output_dir: str = "reports/shap",
    row_index: int = 0,
) -> None:
    """
    Full explainability workflow using a saved bundle.
    """
    bundle = load_bundle(bundle_path)

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]

    X_processed = preprocessor.transform(raw_input_df)
    feature_names = get_transformed_feature_names(preprocessor, raw_input_df)

    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    explain_global(
        model=model,
        X_sample=X_processed,
        feature_names=feature_names,
        output_dir=output_dir,
    )

    explain_local(
        model=model,
        X_sample=X_processed,
        feature_names=feature_names,
        row_index=row_index,
        output_dir=output_dir,
    )

    print(f"SHAP outputs saved in: {output_dir}")


def build_feature_contributions(model: Any, sample: pd.DataFrame) -> list[dict[str, float | str]]:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    transformed = preprocessor.transform(sample)
    feature_names = get_transformed_feature_names(preprocessor, sample)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    if shap is not None:
        try:
            explainer = prepare_explainer(classifier, transformed)
            shap_values = explainer(transformed)
            local_values = getattr(shap_values, "values", shap_values)
            if isinstance(local_values, list):
                local_values = local_values[1] if len(local_values) > 1 else local_values[0]
            local_row = np.asarray(local_values[0])
            ranked = sorted(
                (
                    {"feature": str(name), "impact": round(float(value), 4)}
                    for name, value in zip(feature_names, local_row, strict=False)
                ),
                key=lambda item: abs(item["impact"]),
                reverse=True,
            )
            if ranked:
                return ranked[:5]
        except Exception:
            pass

    if hasattr(classifier, "coef_"):
        raw_scores = transformed[0]
        contributions = classifier.coef_[0] * raw_scores
        pairs = zip(feature_names, contributions, strict=False)
    elif hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        pairs = zip(feature_names, importances, strict=False)
    else:
        return []

    ranked = sorted(
        (
            {"feature": str(name), "impact": round(float(value), 4)}
            for name, value in pairs
        ),
        key=lambda item: abs(item["impact"]),
        reverse=True,
    )
    return ranked[:5]
