from __future__ import annotations

from typing import Any

import pandas as pd


def build_feature_contributions(model: Any, sample: pd.DataFrame) -> list[dict[str, float | str]]:
    classifier = model.named_steps["classifier"]
    transformed = model.named_steps["preprocessor"].transform(sample)
    names = model.named_steps["preprocessor"].get_feature_names_out()

    if hasattr(classifier, "coef_"):
        raw_scores = transformed.toarray()[0] if hasattr(transformed, "toarray") else transformed[0]
        contributions = classifier.coef_[0] * raw_scores
        pairs = zip(names, contributions, strict=False)
    elif hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        pairs = zip(names, importances, strict=False)
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
