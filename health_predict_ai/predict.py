from __future__ import annotations

import pandas as pd

from health_predict_ai.explain import build_feature_contributions
from health_predict_ai.pipeline import load_bundle


def predict_risk(dataset_name: str, payload: dict[str, float | int]) -> dict[str, object]:
    bundle = load_bundle(dataset_name)
    model = bundle["estimator"]
    sample = pd.DataFrame([payload])
    prediction = int(model.predict(sample)[0])
    probability = float(model.predict_proba(sample)[0, 1])

    return {
        "dataset_name": dataset_name,
        "predicted_class": prediction,
        "risk_probability": round(probability, 4),
        "risk_label": "High Risk" if probability >= 0.5 else "Low Risk",
        "top_factors": build_feature_contributions(model, sample),
    }
