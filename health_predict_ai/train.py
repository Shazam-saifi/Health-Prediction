from __future__ import annotations

import json

from health_predict_ai.config import DIABETES_CONFIG, HEART_CONFIG
from health_predict_ai.pipeline import save_bundle, save_metrics_report, train_for_config


def main() -> None:
    bundles = [train_for_config(HEART_CONFIG), train_for_config(DIABETES_CONFIG)]
    for bundle in bundles:
        save_bundle(bundle)
    save_metrics_report(bundles)

    summary = {
        bundle.dataset_name: {
            "best_model": bundle.best_model_name,
            "metrics": bundle.metrics[bundle.best_model_name],
        }
        for bundle in bundles
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
