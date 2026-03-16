from __future__ import annotations

import unittest

from health_predict_ai.config import DIABETES_CONFIG, HEART_CONFIG
from health_predict_ai.pipeline import train_for_config


class TrainingPipelineTests(unittest.TestCase):
    def test_train_for_config_returns_bundle(self) -> None:
        heart_bundle = train_for_config(HEART_CONFIG)
        diabetes_bundle = train_for_config(DIABETES_CONFIG)

        self.assertIn(heart_bundle.best_model_name, {"logistic_regression", "random_forest"})
        self.assertIn(diabetes_bundle.best_model_name, {"logistic_regression", "random_forest"})
        self.assertIn("roc_auc", heart_bundle.metrics[heart_bundle.best_model_name])
        self.assertIn("roc_auc", diabetes_bundle.metrics[diabetes_bundle.best_model_name])


if __name__ == "__main__":
    unittest.main()
