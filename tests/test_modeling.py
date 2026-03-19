from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from health_predict_ai.config import DIABETES_CONFIG, HEART_CONFIG
from health_predict_ai.pipeline import save_best_model_pickle, save_bundle, train_for_config


class ModelingTests(unittest.TestCase):
    def test_train_for_config_returns_comparison_metrics(self) -> None:
        heart_bundle = train_for_config(HEART_CONFIG)
        diabetes_bundle = train_for_config(DIABETES_CONFIG)

        self.assertIn(heart_bundle.best_model_name, {"logistic_regression", "random_forest"})
        self.assertIn(diabetes_bundle.best_model_name, {"logistic_regression", "random_forest"})
        self.assertIn("roc_auc", heart_bundle.metrics[heart_bundle.best_model_name])
        self.assertIn("roc_auc", diabetes_bundle.metrics[diabetes_bundle.best_model_name])
        self.assertIn("comparison_cv_accuracy", heart_bundle.metrics["logistic_regression"])
        self.assertIn("comparison_cv_accuracy", diabetes_bundle.metrics["random_forest"])

    def test_save_bundle_writes_expected_artifacts(self) -> None:
        heart_bundle = train_for_config(HEART_CONFIG)
        diabetes_bundle = train_for_config(DIABETES_CONFIG)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_dir = Path(tmpdir)
            with patch("health_predict_ai.pipeline.ARTIFACTS_DIR", artifacts_dir):
                save_bundle(heart_bundle)
                save_bundle(diabetes_bundle)
                save_best_model_pickle([heart_bundle, diabetes_bundle])

            self.assertTrue((artifacts_dir / "heart_disease_bundle.joblib").exists())
            self.assertTrue((artifacts_dir / "diabetes_bundle.joblib").exists())
            self.assertTrue((artifacts_dir / "heart_disease_model.pkl").exists())
            self.assertTrue((artifacts_dir / "diabetes_model.pkl").exists())
            self.assertTrue((artifacts_dir / "model.pkl").exists())

            with (artifacts_dir / "model.pkl").open("rb") as handle:
                model = pickle.load(handle)
            self.assertTrue(hasattr(model, "predict"))


if __name__ == "__main__":
    unittest.main()
