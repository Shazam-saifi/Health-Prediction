from __future__ import annotations

import unittest

from health_predict_ai.predict import predict_risk


class PredictTests(unittest.TestCase):
    def test_predict_risk_returns_expected_payload(self) -> None:
        payload = {
            "age": 54,
            "sex": 1,
            "chest_pain_type": 2,
            "resting_bp": 138,
            "cholesterol": 245,
            "fasting_blood_sugar": 0,
            "resting_ecg": 1,
            "max_heart_rate": 150,
            "exercise_angina": 0,
            "st_depression": 1.2,
        }

        result = predict_risk("heart_disease", payload)

        self.assertEqual(result["dataset_name"], "heart_disease")
        self.assertIn(result["predicted_class"], {0, 1})
        self.assertGreaterEqual(result["risk_probability"], 0.0)
        self.assertLessEqual(result["risk_probability"], 1.0)
        self.assertIn(result["risk_label"], {"Low Risk", "High Risk"})
        self.assertIsInstance(result["top_factors"], list)


if __name__ == "__main__":
    unittest.main()
