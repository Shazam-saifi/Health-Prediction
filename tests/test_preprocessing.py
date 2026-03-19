from __future__ import annotations

import unittest

from sklearn.model_selection import train_test_split

from health_predict_ai.config import HEART_CONFIG
from health_predict_ai.data import load_dataset
from health_predict_ai.preprocessing import detect_feature_types, preprocess_data


class PreprocessingTests(unittest.TestCase):
    def test_detect_feature_types_splits_numeric_and_categorical(self) -> None:
        heart_df = load_dataset(HEART_CONFIG)

        numeric_features, categorical_features = detect_feature_types(
            heart_df,
            HEART_CONFIG.target_column,
        )

        self.assertIn("age", numeric_features)
        self.assertNotIn(HEART_CONFIG.target_column, numeric_features)
        self.assertEqual(categorical_features, [])

    def test_preprocess_data_returns_balanced_training_output(self) -> None:
        heart_df = load_dataset(HEART_CONFIG)
        train_df, test_df = train_test_split(
            heart_df,
            test_size=0.2,
            random_state=HEART_CONFIG.random_state,
            stratify=heart_df[HEART_CONFIG.target_column],
        )

        artifacts = preprocess_data(
            train_df=train_df,
            test_df=test_df,
            target_column=HEART_CONFIG.target_column,
            use_smote=True,
            random_state=HEART_CONFIG.random_state,
        )

        self.assertEqual(len(artifacts.X_train), len(artifacts.y_train))
        self.assertEqual(len(artifacts.X_test), len(artifacts.y_test))
        self.assertGreaterEqual(len(artifacts.y_train_resampled), len(artifacts.y_train))
        self.assertEqual(artifacts.X_test_processed.shape[0], len(artifacts.y_test))


if __name__ == "__main__":
    unittest.main()
