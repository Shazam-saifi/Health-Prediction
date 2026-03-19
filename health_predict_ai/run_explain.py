from __future__ import annotations

from sklearn.model_selection import train_test_split

from health_predict_ai.config import HEART_CONFIG
from health_predict_ai.data import load_dataset
from health_predict_ai.explain import explain_from_bundle


def main() -> None:
    df = load_dataset(HEART_CONFIG)

    _, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=HEART_CONFIG.random_state,
        stratify=df[HEART_CONFIG.target_column],
    )

    sample_input = test_df.drop(columns=[HEART_CONFIG.target_column]).head(50)

    explain_from_bundle(
        bundle_path="artifacts/heart_disease_bundle.joblib",
        raw_input_df=sample_input,
        output_dir="reports/shap",
        row_index=0,
    )


if __name__ == "__main__":
    main()
