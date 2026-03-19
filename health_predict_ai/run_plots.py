from __future__ import annotations

from sklearn.model_selection import train_test_split

from health_predict_ai.config import HEART_CONFIG
from health_predict_ai.data import load_dataset
from health_predict_ai.modeling import compare_models
from health_predict_ai.plots import plot_roc_curves
from health_predict_ai.preprocessing import preprocess_data


def main() -> None:
    df = load_dataset(HEART_CONFIG)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=HEART_CONFIG.random_state,
        stratify=df[HEART_CONFIG.target_column],
    )

    artifacts = preprocess_data(
        train_df=train_df,
        test_df=test_df,
        target_column=HEART_CONFIG.target_column,
        use_smote=True,
        random_state=HEART_CONFIG.random_state,
    )

    comparison_output = compare_models(
        X_train=artifacts.X_train_resampled,
        y_train=artifacts.y_train_resampled,
        X_test=artifacts.X_test_processed,
        y_test=artifacts.y_test,
        random_state=HEART_CONFIG.random_state,
    )

    plot_roc_curves(
        fitted_models=comparison_output["fitted_models"],
        X_test=artifacts.X_test_processed,
        y_test=artifacts.y_test,
        dataset_name="Heart Disease",
    )


if __name__ == "__main__":
    main()
