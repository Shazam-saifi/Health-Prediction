from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreprocessingArtifacts:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    X_train_processed: Optional[object] = None
    X_test_processed: Optional[object] = None
    X_train_resampled: Optional[object] = None
    y_train_resampled: Optional[object] = None


def detect_feature_types(df: pd.DataFrame, target_column: str) -> Tuple[List[str], List[str]]:
    """
    Detect numeric and categorical feature columns automatically.
    """
    X = df.drop(columns=[target_column])
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()
    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    Build preprocessing transformer:
    - numeric: median imputation + scaling
    - categorical: most frequent imputation + one-hot encoding
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def preprocess_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    use_smote: bool = True,
    random_state: int = 42,
) -> PreprocessingArtifacts:
    """
    Full preprocessing pipeline:
    1. Split features and target
    2. Detect numeric/categorical columns
    3. Handle missing values
    4. Encode categorical variables
    5. Scale numeric features
    6. Apply SMOTE to training data only
    """
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    numeric_features, categorical_features = detect_feature_types(train_df, target_column)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    X_train_resampled = X_train_processed
    y_train_resampled = y_train

    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

    return PreprocessingArtifacts(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        X_train_processed=X_train_processed,
        X_test_processed=X_test_processed,
        X_train_resampled=X_train_resampled,
        y_train_resampled=y_train_resampled,
    )
