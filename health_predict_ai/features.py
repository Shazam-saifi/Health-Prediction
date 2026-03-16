from __future__ import annotations

import pandas as pd


def build_feature_summary(df: pd.DataFrame, target_column: str) -> dict[str, object]:
    feature_df = df.drop(columns=[target_column])
    numeric_columns = feature_df.select_dtypes(include=["number"]).columns.tolist()
    return {
        "row_count": int(df.shape[0]),
        "feature_count": int(feature_df.shape[1]),
        "numeric_columns": numeric_columns,
        "target_balance": df[target_column].value_counts(normalize=True).sort_index().to_dict(),
    }
