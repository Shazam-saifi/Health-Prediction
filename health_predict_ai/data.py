from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from health_predict_ai.config import DatasetConfig, RAW_DATA_DIR


def ensure_directories() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_heart_dataset(rows: int = 700, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(29, 78, rows)
    sex = rng.integers(0, 2, rows)
    resting_bp = rng.normal(132, 18, rows).clip(90, 220)
    cholesterol = rng.normal(238, 44, rows).clip(120, 420)
    max_heart_rate = rng.normal(151, 22, rows).clip(70, 210)
    st_depression = rng.normal(1.0, 1.2, rows).clip(0, 6)
    fasting_blood_sugar = rng.integers(0, 2, rows)
    exercise_angina = rng.integers(0, 2, rows)
    chest_pain_type = rng.integers(0, 4, rows)
    resting_ecg = rng.integers(0, 3, rows)

    risk_score = (
        0.06 * (age - 50)
        + 0.04 * (resting_bp - 130)
        + 0.03 * (cholesterol - 220)
        - 0.05 * (max_heart_rate - 145)
        + 0.9 * exercise_angina
        + 0.45 * fasting_blood_sugar
        + 0.3 * sex
        + 0.4 * chest_pain_type
        + 0.75 * st_depression
    )
    target = (risk_score + rng.normal(0, 1.0, rows) > 2.2).astype(int)

    return pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "chest_pain_type": chest_pain_type,
            "resting_bp": resting_bp.round(0).astype(int),
            "cholesterol": cholesterol.round(0).astype(int),
            "fasting_blood_sugar": fasting_blood_sugar,
            "resting_ecg": resting_ecg,
            "max_heart_rate": max_heart_rate.round(0).astype(int),
            "exercise_angina": exercise_angina,
            "st_depression": st_depression.round(2),
            "target": target,
        }
    )


def generate_diabetes_dataset(rows: int = 900, seed: int = 84) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 81, rows)
    bmi = rng.normal(29, 6, rows).clip(16, 52)
    glucose = rng.normal(124, 34, rows).clip(65, 290)
    blood_pressure = rng.normal(78, 14, rows).clip(40, 140)
    insulin = rng.normal(105, 55, rows).clip(15, 350)
    pregnancies = rng.integers(0, 13, rows)
    physical_activity = rng.integers(0, 2, rows)
    smoker = rng.integers(0, 2, rows)
    family_history = rng.integers(0, 2, rows)

    risk_score = (
        0.05 * (age - 45)
        + 0.12 * (bmi - 27)
        + 0.08 * (glucose - 110)
        + 0.02 * (blood_pressure - 75)
        + 0.018 * (insulin - 90)
        + 0.24 * pregnancies
        - 0.7 * physical_activity
        + 0.5 * smoker
        + 1.0 * family_history
    )
    diabetes = (risk_score + rng.normal(0, 1.2, rows) > 3.0).astype(int)

    return pd.DataFrame(
        {
            "age": age,
            "bmi": bmi.round(2),
            "glucose": glucose.round(0).astype(int),
            "blood_pressure": blood_pressure.round(0).astype(int),
            "insulin": insulin.round(0).astype(int),
            "pregnancies": pregnancies,
            "physical_activity": physical_activity,
            "smoker": smoker,
            "family_history": family_history,
            "diabetes": diabetes,
        }
    )


def create_demo_datasets() -> dict[str, pd.DataFrame]:
    ensure_directories()
    heart_df = generate_heart_dataset()
    diabetes_df = generate_diabetes_dataset()
    heart_df.to_csv(RAW_DATA_DIR / "heart.csv", index=False)
    diabetes_df.to_csv(RAW_DATA_DIR / "diabetes.csv", index=False)
    return {"heart_disease": heart_df, "diabetes": diabetes_df}


def load_dataset(config: DatasetConfig) -> pd.DataFrame:
    if config.csv_path.exists():
        return pd.read_csv(config.csv_path)

    generated = create_demo_datasets()
    return generated[config.name]


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
