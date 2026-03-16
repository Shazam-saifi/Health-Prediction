from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    target_column: str
    csv_path: Path
    random_state: int


HEART_CONFIG = DatasetConfig(
    name="heart_disease",
    target_column="target",
    csv_path=RAW_DATA_DIR / "heart.csv",
    random_state=42,
)

DIABETES_CONFIG = DatasetConfig(
    name="diabetes",
    target_column="diabetes",
    csv_path=RAW_DATA_DIR / "diabetes.csv",
    random_state=84,
)

