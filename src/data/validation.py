
"""Validate the schema and values of the UCI Heart Disease dataset."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_uci_raw.csv"

FEATURE_COLUMNS = (
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
)

TARGET_COLUMN = "num"
REQUIRED_COLUMNS = (*FEATURE_COLUMNS, TARGET_COLUMN)

# Conservative integrity checks, not clinical reference ranges.
VALUE_RANGES = {
    "age": (1, 120),
    "sex": (0, 1),
    "cp": (1, 4),
    "trestbps": (1, 350),
    "chol": (1, 800),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (1, 250),
    "exang": (0, 1),
    "oldpeak": (0, 10),
    "slope": (1, 3),
    "ca": (0, 3),
    "thal": (3, 7),
}

ALLOWED_TARGETS = {0, 1, 2, 3, 4}


def validate_dataframe(df: pd.DataFrame) -> dict:
    """Validate raw UCI schema and return a quality summary."""

    if df.empty:
        raise ValueError("Dataset is empty")

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    unexpected_columns = sorted(set(df.columns) - set(REQUIRED_COLUMNS))

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if unexpected_columns:
        raise ValueError(f"Unexpected columns: {unexpected_columns}")

    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names found")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target contains null values")

    # UCI raw feature values should be numeric or missing.
    for column in REQUIRED_COLUMNS:
        values = df[column].dropna()

        numeric = pd.to_numeric(values, errors="coerce")

        if numeric.isna().any():
            raise ValueError(f"Non-numeric values found in {column}")

        if column in VALUE_RANGES:
            minimum, maximum = VALUE_RANGES[column]

            invalid = ~numeric.between(minimum, maximum)
            if invalid.any():
                raise ValueError(
                    f"Values outside expected range in {column}"
                )

    labels = set(pd.to_numeric(df[TARGET_COLUMN]).unique())

    if not labels.issubset(ALLOWED_TARGETS):
        raise ValueError(f"Unexpected target labels: {labels}")

    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {
            column: int(count)
            for column, count in df.isna().sum().items()
        },
        "target_distribution": {
            str(label): int(count)
            for label, count in df[TARGET_COLUMN].value_counts().items()
        },
    }

    logger.info("Validation passed: %d rows", len(df))
    return summary


def validate_csv(path: Path = RAW_DATA_PATH) -> dict:
    """Read raw CSV and validate its contents."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, na_values=["?", ""])
    return validate_dataframe(df)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RAW_DATA_PATH)
    args = parser.parse_args()

    summary = validate_csv(args.input)

    import json

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
