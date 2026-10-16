
"""Reproducible cleaning for the UCI Heart Disease dataset."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.data.validation import (
    FEATURE_COLUMNS,
    RAW_DATA_PATH,
    validate_dataframe,
)

logger = logging.getLogger(__name__)

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "heart_clean.csv"
)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw data and prepare the binary classification target."""

    validate_dataframe(df)

    cleaned = df.copy()

    # Convert all values to numeric; preserve missing values as NaN.
    for column in (*FEATURE_COLUMNS, "num"):
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")

    # Remove exact duplicate records only.
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # UCI: 0 = absence; 1,2,3,4 = presence of heart disease.
    cleaned["target"] = (cleaned["num"] > 0).astype("int64")

    # The original target must not become an input feature.
    cleaned = cleaned.drop(columns=["num"])

    # Maintain consistent column ordering for downstream consumers.
    cleaned = cleaned[[*FEATURE_COLUMNS, "target"]]

    if cleaned.empty:
        raise ValueError("Cleaning produced an empty dataset")

    logger.info(
        "Cleaning complete: rows=%d, features=%d",
        len(cleaned),
        len(FEATURE_COLUMNS),
    )

    return cleaned


def process_dataset(
    input_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
) -> Path:
    """Read, clean, and atomically write a processed CSV."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {input_path}")

    raw = pd.read_csv(input_path, na_values=["?", ""])
    cleaned = clean_dataframe(raw)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")

    try:
        cleaned.to_csv(temporary_path, index=False)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    logger.info("Processed dataset saved to: %s", output_path)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RAW_DATA_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_PATH,
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    path = process_dataset(args.input, args.output)
    print(f"Processed dataset available at: {path}")


if __name__ == "__main__":
    main()
