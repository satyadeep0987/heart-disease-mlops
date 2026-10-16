
"""Orchestrate the complete Heart Disease data pipeline."""

import argparse
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from src.config import PROJECT_ROOT
from src.data.download import download_dataset
from src.data.eda import run_eda
from src.data.preprocessing import process_dataset
from src.data.validation import validate_csv

logger = logging.getLogger(__name__)

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "heart_uci_raw.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "heart_clean.csv"
FIGURES_PATH = PROJECT_ROOT / "reports" / "figures"
MANIFEST_PATH = PROJECT_ROOT / "reports" / "data_pipeline_manifest.json"


def calculate_sha256(file_path: Path) -> str:
    """Calculate a reproducible SHA-256 checksum."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def run_pipeline(
    raw_path: Path = RAW_PATH,
    processed_path: Path = PROCESSED_PATH,
    figures_path: Path = FIGURES_PATH,
    manifest_path: Path = MANIFEST_PATH,
    force_download: bool = False,
) -> dict:
    """Run data acquisition, validation, cleaning and EDA."""

    logger.info("Starting Heart Disease data pipeline")

    # Step 1: Download or reuse the cached dataset.
    logger.info("Step 1/5: Dataset acquisition")
    raw_path = download_dataset(
        output_path=raw_path,
        force=force_download,
    )

    # Step 2: Validate raw data before modifying it.
    logger.info("Step 2/5: Data validation")
    validation_summary = validate_csv(raw_path)

    # Step 3: Clean and convert the target.
    logger.info("Step 3/5: Data preprocessing")
    processed_path = process_dataset(
        input_path=raw_path,
        output_path=processed_path,
    )

    # Step 4: Generate figures and statistical summaries.
    logger.info("Step 4/5: Exploratory data analysis")
    eda_summary = run_eda(
        input_path=processed_path,
        output_dir=figures_path,
    )

    # Step 5: Record execution metadata.
    logger.info("Step 5/5: Generating pipeline manifest")

    manifest = {
        "pipeline": "heart-disease-data-pipeline",
        "status": "success",
        "executed_at_utc": datetime.now(UTC).isoformat(),
        "dataset_id": 45,
        "raw_data": {
            "path": str(raw_path),
            "sha256": calculate_sha256(raw_path),
            "rows": validation_summary["rows"],
        },
        "processed_data": {
            "path": str(processed_path),
            "sha256": calculate_sha256(processed_path),
            "rows": eda_summary["total_rows"],
            "features": eda_summary["feature_count"],
        },
        "target_distribution": eda_summary["class_distribution"],
        "eda_directory": str(figures_path),
    }

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = manifest_path.with_suffix(".tmp")

    try:
        temp_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(manifest_path)
    finally:
        temp_path.unlink(missing_ok=True)

    logger.info("Pipeline completed successfully")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end Heart Disease data pipeline"
    )

    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download the dataset again instead of using the cache",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    result = run_pipeline(force_download=args.force_download)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
