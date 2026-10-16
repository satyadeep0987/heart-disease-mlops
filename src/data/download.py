
"""Download and cache the original UCI Heart Disease dataset."""

import argparse
import logging
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

DATASET_ID = 45
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_uci_raw.csv"


def download_dataset(
    output_path: Path = RAW_DATA_PATH,
    force: bool = False,
) -> Path:
    """Download the dataset only if it is not already cached."""

    output_path = Path(output_path)

    if output_path.exists() and not force:
        logger.info("Using cached dataset: %s", output_path)
        return output_path

    logger.info("Fetching Heart Disease dataset from UCI")

    dataset = fetch_ucirepo(id=DATASET_ID)

    features = dataset.data.features
    targets = dataset.data.targets

    if features is None or targets is None:
        raise ValueError("UCI returned an incomplete dataset")

    if features.empty or targets.empty:
        raise ValueError("UCI returned an empty dataset")

    if len(features) != len(targets):
        raise ValueError("Feature and target row counts differ")

    dataframe = pd.concat(
        [
            features.reset_index(drop=True),
            targets.reset_index(drop=True),
        ],
        axis=1,
    )

    if "num" not in dataframe.columns:
        raise ValueError("Expected target column 'num' not found")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write atomically so a failed write cannot corrupt cached data.
    temp_path = output_path.with_suffix(".tmp")

    try:
        dataframe.to_csv(temp_path, index=False)
        temp_path.replace(output_path)
    finally:
        temp_path.unlink(missing_ok=True)

    logger.info(
        "Saved %d rows and %d columns to %s",
        len(dataframe),
        len(dataframe.columns),
        output_path,
    )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even if a cached file exists",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    path = download_dataset(force=args.force)
    print(f"Dataset available at: {path}")


if __name__ == "__main__":
    main()
