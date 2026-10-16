
"""Log the selected model and its metadata to MLflow."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from src.config import PROJECT_ROOT
from src.models.train import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)

logger = logging.getLogger(__name__)

MODEL_DIR = PROJECT_ROOT / "models" / "production"
TUNING_REPORT = PROJECT_ROOT / "reports" / "model_tuning.json"
EDA_DIR = PROJECT_ROOT / "reports" / "figures"

EXPERIMENT_NAME = "heart-disease-classification"
REGISTERED_MODEL_NAME = "heart-disease-classifier"


def calculate_sha256(path: Path) -> str:
    """Calculate checksum for model integrity verification."""
    sha = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)

    return sha.hexdigest()


def register_selected_model(model_path: Path) -> str:
    """Log and register the trusted packaged sklearn pipeline."""

    # Load only after the caller has verified the artifact checksum.
    pipeline = joblib.load(model_path)

    # Example input follows the original training feature schema.
    sample_path = (
        PROJECT_ROOT / "data" / "processed" / "heart_clean.csv"
    )

    if not sample_path.is_file():
        raise FileNotFoundError(
            f"Processed dataset required for model signature: {sample_path}"
        )

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    input_example = pd.read_csv(
        sample_path,
        usecols=features,
        nrows=5,
    )[features].astype("float64")

    if input_example.empty:
        raise ValueError("Cannot infer signature from an empty dataset")

    # Validate model loading and prediction.
    predictions = pipeline.predict(input_example)

    signature = infer_signature(input_example, predictions)

    model_info = mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="heart-disease-pipeline",
        signature=signature,
        input_example=input_example,
        registered_model_name=REGISTERED_MODEL_NAME,
        serialization_format="cloudpickle",
    )

    return model_info.model_uri


def log_selected_model(
    model_dir: Path = MODEL_DIR,
    tuning_path: Path = TUNING_REPORT,
    tracking_uri: str | None = None,
) -> str:
    """Track an existing packaged model in MLflow."""

    model_dir = Path(model_dir)
    tuning_path = Path(tuning_path)

    model_path = model_dir / "model.joblib"
    metadata_path = model_dir / "model_metadata.json"

    for path in (model_path, metadata_path, tuning_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required artifact missing: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))

    selected = metadata["model_name"]

    if selected != tuning["selected_model"]:
        raise ValueError("Packaged model differs from tuning selection")

    actual_hash = calculate_sha256(model_path)

    if actual_hash != metadata["model_sha256"]:
        raise ValueError("Model checksum verification failed")

    if tracking_uri is None:
        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5000",
        )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name=f"{selected}-packaged-v1"
    ) as run:

        mlflow.set_tags(
            {
                "project": "heart-disease-mlops",
                "stage": "packaging",
                "cloud_target": "gcp",
                "model_version": metadata["model_version"],
                "dataset_sha256": metadata["dataset_sha256"],
                "model_sha256": actual_hash,
                "evaluation_status": "exploratory_holdout",
            }
        )

        # Record the winning hyperparameters.
        mlflow.log_params(metadata["best_params"])

        # Record original model-selection evidence.
        mlflow.log_metric(
            "cv_roc_auc",
            metadata["cv_roc_auc"],
        )

        # Holdout results were previously inspected.
        # Do not represent them as unbiased final evaluation.
        for name, value in metadata[
            "holdout_metrics_exploratory"
        ].items():
            mlflow.log_metric(f"exploratory_{name}", value)

        mlflow.log_metric(
            "training_samples",
            metadata["training_samples"],
        )

        mlflow.log_metric(
            "holdout_samples",
            metadata["holdout_samples"],
        )

        # Record serialized model and reproducibility information.
        mlflow.log_artifact(
            str(model_path),
            artifact_path="model",
        )

        mlflow.log_artifact(
            str(metadata_path),
            artifact_path="model",
        )

        mlflow.log_artifact(
            str(tuning_path),
            artifact_path="reports",
        )

        if EDA_DIR.is_dir():
            mlflow.log_artifacts(
                str(EDA_DIR),
                artifact_path="eda",
            )

        model_uri = register_selected_model(model_path)

        mlflow.set_tag("registered_model", REGISTERED_MODEL_NAME)
        mlflow.set_tag("registered_model_uri", model_uri)

        logger.info("Registered MLflow model URI: %s", model_uri)

        logger.info("MLflow run created: %s", run.info.run_id)

        return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracking-uri",
        default=None,
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    run_id = log_selected_model(
        tracking_uri=args.tracking_uri,
    )

    print(f"MLflow Run ID: {run_id}")


if __name__ == "__main__":
    main()
