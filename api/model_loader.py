
"""Load and cache the trained model for FastAPI inference."""

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "production" / "model.joblib"
)

DEFAULT_METADATA_PATH = (
    PROJECT_ROOT / "models" / "production" / "model_metadata.json"
)

DEFAULT_MLFLOW_URI = (
    "models:/heart-disease-classifier@candidate"
)


@dataclass(frozen=True)
class LoadedModel:
    """Container for the fitted pipeline and its identity."""

    pipeline: Any
    name: str
    version: str
    source: str


def calculate_sha256(path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def validate_pipeline(pipeline: Any) -> None:
    """Ensure the loaded model supports required inference methods."""

    for method in ("predict", "predict_proba"):
        if not callable(getattr(pipeline, method, None)):
            raise TypeError(
                f"Loaded model does not implement {method}()"
            )


def load_local_model(
    model_path: Path = DEFAULT_MODEL_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> LoadedModel:
    """Load a trusted, checksum-verified local model."""

    model_path = Path(model_path)
    metadata_path = Path(metadata_path)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Model metadata not found: {metadata_path}"
        )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    expected_hash = metadata["model_sha256"]
    actual_hash = calculate_sha256(model_path)

    if actual_hash != expected_hash:
        raise ValueError("Model checksum verification failed")

    # Only load artifacts from our trusted model build pipeline.
    pipeline = joblib.load(model_path)

    validate_pipeline(pipeline)

    logger.info("Local model loaded successfully")

    return LoadedModel(
        pipeline=pipeline,
        name=metadata["model_name"],
        version=metadata["model_version"],
        source="local",
    )


def load_mlflow_model(
    model_uri: str = DEFAULT_MLFLOW_URI,
    tracking_uri: str | None = None,
) -> LoadedModel:
    """Load a registered model from the configured MLflow server."""

    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    uri = (
        tracking_uri
        or os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5000",
        )
    )

    mlflow.set_tracking_uri(uri)

    logger.info("Loading MLflow model: %s", model_uri)

    pipeline = mlflow.sklearn.load_model(model_uri)

    validate_pipeline(pipeline)

    # Resolve the exact registered version for reproducibility.
    model_name = "heart-disease-classifier"
    client = MlflowClient(tracking_uri=uri)

    if model_uri == DEFAULT_MLFLOW_URI:
        version = client.get_model_version_by_alias(
            model_name,
            "candidate",
        ).version
    else:
        # This loader currently supports the candidate alias.
        raise ValueError(
            "Unsupported MLflow model URI for version resolution"
        )

    logger.info(
        "MLflow model loaded: %s version %s",
        model_name,
        version,
    )

    return LoadedModel(
        pipeline=pipeline,
        name=model_name,
        version=str(version),
        source="mlflow",
    )


def load_model() -> LoadedModel:
    """Choose the model source using environment configuration."""

    source = os.getenv("MODEL_SOURCE", "local").lower()

    if source == "local":
        return load_local_model()

    if source == "mlflow":
        return load_mlflow_model()

    raise ValueError(
        "MODEL_SOURCE must be either 'local' or 'mlflow'"
    )
