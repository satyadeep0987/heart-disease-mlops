
"""Package the selected model with preprocessing and reproducibility metadata."""

import argparse
import hashlib
import json
import logging
import os
import platform
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config import PROJECT_ROOT
from src.models.train import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    create_models,
    create_preprocessor,
    load_dataset,
)

logger = logging.getLogger(__name__)

TUNING_PATH = PROJECT_ROOT / "reports" / "model_tuning.json"
MODEL_DIR = PROJECT_ROOT / "models" / "production"


def file_sha256(path: Path) -> str:
    """Calculate dataset or artifact checksum."""
    sha = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)

    return sha.hexdigest()


def atomic_joblib_dump(model: Pipeline, path: Path) -> None:
    """Save the model without leaving partially written artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=".model-",
        suffix=".joblib",
        delete=False,
    ) as temp:
        temp_path = Path(temp.name)

    try:
        joblib.dump(model, temp_path, compress=3)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def package_model(
    data_path: Path = DATA_PATH,
    tuning_path: Path = TUNING_PATH,
    output_dir: Path = MODEL_DIR,
) -> dict:
    """Train and package the CV-selected model."""

    data_path = Path(data_path)
    tuning_path = Path(tuning_path)
    output_dir = Path(output_dir)

    report = json.loads(tuning_path.read_text(encoding="utf-8"))

    if report.get("selection_metric") != "cv_roc_auc":
        raise ValueError("Expected ROC-AUC model selection")

    if report.get("holdout_used_for_selection") is not False:
        raise ValueError("Holdout must not be used for model selection")

    selected = report["selected_model"]
    models = create_models()

    if selected not in models:
        raise ValueError(f"Unknown model selected: {selected}")

    tuning_result = report["models"][selected]
    best_params = tuning_result["best_params"]

    # Only permit parameter names belonging to the selected classifier.
    allowed_params = {
        f"classifier__{parameter}"
        for parameter in models[selected].get_params()
    }

    if not set(best_params).issubset(allowed_params):
        raise ValueError("Tuning report contains unsupported parameters")

    logger.info("Packaging selected model: %s", selected)

    # Use the same deterministic split as the tuning stage.
    X_train, X_test, y_train, y_test = load_dataset(data_path)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("classifier", models[selected]),
        ]
    )

    pipeline.set_params(**best_params)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.joblib"
    metadata_path = output_dir / "model_metadata.json"

    atomic_joblib_dump(pipeline, model_path)

    metadata = {
        "model_name": selected,
        "model_version": "1.0.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection_metric": "cv_roc_auc",
        "cv_roc_auc": float(tuning_result["best_cv_roc_auc"]),
        "best_params": best_params,
        "holdout_metrics_exploratory": metrics,
        "training_samples": len(X_train),
        "holdout_samples": len(X_test),
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "target": "target",
        "random_state": RANDOM_STATE,
        "dataset_sha256": file_sha256(data_path),
        "model_sha256": file_sha256(model_path),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "artifact": str(model_path),
    }

    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=output_dir,
        prefix=".metadata-",
        suffix=".json",
        encoding="utf-8",
        delete=False,
    ) as temp:
        metadata_temp = Path(temp.name)

    try:
        metadata_temp.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        os.replace(metadata_temp, metadata_path)
    finally:
        metadata_temp.unlink(missing_ok=True)

    logger.info("Model saved to %s", model_path)
    logger.info("Metadata saved to %s", metadata_path)

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA_PATH)
    parser.add_argument("--tuning", type=Path, default=TUNING_PATH)
    parser.add_argument("--output", type=Path, default=MODEL_DIR)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    metadata = package_model(
        data_path=args.input,
        tuning_path=args.tuning,
        output_dir=args.output,
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
