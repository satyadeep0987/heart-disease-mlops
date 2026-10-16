
"""Tests for MLflow experiment tracking."""

import json

import pytest

from src.models.tracking import calculate_sha256, log_selected_model


def test_sha256_checksum(tmp_path):
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"test-model-content")

    result = calculate_sha256(model_file)

    assert len(result) == 64


def test_tracking_rejects_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError):
        log_selected_model(
            model_dir=tmp_path / "missing",
            tuning_path=tmp_path / "missing.json",
            tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        )


def test_tracking_rejects_checksum_mismatch(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    (model_dir / "model.joblib").write_bytes(b"model")

    metadata = {
        "model_name": "logistic_regression",
        "model_sha256": "incorrect-hash",
    }

    (model_dir / "model_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    tuning_path = tmp_path / "tuning.json"
    tuning_path.write_text(
        json.dumps(
            {"selected_model": "logistic_regression"}
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checksum"):
        log_selected_model(
            model_dir=model_dir,
            tuning_path=tuning_path,
            tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        )
