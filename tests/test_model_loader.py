
"""Tests for loading packaged models safely."""

import hashlib
import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from api.model_loader import (
    calculate_sha256,
    load_local_model,
    load_model,
    validate_pipeline,
)


@pytest.fixture
def model_artifacts(tmp_path):
    """Create a real, fitted sklearn model for loader tests."""

    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "model_metadata.json"

    X = np.array([
        [25, 120],
        [35, 130],
        [45, 140],
        [55, 150],
        [65, 160],
        [75, 170],
    ])

    y = np.array([0, 0, 0, 1, 1, 1])

    pipeline = Pipeline([
        ("classifier", LogisticRegression(max_iter=200)),
    ])

    pipeline.fit(X, y)

    joblib.dump(pipeline, model_path)

    checksum = hashlib.sha256(
        model_path.read_bytes()
    ).hexdigest()

    metadata_path.write_text(
        json.dumps({
            "model_name": "test-classifier",
            "model_version": "1.0.0",
            "model_sha256": checksum,
        }),
        encoding="utf-8",
    )

    return model_path, metadata_path


def test_checksum(tmp_path):
    path = tmp_path / "test.txt"
    path.write_bytes(b"hello")

    assert calculate_sha256(path) == hashlib.sha256(
        b"hello"
    ).hexdigest()



def test_load_local_model(model_artifacts):
    model_path, metadata_path = model_artifacts

    loaded = load_local_model(model_path, metadata_path)

    assert loaded.name == "test-classifier"
    assert loaded.version == "1.0.0"
    assert loaded.source == "local"

    sample = [[50, 145]]

    prediction = loaded.pipeline.predict(sample)
    probabilities = loaded.pipeline.predict_proba(sample)

    assert prediction[0] in (0, 1)
    assert probabilities.shape == (1, 2)
    assert probabilities.sum() == pytest.approx(1.0)



def test_reject_checksum_mismatch(model_artifacts):
    model_path, metadata_path = model_artifacts

    model_path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="checksum"):
        load_local_model(model_path, metadata_path)


def test_missing_model_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_local_model(
            tmp_path / "missing.joblib",
            tmp_path / "missing.json",
        )


def test_invalid_pipeline():
    with pytest.raises(TypeError):
        validate_pipeline(object())


def test_invalid_model_source(monkeypatch):
    monkeypatch.setenv("MODEL_SOURCE", "invalid")

    with pytest.raises(ValueError, match="MODEL_SOURCE"):
        load_model()
