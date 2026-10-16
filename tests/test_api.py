
"""Integration tests for the FastAPI prediction service."""

from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from api.main import app
from api.model_loader import LoadedModel


def create_test_model():
    pipeline = MagicMock()
    pipeline.predict.return_value = np.array([1])
    pipeline.predict_proba.return_value = np.array([[0.2, 0.8]])

    return LoadedModel(
        pipeline=pipeline,
        name="test-classifier",
        version="1.0.0",
        source="test",
    )


VALID_PATIENT = {
    "age": 52,
    "sex": 1,
    "cp": 2,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 2,
    "thal": 3,
}



def test_health():
    with (
        patch("api.main.load_model", return_value=create_test_model()),
        TestClient(app) as client,
    ):
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ready():
    with (
        patch("api.main.load_model", return_value=create_test_model()),
        TestClient(app) as client,
    ):
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_prediction():
    with (
        patch("api.main.load_model", return_value=create_test_model()),
        TestClient(app) as client,
    ):
        response = client.post("/predict", json=VALID_PATIENT)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] == 1
    assert result["probability_disease"] == 0.8
    assert result["model_version"] == "1.0.0"


def test_invalid_input():
    invalid_patient = {**VALID_PATIENT, "age": 150}

    with (
        patch("api.main.load_model", return_value=create_test_model()),
        TestClient(app) as client,
    ):
        response = client.post("/predict", json=invalid_patient)

    assert response.status_code == 422
