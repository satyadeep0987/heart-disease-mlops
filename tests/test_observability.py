
"""Tests for API observability."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app
from tests.test_api import create_test_model


def test_metrics_endpoint():
    with (
        patch("api.main.load_model", return_value=create_test_model()),
        TestClient(app) as client,
    ):
        client.get("/health")
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "heart_api_requests_total" in response.text
    assert "heart_api_request_duration_seconds" in response.text


def test_request_id_header():
    with (
        patch("api.main.load_model", return_value=create_test_model()),
        TestClient(app) as client,
    ):
        response = client.get("/health")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"]
