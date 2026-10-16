
"""Tests for FastAPI inference schemas."""

import pandas as pd
import pytest
from pydantic import ValidationError

from api.schemas import PatientFeatures, PredictionResponse


@pytest.fixture
def valid_patient():
    return {
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


def test_valid_patient(valid_patient):
    patient = PatientFeatures(**valid_patient)
    assert patient.age == 52


def test_dataframe_conversion(valid_patient):
    patient = PatientFeatures(**valid_patient)
    frame = patient.to_dataframe()

    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (1, 13)
    assert frame["age"].iloc[0] == 52


def test_invalid_age_rejected(valid_patient):
    valid_patient["age"] = 150

    with pytest.raises(ValidationError):
        PatientFeatures(**valid_patient)


def test_unexpected_feature_rejected(valid_patient):
    valid_patient["patient_name"] = "example"

    with pytest.raises(ValidationError):
        PatientFeatures(**valid_patient)


def test_missing_feature_allowed(valid_patient):
    valid_patient["chol"] = None

    patient = PatientFeatures(**valid_patient)
    frame = patient.to_dataframe()

    assert pd.isna(frame["chol"].iloc[0])


def test_invalid_thal_rejected(valid_patient):
    valid_patient["thal"] = 5

    with pytest.raises(ValidationError):
        PatientFeatures(**valid_patient)


def test_invalid_numeric_value_rejected(valid_patient):
    valid_patient["oldpeak"] = float("nan")

    with pytest.raises(ValidationError):
        PatientFeatures(**valid_patient)


def test_prediction_response():
    response = PredictionResponse(
        prediction=1,
        probability_no_disease=0.2,
        probability_disease=0.8,
        model_version="1",
        model_name="heart-disease-classifier",
    )

    assert response.probability_disease == 0.8
