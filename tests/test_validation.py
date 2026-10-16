
"""Unit tests for UCI dataset validation."""

import pandas as pd
import pytest

from src.data.validation import (
    FEATURE_COLUMNS,
    validate_dataframe,
)


@pytest.fixture
def valid_data():
    """Minimal valid UCI-style dataset."""
    return pd.DataFrame(
        {
            "age": [45, 60],
            "sex": [1, 0],
            "cp": [2, 4],
            "trestbps": [120, 140],
            "chol": [220, 250],
            "fbs": [0, 1],
            "restecg": [0, 1],
            "thalach": [165, 130],
            "exang": [0, 1],
            "oldpeak": [1.2, 2.5],
            "slope": [1, 2],
            "ca": [0, None],
            "thal": [3, 7],
            "num": [0, 2],
        }
    )


def test_valid_data_passes(valid_data):
    result = validate_dataframe(valid_data)

    assert result["rows"] == 2
    assert result["columns"] == 14


def test_missing_column_fails(valid_data):
    data = valid_data.drop(columns=[FEATURE_COLUMNS[0]])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(data)


def test_invalid_target_fails(valid_data):
    valid_data.loc[0, "num"] = 9

    with pytest.raises(ValueError, match="Unexpected target labels"):
        validate_dataframe(valid_data)


def test_invalid_age_fails(valid_data):
    valid_data.loc[0, "age"] = -10

    with pytest.raises(ValueError, match="outside expected range"):
        validate_dataframe(valid_data)


def test_missing_feature_is_allowed(valid_data):
    valid_data.loc[0, "chol"] = None

    result = validate_dataframe(valid_data)

    assert result["missing_values"]["chol"] == 1


def test_invalid_text_fails(valid_data):
    valid_data["age"] = valid_data["age"].astype(object)
    valid_data.loc[0, "age"] = "invalid"

    with pytest.raises(ValueError, match="Non-numeric values"):
        validate_dataframe(valid_data)
