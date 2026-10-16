
"""Tests for the deterministic data cleaning process."""

import pandas as pd
import pytest

from src.data.preprocessing import clean_dataframe, process_dataset


@pytest.fixture
def raw_data():
    return pd.DataFrame(
        {
            "age": [45, 60, 45],
            "sex": [1, 0, 1],
            "cp": [2, 4, 2],
            "trestbps": [120, 140, 120],
            "chol": [220, 250, 220],
            "fbs": [0, 1, 0],
            "restecg": [0, 1, 0],
            "thalach": [165, 130, 165],
            "exang": [0, 1, 0],
            "oldpeak": [1.2, 2.5, 1.2],
            "slope": [1, 2, 1],
            "ca": [0, None, 0],
            "thal": [3, 7, 3],
            "num": [0, 3, 0],
        }
    )


def test_binary_target_conversion(raw_data):
    cleaned = clean_dataframe(raw_data)

    assert cleaned["target"].tolist() == [0, 1]
    assert "num" not in cleaned.columns


def test_duplicate_rows_removed(raw_data):
    cleaned = clean_dataframe(raw_data)

    assert len(cleaned) == 2


def test_missing_features_preserved(raw_data):
    cleaned = clean_dataframe(raw_data)

    assert pd.isna(cleaned.loc[1, "ca"])


def test_raw_dataframe_unchanged(raw_data):
    original = raw_data.copy(deep=True)

    clean_dataframe(raw_data)

    pd.testing.assert_frame_equal(raw_data, original)


def test_process_dataset_creates_csv(tmp_path, raw_data):
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "processed" / "clean.csv"

    raw_data.to_csv(input_path, index=False)

    result = process_dataset(input_path, output_path)

    assert result == output_path
    assert output_path.exists()

    saved = pd.read_csv(output_path)
    assert saved["target"].tolist() == [0, 1]
