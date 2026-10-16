
"""Tests for EDA artifact generation."""

import json

import pandas as pd
import pytest

from src.data.eda import run_eda


@pytest.fixture
def sample_dataset(tmp_path):
    data = pd.DataFrame(
        {
            "age": [45, 55, 60, 48],
            "trestbps": [120, 130, 140, 125],
            "chol": [200, 240, 260, 220],
            "thalach": [170, 155, 140, 165],
            "oldpeak": [0.5, 1.0, 2.0, 0.8],
            "target": [0, 1, 1, 0],
        }
    )

    path = tmp_path / "sample.csv"
    data.to_csv(path, index=False)

    return path


def test_eda_generates_all_artifacts(tmp_path, sample_dataset):
    output_dir = tmp_path / "figures"

    summary = run_eda(sample_dataset, output_dir)

    expected_files = [
        "target_distribution.png",
        "feature_histograms.png",
        "correlation_heatmap.png",
        "missing_values.png",
        "age_vs_target.png",
        "eda_summary.json",
    ]

    for filename in expected_files:
        assert (output_dir / filename).is_file()

    assert summary["total_rows"] == 4
    assert summary["class_distribution"] == {"0": 2, "1": 2}

    saved = json.loads(
        (output_dir / "eda_summary.json").read_text()
    )
    assert saved["total_rows"] == 4


def test_eda_rejects_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_eda(tmp_path / "missing.csv", tmp_path / "output")


def test_eda_rejects_invalid_schema(tmp_path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"age": [40], "target": [1]}).to_csv(
        path, index=False
    )

    with pytest.raises(ValueError, match="Missing required"):
        run_eda(path, tmp_path / "output")
