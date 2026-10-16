
"""Tests for hyperparameter tuning and model selection."""

import json

import pandas as pd
import pytest

from src.models.tune import select_best_model, tune_models


@pytest.fixture
def training_csv(tmp_path):
    rows = []

    for i in range(60):
        rows.append(
            {
                "age": 35 + i % 30,
                "sex": i % 2,
                "cp": 1 + i % 4,
                "trestbps": 110 + i % 40,
                "chol": 180 + i % 80,
                "fbs": i % 2,
                "restecg": i % 3,
                "thalach": 130 + i % 50,
                "exang": i % 2,
                "oldpeak": (i % 10) / 5,
                "slope": 1 + i % 3,
                "ca": i % 4,
                "thal": [3, 6, 7][i % 3],
                "target": i % 2,
            }
        )

    path = tmp_path / "training.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_select_best_model():
    results = {
        "logistic_regression": {"best_cv_roc_auc": 0.90},
        "random_forest": {"best_cv_roc_auc": 0.87},
    }

    assert select_best_model(results) == "logistic_regression"


def test_empty_results_rejected():
    with pytest.raises(ValueError, match="No models"):
        select_best_model({})


def test_tuning_generates_report(tmp_path, training_csv, monkeypatch):
    # Use one configuration per model for a lightweight unit test.
    from src.models import tune

    monkeypatch.setattr(
        tune,
        "PARAMETER_GRIDS",
        {
            "logistic_regression": {
                "classifier__C": [1.0],
            },
            "random_forest": {
                "classifier__n_estimators": [10],
                "classifier__max_depth": [5],
            },
        },
    )

    output = tmp_path / "tuning.json"

    result = tune_models(
        data_path=training_csv,
        output_path=output,
        cv_splits=3,
    )

    assert output.exists()
    assert result["selected_model"] in {
        "logistic_regression",
        "random_forest",
    }
    assert result["holdout_used_for_selection"] is False

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["selection_metric"] == "cv_roc_auc"

    for metrics in saved["models"].values():
        assert 0 <= metrics["best_cv_roc_auc"] <= 1
