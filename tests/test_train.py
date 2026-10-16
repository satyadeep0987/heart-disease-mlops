
"""Tests for baseline model training."""

import pandas as pd
import pytest

from src.models.train import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    create_models,
    create_preprocessor,
    load_dataset,
    train_models,
)


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


def test_preprocessor_creation():
    preprocessor = create_preprocessor()

    assert len(preprocessor.transformers) == 2


def test_two_baseline_models_exist():
    models = create_models()

    assert "logistic_regression" in models
    assert "random_forest" in models


def test_dataset_split(training_csv):
    X_train, X_test, y_train, y_test = load_dataset(training_csv)

    assert len(X_train) == 48
    assert len(X_test) == 12
    assert len(y_train) == 48
    assert len(y_test) == 12
    assert list(X_train.columns) == (
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    )


def test_model_training(training_csv):
    results = train_models(training_csv)

    assert len(results) == 2

    for metrics in results.values():
        assert 0 <= metrics["cv_roc_auc"] <= 1
        assert 0 <= metrics["test_roc_auc"] <= 1
