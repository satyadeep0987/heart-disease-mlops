
"""Tests for model packaging and inference reproducibility."""

import json

import joblib
import pandas as pd

from src.models.package import file_sha256, package_model
from src.models.train import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def test_package_model(tmp_path):
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

    dataset = pd.DataFrame(rows)
    data_path = tmp_path / "heart.csv"
    dataset.to_csv(data_path, index=False)

    tuning_path = tmp_path / "tuning.json"
    tuning_path.write_text(
        json.dumps(
            {
                "selection_metric": "cv_roc_auc",
                "holdout_used_for_selection": False,
                "selected_model": "logistic_regression",
                "models": {
                    "logistic_regression": {
                        "best_params": {
                            "classifier__C": 1.0,
                        },
                        "best_cv_roc_auc": 0.90,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "models"

    metadata = package_model(
        data_path=data_path,
        tuning_path=tuning_path,
        output_dir=output_dir,
    )

    model_path = output_dir / "model.joblib"
    metadata_path = output_dir / "model_metadata.json"

    assert model_path.exists()
    assert metadata_path.exists()

    assert metadata["model_name"] == "logistic_regression"
    assert metadata["model_sha256"] == file_sha256(model_path)

    # Load only the trusted artifact produced by this test.
    model = joblib.load(model_path)

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    sample = dataset.loc[[0], features]

    probabilities = model.predict_proba(sample)

    assert probabilities.shape == (1, 2)
    assert 0 <= probabilities[0, 1] <= 1
