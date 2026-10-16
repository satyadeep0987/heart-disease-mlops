
"""Tests for model evaluation artifacts."""

import json

import pandas as pd

from src.models.evaluate import evaluate_models


def test_evaluation_generates_artifacts(tmp_path):
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

    input_path = tmp_path / "training.csv"
    output_dir = tmp_path / "evaluation"

    pd.DataFrame(rows).to_csv(input_path, index=False)

    results = evaluate_models(input_path, output_dir)

    assert len(results) == 2

    for model_name, metrics in results.items():
        assert 0 <= metrics["roc_auc"] <= 1
        assert 0 <= metrics["accuracy"] <= 1
        assert len(metrics["confusion_matrix"]) == 2

        assert (
            output_dir / f"{model_name}_confusion_matrix.png"
        ).exists()

        assert (
            output_dir / f"{model_name}_roc_curve.png"
        ).exists()

    report = json.loads(
        (output_dir / "evaluation_metrics.json").read_text(
            encoding="utf-8"
        )
    )

    assert set(report) == {
        "logistic_regression",
        "random_forest",
    }
