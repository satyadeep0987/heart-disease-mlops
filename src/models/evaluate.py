
"""Generate evaluation reports and charts for baseline ML models."""

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config import PROJECT_ROOT
from src.models.train import (
    DATA_PATH,
    create_models,
    create_preprocessor,
    load_dataset,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "model_evaluation"


def evaluate_models(
    data_path: Path = DATA_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Fit baselines and generate exploratory holdout evaluation artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_dataset(data_path)

    results = {}

    for name, classifier in create_models().items():
        logger.info("Evaluating model: %s", name)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                ("classifier", classifier),
            ]
        )

        # Fit preprocessing and classifier on training data only.
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=[0, 1]
            ).tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                output_dict=True,
                zero_division=0,
            ),
        }

        # Confusion matrix visualization.
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            labels=[0, 1],
            display_labels=["No Disease", "Disease"],
            cmap="Blues",
            colorbar=False,
        )
        plt.title(f"{name} — Confusion Matrix")
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{name}_confusion_matrix.png",
            dpi=150,
        )
        plt.close()

        # ROC curve visualization.
        RocCurveDisplay.from_predictions(
            y_test,
            probabilities,
            name=name,
        )
        plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
        plt.title(f"{name} — ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{name}_roc_curve.png",
            dpi=150,
        )
        plt.close()

        results[name] = metrics

        logger.info(
            "%s | Holdout ROC-AUC: %.4f",
            name,
            metrics["roc_auc"],
        )

    report_path = output_dir / "evaluation_metrics.json"
    report_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    logger.info("Evaluation artifacts saved to %s", output_dir)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    results = evaluate_models(args.input, args.output)

    for name, metrics in results.items():
        print(
            f"{name}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}"
        )


if __name__ == "__main__":
    main()
