
"""Hyperparameter tuning with optional per-candidate MLflow tracking."""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import mlflow
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import PROJECT_ROOT
from src.models.train import (
    DATA_PATH,
    RANDOM_STATE,
    create_models,
    create_preprocessor,
    load_dataset,
)

logger = logging.getLogger(__name__)

OUTPUT_PATH = PROJECT_ROOT / "reports" / "model_tuning.json"
EXPERIMENT_NAME = "heart-disease-classification"

PARAMETER_GRIDS = {
    "logistic_regression": {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__class_weight": [None, "balanced"],
    },
    "random_forest": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [5, 10],
        "classifier__min_samples_leaf": [1, 3],
    },
}


def calculate_sha256(path: Path) -> str:
    """Compute dataset checksum for experiment traceability."""
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def select_best_model(results: dict) -> str:
    """Choose model using training cross-validation ROC-AUC."""
    if not results:
        raise ValueError("No models available for selection")

    return max(
        results,
        key=lambda name: results[name]["best_cv_roc_auc"],
    )


def log_cv_candidates(
    model_name: str,
    search: GridSearchCV,
    dataset_hash: str,
    cv_splits: int,
) -> list[str]:
    """Log every evaluated candidate without fitting again."""
    run_ids = []
    cv_results = search.cv_results_

    for index, params in enumerate(cv_results["params"]):
        rank = int(cv_results["rank_test_roc_auc"][index])

        with mlflow.start_run(
            run_name=f"{model_name}-candidate-{index + 1}"
        ) as run:
            mlflow.set_tags(
                {
                    "project": "heart-disease-mlops",
                    "stage": "hyperparameter_tuning",
                    "algorithm": model_name,
                    "cloud_target": "gcp",
                    "dataset_sha256": dataset_hash,
                    "candidate_index": str(index),
                    "is_best_for_algorithm": str(
                        index == search.best_index_
                    ).lower(),
                }
            )

            mlflow.log_params(params)
            mlflow.log_param("cv_folds", cv_splits)

            for metric in (
                "roc_auc",
                "accuracy",
                "precision",
                "recall",
            ):
                mean = float(
                    cv_results[f"mean_test_{metric}"][index]
                )
                std = float(
                    cv_results[f"std_test_{metric}"][index]
                )

                mlflow.log_metric(f"cv_{metric}_mean", mean)
                mlflow.log_metric(f"cv_{metric}_std", std)

            mlflow.log_metric("cv_roc_auc_rank", rank)

            run_ids.append(run.info.run_id)

    return run_ids


def tune_models(
    data_path: Path = DATA_PATH,
    output_path: Path = OUTPUT_PATH,
    cv_splits: int = 5,
    track_mlflow: bool = False,
    tracking_uri: str | None = None,
) -> dict:
    """Tune models using training data and optionally track candidates."""

    data_path = Path(data_path)
    output_path = Path(output_path)

    X_train, _, y_train, _ = load_dataset(data_path)

    dataset_hash = calculate_sha256(data_path)

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    if track_mlflow:
        mlflow.set_tracking_uri(
            tracking_uri
            or os.getenv(
                "MLFLOW_TRACKING_URI",
                "http://127.0.0.1:5000",
            )
        )
        mlflow.set_experiment(EXPERIMENT_NAME)

    results = {}

    for name, classifier in create_models().items():
        logger.info("Starting GridSearchCV: %s", name)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                ("classifier", classifier),
            ]
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=PARAMETER_GRIDS[name],
            scoring={
                "roc_auc": "roc_auc",
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
            },
            refit="roc_auc",
            cv=cv,
            n_jobs=1,
            pre_dispatch=1,
            error_score="raise",
            return_train_score=False,
        )

        search.fit(X_train, y_train)

        best_index = search.best_index_

        results[name] = {
            "best_params": search.best_params_,
            "best_cv_roc_auc": float(search.best_score_),
            "best_cv_accuracy": float(
                search.cv_results_["mean_test_accuracy"][best_index]
            ),
            "best_cv_precision": float(
                search.cv_results_["mean_test_precision"][best_index]
            ),
            "best_cv_recall": float(
                search.cv_results_["mean_test_recall"][best_index]
            ),
            "total_candidates": len(search.cv_results_["params"]),
            "cv_folds": cv_splits,
        }

        if track_mlflow:
            run_ids = log_cv_candidates(
                model_name=name,
                search=search,
                dataset_hash=dataset_hash,
                cv_splits=cv_splits,
            )

            results[name]["mlflow_run_ids"] = run_ids

        logger.info(
            "%s | Best CV ROC-AUC: %.4f",
            name,
            search.best_score_,
        )

    selected_model = select_best_model(results)

    report = {
        "selection_metric": "cv_roc_auc",
        "selected_model": selected_model,
        "models": results,
        "dataset_sha256": dataset_hash,
        "holdout_used_for_selection": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    logger.info("Selected model: %s", selected_model)

    return report


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument(
        "--track-mlflow",
        action="store_true",
        help="Log all GridSearchCV candidates to MLflow",
    )
    parser.add_argument("--tracking-uri", default=None)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    report = tune_models(
        data_path=args.input,
        output_path=args.output,
        cv_splits=args.cv,
        track_mlflow=args.track_mlflow,
        tracking_uri=args.tracking_uri,
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
