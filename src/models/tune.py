
"""Hyperparameter tuning and cross-validation-based model selection."""

import argparse
import json
import logging
from pathlib import Path

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


def select_best_model(results: dict) -> str:
    """Select the model with the highest cross-validation ROC-AUC."""
    if not results:
        raise ValueError("No models available for selection")

    return max(
        results,
        key=lambda name: results[name]["best_cv_roc_auc"],
    )


def tune_models(
    data_path: Path = DATA_PATH,
    output_path: Path = OUTPUT_PATH,
    cv_splits: int = 5,
) -> dict:
    """Tune two classifiers without using holdout data for selection."""

    X_train, _, y_train, _ = load_dataset(data_path)

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = {}

    for name, classifier in create_models().items():
        logger.info("Starting hyperparameter search: %s", name)

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
        "holdout_used_for_selection": False,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    logger.info("Selected model: %s", selected_model)
    logger.info("Tuning report saved: %s", output_path)

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    report = tune_models(
        data_path=args.input,
        output_path=args.output,
        cv_splits=args.cv,
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
