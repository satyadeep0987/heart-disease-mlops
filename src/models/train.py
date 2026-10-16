
"""Train baseline heart disease classification models."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart_clean.csv"

NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

TARGET = "target"
RANDOM_STATE = 42


def create_preprocessor() -> ColumnTransformer:
    """Create preprocessing fitted only on training data."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def create_models() -> dict:
    """Define baseline classification algorithms."""

    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def load_dataset(path: Path = DATA_PATH):
    """Load and split data without fitting preprocessing yet."""

    df = pd.read_csv(path)

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    required = set(features + [TARGET])

    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Missing columns: {missing}")

    if df.empty:
        raise ValueError("Training dataset is empty")

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values")

    X = df[features]
    y = df[TARGET]

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1")

    if y.nunique() != 2:
        raise ValueError("Both target classes are required")

    return train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def train_models(data_path: Path = DATA_PATH) -> dict:
    """Train and cross-validate baseline models."""

    X_train, X_test, y_train, y_test = load_dataset(data_path)

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = {}

    for name, classifier in create_models().items():
        logger.info("Training model: %s", name)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                ("classifier", classifier),
            ]
        )

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cross_validation,
            scoring={
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "roc_auc": "roc_auc",
            },
            n_jobs=1,
            error_score="raise",
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        results[name] = {
            "cv_accuracy": float(
                cv_results["test_accuracy"].mean()
            ),
            "cv_precision": float(
                cv_results["test_precision"].mean()
            ),
            "cv_recall": float(
                cv_results["test_recall"].mean()
            ),
            "cv_roc_auc": float(
                cv_results["test_roc_auc"].mean()
            ),
            "test_accuracy": float(
                accuracy_score(y_test, predictions)
            ),
            "test_precision": float(
                precision_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "test_recall": float(
                recall_score(y_test, predictions)
            ),
            "test_roc_auc": float(
                roc_auc_score(y_test, probabilities)
            ),
        }

        logger.info(
            "%s | CV ROC-AUC: %.4f",
            name,
            results[name]["cv_roc_auc"],
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA_PATH)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    results = train_models(args.input)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
