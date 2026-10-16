
"""Manage MLflow registered model aliases."""

import argparse
import os

import mlflow
from mlflow import MlflowClient

REGISTERED_MODEL_NAME = "heart-disease-classifier"


def set_candidate_alias(
    version: str,
    tracking_uri: str | None = None,
) -> None:
    """Point the candidate alias to a registered model version."""

    mlflow.set_tracking_uri(
        tracking_uri
        or os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )

    client = MlflowClient()

    # Verify the requested version exists before changing the alias.
    model_version = client.get_model_version(
        name=REGISTERED_MODEL_NAME,
        version=version,
    )

    if model_version.status != "READY":
        raise ValueError(
            f"Model version {version} is not ready"
        )

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="candidate",
        version=version,
    )

    print(
        f"Candidate alias assigned to "
        f"{REGISTERED_MODEL_NAME} version {version}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    args = parser.parse_args()

    set_candidate_alias(args.version)


if __name__ == "__main__":
    main()
