"""
Validate the newest registered model before promotion.
"""

import mlflow

from mlflow import MlflowClient

from src.config import (
    ACCURACY_THRESHOLD,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
)


def validate_latest_model() -> str:
    """
    Validate the latest model using its MLflow run metrics.
    """

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    client = MlflowClient()

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    if not versions:
        raise RuntimeError(
            f"No versions found for {MODEL_NAME}."
        )

    latest_version = max(
        versions,
        key=lambda version: int(
            version.version
        ),
    )

    run = client.get_run(
        latest_version.run_id
    )

    accuracy = run.data.metrics.get(
        "accuracy"
    )

    if accuracy is None:
        raise RuntimeError(
            "Accuracy metric was not found."
        )

    print(
        f"Validating model version "
        f"{latest_version.version}"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Threshold: "
        f"{ACCURACY_THRESHOLD:.4f}"
    )

    if accuracy < ACCURACY_THRESHOLD:
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=latest_version.version,
            key="validation_status",
            value="failed",
        )

        raise RuntimeError(
            "Model validation failed."
        )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=latest_version.version,
        key="validation_status",
        value="approved",
    )

    print(
        "Model validation passed."
    )

    return latest_version.version


if __name__ == "__main__":
    validate_latest_model()
