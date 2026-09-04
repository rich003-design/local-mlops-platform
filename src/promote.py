"""
Promote an approved MLflow model version to the champion alias.
"""

import mlflow

from mlflow import MlflowClient

from src.config import (
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    MODEL_NAME,
)


def promote_latest_model() -> None:
    """
    Promote the newest validated model to the champion alias.
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
            "No registered models were found."
        )

    approved_versions = []

    for version in versions:
        model_version = (
            client.get_model_version(
                name=MODEL_NAME,
                version=version.version,
            )
        )

        status = (
            model_version.tags.get(
                "validation_status"
            )
        )

        if status == "approved":
            approved_versions.append(
                model_version
            )

    if not approved_versions:
        raise RuntimeError(
            "No approved model version found."
        )

    latest_approved = max(
        approved_versions,
        key=lambda version: int(
            version.version
        ),
    )

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
        version=latest_approved.version,
    )

    print(
        f"Model version "
        f"{latest_approved.version} "
        f"is now @{MODEL_ALIAS}."
    )


if __name__ == "__main__":
    promote_latest_model()
