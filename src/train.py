"""
Train, evaluate, track, and register a machine-learning model.
"""

import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
)

from src.data import load_dataset


EXPERIMENT_NAME = "breast-cancer-training"


def train_model() -> None:
    """
    Train a Random Forest classifier and register it in MLflow.
    """

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
    ) = load_dataset()

    parameters = {
        "n_estimators": 200,
        "max_depth": 8,
        "random_state": 42,
    }

    model = RandomForestClassifier(
        **parameters
    )

    with mlflow.start_run() as run:
        print(
            f"Training run: "
            f"{run.info.run_id}"
        )

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        precision = precision_score(
            y_test,
            predictions,
        )

        recall = recall_score(
            y_test,
            predictions,
        )

        f1 = f1_score(
            y_test,
            predictions,
        )

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        mlflow.log_params(
            parameters
        )

        mlflow.log_metrics(
            metrics
        )

        mlflow.log_param(
            "feature_count",
            len(feature_names),
        )

        signature = infer_signature(
            X_train,
            model.predict(X_train),
        )

        input_example = X_train.head(3)

        model_info = (
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=MODEL_NAME,
            )
        )

        print()
        print("Training complete.")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print()
        print(
            f"Registered model: "
            f"{MODEL_NAME}"
        )
        print(
            f"Model URI: "
            f"{model_info.model_uri}"
        )


if __name__ == "__main__":
    train_model()
