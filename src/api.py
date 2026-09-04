"""
FastAPI inference server for the champion MLflow model.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import (
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    MODEL_NAME,
    MONITORING_LOG,
)


mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)


app = FastAPI(
    title="Local MLOps Prediction API",
    version="1.0.0",
    description=(
        "Local production-style model endpoint "
        "backed by MLflow Model Registry."
    ),
)


MODEL_URI = (
    f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
)


model = None


class PredictionRequest(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float

    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float

    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float


def load_model():
    """
    Load champion model from MLflow Model Registry.
    """

    print(
        f"Loading model: {MODEL_URI}"
    )

    return mlflow.pyfunc.load_model(
        MODEL_URI
    )


@app.on_event("startup")
def startup_event():
    """
    Load the model once when the API starts.
    """

    global model

    model = load_model()


@app.get("/health")
def health():
    """
    Health endpoint.
    """

    return {
        "status": "healthy",
        "model": MODEL_URI,
    }


@app.post("/predict")
def predict(
    request: PredictionRequest,
):
    """
    Generate a model prediction.
    """

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    payload = request.model_dump()

    dataframe = pd.DataFrame(
        [payload]
    )

    dataframe.columns = [
        column.replace(
            "_",
            " ",
        )
        for column in dataframe.columns
    ]

    try:
        prediction = model.predict(
            dataframe
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    predicted_class = int(
        prediction[0]
    )

    label = (
        "benign"
        if predicted_class == 1
        else "malignant"
    )

    response = {
        "prediction": predicted_class,
        "label": label,
        "model": MODEL_URI,
    }

    log_prediction(
        request_data=payload,
        response_data=response,
    )

    return response


def log_prediction(
    request_data: dict,
    response_data: dict,
) -> None:
    """
    Append prediction information to a local monitoring log.
    """

    log_path = Path(
        MONITORING_LOG
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    entry = {
        "timestamp": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "request": request_data,
        "response": response_data,
    }

    with log_path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(entry)
            + "\n"
        )
