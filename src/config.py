"""
Central configuration for the local MLOps platform.
"""

import os

from dotenv import load_dotenv


load_dotenv()


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MODEL_NAME = os.getenv(
    "MLFLOW_MODEL_NAME",
    "breast-cancer-classifier",
)

MODEL_ALIAS = os.getenv(
    "MLFLOW_MODEL_ALIAS",
    "champion",
)

ACCURACY_THRESHOLD = float(
    os.getenv(
        "MODEL_ACCURACY_THRESHOLD",
        "0.94",
    )
)

API_HOST = os.getenv(
    "API_HOST",
    "127.0.0.1",
)

API_PORT = int(
    os.getenv(
        "API_PORT",
        "8000",
    )
)

MONITORING_LOG = os.getenv(
    "MONITORING_LOG",
    "monitoring/predictions.jsonl",
)
