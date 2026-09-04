# Local MLOps Flow

## Objective

This project demonstrates an end-to-end MLOps lifecycle without using a
managed cloud provider.

## Architecture

```text
Dataset
 ↓
Training
 ↓
Experiment Tracking
 ↓
Evaluation
 ↓
Model Registry
 ↓
Validation
 ↓
Promotion
 ↓
Model Serving
 ↓
Monitoring
```

## Training

The training module loads a scikit-learn dataset and trains a Random Forest
classifier. Parameters and metrics are recorded in MLflow.

## Experiment Tracking

MLflow tracks:

- Parameters
- Metrics
- Runs
- Model artifacts
- Input examples
- Model signatures

## Model Registry

Each successful training run creates a new model version.

Example: `breast-cancer-classifier`

- Version 1
- Version 2
- Version 3

## Validation

The model must meet the configured minimum accuracy before deployment.

Example: `Accuracy >= 0.94`

Approved models receive: `validation_status=approved`

## Promotion

The latest approved model receives the alias `champion`. This means
consumers can load:

```text
models:/breast-cancer-classifier@champion
```

without knowing the exact model version.

## Serving

FastAPI loads the champion model from the MLflow Model Registry and exposes:

- `GET /health`
- `POST /predict`

This simulates a managed inference endpoint.

## Monitoring

Every prediction is written to a JSONL log.

Future improvements can calculate:

- Prediction distribution
- Feature drift
- Data drift
- Accuracy drift
- Latency
- Error rate
