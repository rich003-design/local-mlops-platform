# Local MLOps Platform

A cloud-neutral MLOps project demonstrating the full model lifecycle —
training, experiment tracking, model registry, validation, promotion,
serving, and monitoring — running entirely on your local machine.

## Stack

- Python 3.11
- scikit-learn
- MLflow
- FastAPI
- SQLite
- Docker

## Workflow

```text
Train → Track → Register → Validate → Promote → Serve → Monitor
```

The model is a Random Forest classifier trained on the scikit-learn
breast cancer dataset. See [docs/MLOPS_FLOW.md](docs/MLOPS_FLOW.md) for a
detailed walkthrough of each stage.

## Getting Started

### 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
make install
```

### 2. Configure environment

```bash
cp .env.example .env
```

### 3. Start the MLflow tracking server

```bash
make mlflow
```

Open the UI at [http://127.0.0.1:5000](http://127.0.0.1:5000).

### 4. Train a model

```bash
make train
```

### 5. Validate the latest run

```bash
make validate
```

Checks the new model against the minimum accuracy threshold
(`MODEL_ACCURACY_THRESHOLD` in `.env`) and marks it `approved` if it passes.

### 6. Promote the approved model

```bash
make promote
```

Assigns the `champion` alias to the latest approved model version, so the
API always serves the best validated model.

### 7. Serve the model

```bash
make serve
```

Open the API docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 8. Send a test prediction

```bash
make predict
```

### 9. Monitor predictions

```bash
make monitor
```

Summarizes the prediction log written to `monitoring/predictions.jsonl`.

## API

| Method | Endpoint   | Description                              |
|--------|------------|-------------------------------------------|
| GET    | `/health`  | Health check and current model URI        |
| POST   | `/predict` | Run inference against the champion model  |

## Project Layout

```text
src/          Core modules: config, data, train, validate, promote, api, monitor
scripts/      Helper scripts (e.g. test_endpoint.py)
tests/        Automated tests (pytest)
docs/         Architecture and workflow documentation
Dockerfile    Container image for the FastAPI serving layer
Makefile      Shortcuts for every workflow stage
```

## Testing

```bash
pytest
```

## Docker

```bash
docker build -t local-mlops-platform .
docker run -p 8000:8000 local-mlops-platform
```

> Note: the container serves the API only. Point `MLFLOW_TRACKING_URI` in
> your `.env` at a reachable MLflow server before building the image.
