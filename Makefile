install:
	python3 -m pip install -r requirements.txt

mlflow:
	mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlartifacts \
		--host 127.0.0.1 \
		--port 5000

train:
	python3 -m src.train

validate:
	python3 -m src.validate

promote:
	python3 -m src.promote

serve:
	uvicorn src.api:app --host 127.0.0.1 --port 8000

predict:
	python3 scripts/test_endpoint.py

monitor:
	python3 -m src.monitor
