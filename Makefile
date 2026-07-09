.PHONY: install lint test train train-all mlflow-up mlflow-down azure-uri deploy-staging deploy-production notebook

CONFIG ?= configs/housing_xgb.yaml
CONFIGS := configs/housing_ridge.yaml configs/housing_xgb.yaml configs/cancer_logreg.yaml configs/cancer_xgb.yaml

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

# Local throwaway MLflow for offline play; the real registry is Azure ML.
mlflow-up:
	./scripts/mlflow_server.sh start

mlflow-down:
	./scripts/mlflow_server.sh stop

# Print the Azure ML workspace tracking URI (export it before train/deploy):
#   export MLFLOW_TRACKING_URI=$$(make -s azure-uri)
azure-uri:
	@az ml workspace show -n mlops-lab-ws -g mlops-lab-rg --query mlflow_tracking_uri -o tsv

train:
	uv run mlops train --config $(CONFIG)

train-all:
	for c in $(CONFIGS); do uv run mlops train --config $$c || exit 1; done

deploy-staging:
	uv run mlops deploy --model $(MODEL) --stage staging --env staging

deploy-production:
	uv run mlops promote --model $(MODEL)
	uv run mlops deploy --model $(MODEL) --stage production --env production

notebook:
	uv run jupyter lab
