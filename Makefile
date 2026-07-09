.PHONY: install lint test train train-all mlflow-up mlflow-down registry-pull deploy-staging deploy-production notebook

CONFIG ?= configs/housing_xgb.yaml
CONFIGS := configs/housing_ridge.yaml configs/housing_xgb.yaml configs/cancer_logreg.yaml configs/cancer_xgb.yaml

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

# Dev server (throwaway store) on :5000; shared registry server on :5001.
# Usage: `make mlflow-up DEV=1` for dev, `make mlflow-up` for the registry checkout.
mlflow-up:
	./scripts/mlflow_server.sh start $(if $(DEV),dev,registry)

mlflow-down:
	./scripts/mlflow_server.sh stop

train:
	uv run mlops train --config $(CONFIG)

train-all:
	for c in $(CONFIGS); do uv run mlops train --config $$c || exit 1; done

# Fetch the shared registry branch into ./registry for local inspection (read-only)
registry-pull:
	./scripts/registry_checkout.sh

deploy-staging:
	uv run mlops deploy --model $(MODEL) --alias staging --env staging

deploy-production:
	uv run mlops promote --model $(MODEL) --from-alias staging --to-alias production
	uv run mlops deploy --model $(MODEL) --alias production --env production

notebook:
	uv run jupyter lab
