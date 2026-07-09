# mlflow-registry branch

Shared MLflow tracking + model registry state for this repo's MLOps lab:
sqlite backend store (`mlflow.db`), proxied artifacts (`mlruns/`), and
deployment manifests (`deployments/`). Written only by CI workflows,
serialized via a shared Actions concurrency group. A real org would use a
hosted tracking server + object storage; this branch is the lab stand-in.

Inspect locally:  make registry-pull && make mlflow-up
