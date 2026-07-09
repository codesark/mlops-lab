# mlops-lab

An end-to-end simulation of how a company ships ML models: data scientists explore
in notebooks, refactor into a tested package, and open PRs; CI enforces quality
gates; merges register models in the MLflow Model Registry; a manager's explicit
approval gates (mocked) production deployment.

```
 notebook            PR                merge to main            manual dispatch
┌──────────┐   ┌──────────────┐   ┌───────────────────┐   ┌─────────────────────────┐
│ explore  │ → │ CI: lint     │ → │ CD: full train    │ → │ Deploy: staging (auto)  │
│ prototype│   │ unit tests   │   │ register best     │   │ ──────────────────────  │
│ hand off │   │ train + gates│   │ candidate         │   │ production (requires    │
│ to YAML  │   │ vs champion  │   │ → @staging        │   │ reviewer approval)      │
└──────────┘   └──────────────┘   └───────────────────┘   │ → @production           │
                                                          └─────────────────────────┘
```

## The pieces

| Piece | Where | What it does |
|---|---|---|
| Model candidates | `configs/*.yaml` | One YAML per candidate: dataset, params, quality gates, champion policy |
| Training/eval code | `src/mlops_lab/` | Config-driven pipeline: data → features → train → metrics → gates |
| Unit tests | `tests/` | Fast (<10s), no real training — gate logic, metrics, config validation |
| CI | `.github/workflows/ci.yml` | On PRs: lint, tests, train each candidate, check gates, compare vs `@production` champion, post metrics to the PR |
| CD | `.github/workflows/cd.yml` | On merge: retrain, register best gate-passer per model, set `@staging` |
| Deploy | `.github/workflows/deploy.yml` | Dispatch: mock-deploy to staging, then **pause for manager approval**, promote to `@production`, mock-deploy |
| Registry state | `mlflow-registry` branch | sqlite DB + artifacts + deployment manifests; every mutation is a commit |

Two model families ship from this repo:

- **housing-price** (regression, California Housing): Ridge baseline vs XGBoost — gated on RMSE/R².
- **cancer-classifier** (classification, Breast Cancer): LogisticRegression vs XGBoost — gated on F1/ROC-AUC.

## How the registry works without a hosted MLflow

There is no tracking server running anywhere. Instead, the registry state
(`mlflow.db` + `mlruns/` artifacts + `deployments/` manifests) lives on the orphan
git branch **`mlflow-registry`**. Every job that needs MLflow:

1. checks that branch out into `./registry`,
2. starts an ephemeral `mlflow server --serve-artifacts` against it (proxied
   artifacts keep URIs portable between machines),
3. does its work, stops the server, commits and pushes the branch.

Writers are serialized by a shared Actions concurrency group. Every registry
mutation — registration, promotion, deployment — is an auditable git commit. A real
org would use a hosted tracking server + object storage; the branch is the lab
stand-in.

## Local development

```bash
brew install libomp        # macOS only, for xgboost
uv sync
make test                  # unit tests
make mlflow-up DEV=1       # throwaway dev MLflow at http://127.0.0.1:5000
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
make train-all             # train all four candidates
uv run mlops evaluate --config configs/housing_xgb.yaml   # check gates
```

Notebooks (`make notebook`): `01_eda_and_baseline.ipynb` is the messy exploratory
phase; `02_prototype_to_package.ipynb` shows the prototype→package handoff and
proves the package reproduces the ad-hoc experiment exactly.

Inspect the **shared** registry (read-only) with the full MLflow UI:

```bash
make registry-pull         # clones the mlflow-registry branch into ./registry
make mlflow-up             # serves it at http://127.0.0.1:5001
```

## The day-2 loop (what a data scientist does)

1. Edit a config (or add a new one) — e.g. bump `max_depth` in
   `configs/housing_xgb.yaml`.
2. Open a PR. CI posts a metrics table per candidate: absolute gates plus a
   comparison against the current `@production` champion. A model that regresses
   past tolerance fails the check and can't merge.
3. Merge. CD registers the best gate-passing candidate and points `@staging` at it.
4. Anyone runs the **Deploy** workflow (Actions → Deploy → Run workflow, pick the
   model). Staging deploys automatically; the production job waits until the
   environment reviewer (the "manager") approves. Approval promotes
   `@staging` → `@production` and writes a deployment manifest with the model
   version, metrics, actor, and git SHA to the registry branch.

The new `@production` version's metrics become the champion baseline every future
PR is compared against.

## Deliberate simplifications

- Deployments are **mocked**: "deploy" = load the model from the registry by alias,
  smoke-test predictions, write a manifest. The gating, registry promotion, and
  audit trail are real.
- CI trains on the full dataset in-job (minutes). Real pipelines would use data
  versioning, caching, and remote compute.
- The registry branch stands in for a hosted tracking server + object store.
