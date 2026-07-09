# mlops-lab

An end-to-end simulation of how a company ships ML models: data scientists explore
in notebooks, refactor into a tested package, and open PRs; CI enforces quality
gates; merges register models in the **Azure ML Model Registry**; a manager's
explicit approval gates (mocked) production deployment.

```
 notebook            PR                merge to main            manual dispatch
┌──────────┐   ┌──────────────┐   ┌───────────────────┐   ┌─────────────────────────┐
│ explore  │ → │ CI: lint     │ → │ CD: full train    │ → │ Deploy: staging (auto)  │
│ prototype│   │ unit tests   │   │ register best     │   │ ──────────────────────  │
│ hand off │   │ train + gates│   │ candidate         │   │ production (requires    │
│ to YAML  │   │ vs champion  │   │ → stage=staging   │   │ reviewer approval)      │
└──────────┘   └──────────────┘   │  in Azure ML      │   │ → stage=production      │
                                  └───────────────────┘   └─────────────────────────┘
```

Tracking + registry live in an **Azure ML workspace** (`mlops-lab-ws`, Central
India). Browse everything in [Azure ML Studio](https://ml.azure.com): runs under
**Jobs**, the registry under **Models** (stage/metric/deployment tags on each
version). GitHub Actions authenticates via **OIDC federated identity** — no stored
secrets; see `scripts/azure_setup.sh` for the full one-time bootstrap.

## The pieces

| Piece | Where | What it does |
|---|---|---|
| Model candidates | `configs/*.yaml` | One YAML per candidate: dataset, params, quality gates, champion policy |
| Training/eval code | `src/mlops_lab/` | Config-driven pipeline: data → features → train → metrics → gates |
| Unit tests | `tests/` | Fast (<10s), no real training — gate logic, metrics, config validation |
| CI | `.github/workflows/ci.yml` | On PRs: lint, tests, train each candidate, check gates, compare vs production champion, post metrics to the PR |
| CD | `.github/workflows/cd.yml` | On merge: retrain, register best gate-passer per model, tag `stage=staging` |
| Deploy | `.github/workflows/deploy.yml` | Dispatch: mock-deploy to staging, then **pause for manager approval**, tag `stage=production`, mock-deploy |
| Azure bootstrap | `scripts/azure_setup.sh` | One-time: workspace, OIDC app + federated credentials, RBAC, repo variables |

Two model families ship from this repo:

- **housing-price** (regression, California Housing): Ridge baseline vs XGBoost — gated on RMSE/R².
- **cancer-classifier** (classification, Breast Cancer): LogisticRegression vs XGBoost — gated on F1/ROC-AUC.

## Azure ML compatibility notes (why the code looks this way)

- **MLflow is pinned to 2.22.5**: Azure ML's MLflow backend doesn't implement the
  MLflow 3.x logged-models endpoints (`mlflow<3` is Microsoft's guidance).
  `mlflow-skinny` is pinned to the same version — without that, uv resolves a 3.x
  skinny from `azureml-mlflow`'s loose dependency and clobbers the package.
- **No registry aliases on Azure ML** (the API 404s): promotion state lives in a
  single-valued `stage` model-version tag (`staging` / `production`). Azure ML's
  `search_model_versions` only supports `name = '...'` filters, so stage resolution
  filters tags client-side (`src/mlops_lab/registry.py`).
- **Auth is automatic**: after `az login` locally or `azure/login` (OIDC) in
  Actions, `azureml-mlflow` picks up the CLI credential — `MLFLOW_TRACKING_URI` is
  the only configuration.

## Local development

```bash
brew install libomp        # macOS only, for xgboost
uv sync
make test                  # unit tests
az login                   # once
export MLFLOW_TRACKING_URI=$(make -s azure-uri)
make train-all             # train all four candidates; runs appear in Studio → Jobs
uv run mlops evaluate --config configs/housing_xgb.yaml   # check gates
```

Offline? `make mlflow-up` starts a local throwaway MLflow at `http://127.0.0.1:5000`.

Notebooks (`make notebook`): `01_eda_and_baseline.ipynb` is the messy exploratory
phase; `02_prototype_to_package.ipynb` shows the prototype→package handoff and
proves the package reproduces the ad-hoc experiment exactly.

## The day-2 loop (what a data scientist does)

1. Edit a config (or add a new one) — e.g. bump `max_depth` in
   `configs/housing_xgb.yaml`.
2. Open a PR. CI posts a metrics table per candidate: absolute gates plus a
   comparison against the current production champion. A model that regresses past
   tolerance fails the check and can't merge.
3. Merge. CD registers the best gate-passing candidate and tags it `stage=staging`.
4. Anyone runs the **Deploy** workflow (Actions → Deploy → Run workflow, pick the
   model). Staging deploys automatically; the production job waits until the
   environment reviewer (the "manager") approves. Approval moves the `stage` tag to
   `production`, mock-deploys, stamps `deployed.production.at/by` tags on the
   version (visible in Studio), and uploads the deployment manifest as a workflow
   artifact.

The production version's metrics become the champion baseline every future PR is
compared against.

## Deliberate simplifications

- Deployments are **mocked**: "deploy" = load the model from the registry by stage,
  smoke-test predictions, record manifest + tags. Graduating for real means
  pointing the deploy step at an Azure ML managed online endpoint — the gating flow
  doesn't change.
- CI trains on the full dataset in-job (minutes). Real pipelines would use data
  versioning, caching, and remote compute (Azure ML jobs/pipelines).

## History

This repo previously ran without any hosted services: the registry lived on the
orphan `mlflow-registry` git branch with an ephemeral `mlflow server` in every job.
That branch is kept frozen as an archive of the pre-Azure era.
