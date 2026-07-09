"""Mock deployment: load a registry model by alias, smoke-test it, record a manifest.

Stands in for a real serving rollout — the registry interaction, smoke test, and
audit trail are real; only the serving infrastructure is simulated.
"""

import getpass
import json
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
from mlflow import MlflowClient

from mlops_lab.data import sample_rows
from mlops_lab.registry import metrics_from_tags


def mock_deploy(
    model_name: str,
    alias: str,
    env: str,
    registry_root: Path,
    actor: str | None = None,
    git_sha: str | None = None,
) -> dict:
    client = MlflowClient()
    version = client.get_model_version_by_alias(model_name, alias)

    model = mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")
    dataset = version.tags.get("dataset")
    if dataset is None:
        raise RuntimeError(
            f"model version {model_name} v{version.version} has no 'dataset' tag; cannot smoke-test"
        )
    rows = sample_rows(dataset, n=5)
    predictions = np.asarray(model.predict(rows), dtype=float)
    if predictions.shape[0] != len(rows) or not np.all(np.isfinite(predictions)):
        raise RuntimeError(f"smoke test failed: bad predictions {predictions!r}")

    manifest = {
        "model": model_name,
        "version": version.version,
        "run_id": version.run_id,
        "alias": alias,
        "environment": env,
        "metrics": metrics_from_tags(version.tags),
        "git_sha": git_sha or version.tags.get("git_sha"),
        "actor": actor or getpass.getuser(),
        "deployed_at": datetime.now(UTC).isoformat(),
        "smoke_test": {"rows": len(rows), "predictions": predictions.tolist()},
    }

    env_dir = registry_root / "deployments" / env
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "latest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    history = registry_root / "deployments" / "history.ndjson"
    with history.open("a") as f:
        f.write(json.dumps(manifest) + "\n")
    return manifest
