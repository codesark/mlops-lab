"""Mock deployment: load a registry model by stage tag, smoke-test it, record a manifest.

Stands in for a real serving rollout — the registry interaction, smoke test, and
audit trail are real; only the serving infrastructure is simulated. The manifest is
written locally (uploaded as a workflow artifact in CI) and the deployment is also
stamped onto the model version as tags, visible in Azure ML Studio.
"""

import getpass
import json
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
from mlflow import MlflowClient

from mlops_lab.data import sample_rows
from mlops_lab.registry import get_stage_version, metrics_from_tags


def mock_deploy(
    model_name: str,
    stage: str,
    env: str,
    out_dir: Path,
    actor: str | None = None,
    git_sha: str | None = None,
) -> dict:
    client = MlflowClient()
    version = get_stage_version(client, model_name, stage)
    if version is None:
        raise RuntimeError(f"no version of {model_name!r} is tagged stage={stage!r}")

    # Azure ML has no @alias URIs; load the concrete resolved version.
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{version.version}")
    dataset = version.tags.get("dataset")
    if dataset is None:
        raise RuntimeError(
            f"model version {model_name} v{version.version} has no 'dataset' tag; cannot smoke-test"
        )
    rows = sample_rows(dataset, n=5)
    predictions = np.asarray(model.predict(rows), dtype=float)
    if predictions.shape[0] != len(rows) or not np.all(np.isfinite(predictions)):
        raise RuntimeError(f"smoke test failed: bad predictions {predictions!r}")

    deployed_at = datetime.now(UTC).isoformat()
    actor = actor or getpass.getuser()
    manifest = {
        "model": model_name,
        "version": version.version,
        "run_id": version.run_id,
        "stage": stage,
        "environment": env,
        "metrics": metrics_from_tags(version.tags),
        "git_sha": git_sha or version.tags.get("git_sha"),
        "actor": actor,
        "deployed_at": deployed_at,
        "smoke_test": {"rows": len(rows), "predictions": predictions.tolist()},
    }

    # Audit trail in Studio: stamp the deployment onto the model version itself.
    client.set_model_version_tag(model_name, version.version, f"deployed.{env}.at", deployed_at)
    client.set_model_version_tag(model_name, version.version, f"deployed.{env}.by", actor)

    env_dir = out_dir / env
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "latest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
