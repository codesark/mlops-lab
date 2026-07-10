"""MLflow Model Registry helpers against Azure ML: stage tags, champion lookup, metric tags.

Azure ML's MLflow backend does not support registry aliases, and its
search_model_versions only accepts exact `name = '...'` filters (no tag filters,
no LIKE). Promotion state therefore lives in a single-valued `stage` model-version
tag, resolved client-side. Champion metrics are duplicated into model-version tags
(``metric.<name>``) so PR-time comparison needs no artifact loading.
"""

from datetime import UTC, datetime

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

METRIC_TAG_PREFIX = "metric."
STAGE_TAG = "stage"

STAGING = "staging"
PRODUCTION = "production"


def get_stage_version(client: MlflowClient, model_name: str, stage: str) -> ModelVersion | None:
    """The version currently tagged with this stage, or None. Highest version wins ties."""
    try:
        versions = client.search_model_versions(f"name = '{model_name}'")
    except Exception:
        # Azure ML raises if the registered model doesn't exist yet.
        return None
    tagged = [v for v in versions if v.tags.get(STAGE_TAG) == stage]
    if not tagged:
        return None
    return max(tagged, key=lambda v: int(v.version))


def _set_stage(client: MlflowClient, model_name: str, version: str, stage: str) -> None:
    """Point a stage tag at a version, clearing it from any previous holder."""
    previous = get_stage_version(client, model_name, stage)
    if previous is not None and previous.version != version:
        client.delete_model_version_tag(model_name, previous.version, STAGE_TAG)
    client.set_model_version_tag(model_name, version, STAGE_TAG, stage)


def metrics_from_tags(tags: dict[str, str]) -> dict[str, float]:
    return {
        key[len(METRIC_TAG_PREFIX) :]: float(value)
        for key, value in tags.items()
        if key.startswith(METRIC_TAG_PREFIX)
    }


def get_champion_metrics(
    client: MlflowClient, model_name: str, stage: str = PRODUCTION
) -> dict[str, float] | None:
    """Metrics of the current champion, or None if no champion exists yet."""
    version = get_stage_version(client, model_name, stage)
    if version is None:
        return None
    return metrics_from_tags(version.tags)


def register_and_stage(
    client: MlflowClient,
    model_uri: str,
    model_name: str,
    metrics: dict[str, float],
    dataset: str,
    git_sha: str | None = None,
    stage: str = STAGING,
) -> ModelVersion:
    """Register a trained model as a new version, tag it, and move the stage tag to it."""
    version = mlflow.register_model(model_uri, model_name)
    for key, value in metrics.items():
        client.set_model_version_tag(
            model_name, version.version, f"{METRIC_TAG_PREFIX}{key}", repr(value)
        )
    client.set_model_version_tag(model_name, version.version, "dataset", dataset)
    if git_sha:
        client.set_model_version_tag(model_name, version.version, "git_sha", git_sha)
    _set_stage(client, model_name, version.version, stage)
    return version


def promote(client: MlflowClient, model_name: str, from_stage: str, to_stage: str) -> ModelVersion:
    """Move the to_stage tag onto the version currently holding from_stage."""
    version = get_stage_version(client, model_name, from_stage)
    if version is None:
        raise ValueError(f"no version of {model_name!r} is tagged {STAGE_TAG}={from_stage!r}")
    _set_stage(client, model_name, version.version, to_stage)
    return version


def rollback(
    client: MlflowClient,
    model_name: str,
    to_version: str,
    actor: str | None = None,
    reason: str | None = None,
) -> ModelVersion:
    """Move the production stage tag back to a specific earlier version.

    Versions are immutable, so rollback is a pointer move — no retraining. The
    rollback is stamped onto the target version as tags (who, when, from what,
    why) so the registry itself carries the audit trail.
    """
    target = client.get_model_version(model_name, to_version)  # raises if missing
    current = get_stage_version(client, model_name, PRODUCTION)
    if current is not None and current.version == target.version:
        raise ValueError(f"{model_name} v{to_version} is already production")

    _set_stage(client, model_name, target.version, PRODUCTION)

    rolled_at = datetime.now(UTC).isoformat()
    client.set_model_version_tag(model_name, target.version, "rollback.at", rolled_at)
    client.set_model_version_tag(
        model_name, target.version, "rollback.from", current.version if current else "none"
    )
    if actor:
        client.set_model_version_tag(model_name, target.version, "rollback.by", actor)
    if reason:
        client.set_model_version_tag(model_name, target.version, "rollback.reason", reason)
    return target
