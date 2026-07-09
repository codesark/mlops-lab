"""MLflow Model Registry helpers: aliases, champion lookup, metric tags.

Champion metrics live in model-version tags (``metric.<name>``) so that PR-time
comparison only needs the registry DB — no artifact loading.
"""

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

METRIC_TAG_PREFIX = "metric."

STAGING = "staging"
PRODUCTION = "production"


def get_version_by_alias(client: MlflowClient, model_name: str, alias: str) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None


def metrics_from_tags(tags: dict[str, str]) -> dict[str, float]:
    return {
        key[len(METRIC_TAG_PREFIX) :]: float(value)
        for key, value in tags.items()
        if key.startswith(METRIC_TAG_PREFIX)
    }


def get_champion_metrics(
    client: MlflowClient, model_name: str, alias: str = PRODUCTION
) -> dict[str, float] | None:
    """Metrics of the current champion, or None if no champion exists yet."""
    version = get_version_by_alias(client, model_name, alias)
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
    alias: str = STAGING,
) -> ModelVersion:
    """Register a trained model as a new version, tag it, and point @staging at it."""
    version = mlflow.register_model(model_uri, model_name)
    for key, value in metrics.items():
        client.set_model_version_tag(
            model_name, version.version, f"{METRIC_TAG_PREFIX}{key}", repr(value)
        )
    client.set_model_version_tag(model_name, version.version, "dataset", dataset)
    if git_sha:
        client.set_model_version_tag(model_name, version.version, "git_sha", git_sha)
    client.set_registered_model_alias(model_name, alias, version.version)
    return version


def promote(client: MlflowClient, model_name: str, from_alias: str, to_alias: str) -> ModelVersion:
    """Reassign an alias, e.g. point @production at the current @staging version."""
    version = client.get_model_version_by_alias(model_name, from_alias)
    client.set_registered_model_alias(model_name, to_alias, version.version)
    return version
