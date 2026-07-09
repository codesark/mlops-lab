from unittest.mock import MagicMock, call

import pytest
from mlflow.exceptions import MlflowException

from mlops_lab.registry import (
    STAGE_TAG,
    get_champion_metrics,
    get_stage_version,
    metrics_from_tags,
    promote,
)


def make_version(tags: dict[str, str], version: str = "3"):
    mv = MagicMock()
    mv.tags = tags
    mv.version = version
    return mv


def client_with_versions(*versions):
    client = MagicMock()
    client.search_model_versions.return_value = list(versions)
    return client


def test_metrics_from_tags_parses_prefixed_floats():
    tags = {"metric.rmse": "0.53", "metric.r2": "0.81", "dataset": "california_housing"}
    assert metrics_from_tags(tags) == {"rmse": 0.53, "r2": 0.81}


def test_stage_version_resolves_by_tag():
    client = client_with_versions(
        make_version({STAGE_TAG: "production", "metric.rmse": "0.5"}, version="2"),
        make_version({STAGE_TAG: "staging"}, version="3"),
    )
    found = get_stage_version(client, "housing-price", "production")
    assert found.version == "2"
    client.search_model_versions.assert_called_once_with("name = 'housing-price'")


def test_stage_version_highest_wins_on_duplicate_tags():
    client = client_with_versions(
        make_version({STAGE_TAG: "production"}, version="2"),
        make_version({STAGE_TAG: "production"}, version="10"),
    )
    assert get_stage_version(client, "m", "production").version == "10"


def test_stage_version_none_when_untagged():
    client = client_with_versions(make_version({}, version="1"))
    assert get_stage_version(client, "m", "production") is None


def test_stage_version_none_when_model_missing():
    client = MagicMock()
    client.search_model_versions.side_effect = MlflowException("model not found")
    assert get_stage_version(client, "m", "production") is None


def test_champion_metrics_from_production_tags():
    client = client_with_versions(
        make_version({STAGE_TAG: "production", "metric.rmse": "0.5"}, version="1")
    )
    assert get_champion_metrics(client, "housing-price") == {"rmse": 0.5}


def test_champion_metrics_none_without_production():
    client = client_with_versions(make_version({STAGE_TAG: "staging"}, version="1"))
    assert get_champion_metrics(client, "housing-price") is None


def test_promote_moves_tag_off_previous_holder():
    staging = make_version({STAGE_TAG: "staging"}, version="7")
    old_prod = make_version({STAGE_TAG: "production"}, version="4")
    client = client_with_versions(staging, old_prod)

    promoted = promote(client, "housing-price", "staging", "production")

    assert promoted.version == "7"
    client.delete_model_version_tag.assert_called_once_with("housing-price", "4", STAGE_TAG)
    assert (
        call("housing-price", "7", STAGE_TAG, "production")
        in client.set_model_version_tag.call_args_list
    )


def test_promote_without_staging_raises():
    client = client_with_versions(make_version({}, version="1"))
    with pytest.raises(ValueError, match="staging"):
        promote(client, "housing-price", "staging", "production")
