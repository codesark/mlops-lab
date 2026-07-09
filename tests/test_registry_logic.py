from unittest.mock import MagicMock

from mlflow.exceptions import MlflowException

from mlops_lab.registry import get_champion_metrics, metrics_from_tags, promote


def make_version(tags: dict[str, str], version: str = "3"):
    mv = MagicMock()
    mv.tags = tags
    mv.version = version
    return mv


def test_metrics_from_tags_parses_prefixed_floats():
    tags = {"metric.rmse": "0.53", "metric.r2": "0.81", "dataset": "california_housing"}
    assert metrics_from_tags(tags) == {"rmse": 0.53, "r2": 0.81}


def test_champion_metrics_when_alias_exists():
    client = MagicMock()
    client.get_model_version_by_alias.return_value = make_version({"metric.rmse": "0.5"})
    assert get_champion_metrics(client, "housing-price") == {"rmse": 0.5}
    client.get_model_version_by_alias.assert_called_once_with("housing-price", "production")


def test_champion_metrics_none_when_alias_missing():
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = MlflowException("alias not found")
    assert get_champion_metrics(client, "housing-price") is None


def test_champion_with_no_metric_tags_is_empty_dict():
    client = MagicMock()
    client.get_model_version_by_alias.return_value = make_version({"dataset": "x"})
    assert get_champion_metrics(client, "housing-price") == {}


def test_promote_reassigns_alias():
    client = MagicMock()
    client.get_model_version_by_alias.return_value = make_version({}, version="7")
    promote(client, "housing-price", "staging", "production")
    client.set_registered_model_alias.assert_called_once_with("housing-price", "production", "7")
