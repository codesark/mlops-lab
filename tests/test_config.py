import pytest
from pydantic import ValidationError

from mlops_lab.config import load_config

VALID_YAML = """
name: housing_xgb
task: regression
dataset: california_housing
registered_model: housing-price
split: {test_size: 0.25, seed: 7}
model:
  type: xgboost.XGBRegressor
  params: {n_estimators: 10}
gates:
  rmse: {max: 0.6}
  r2: {min: 0.75}
champion:
  primary_metric: rmse
  direction: minimize
  tolerance: 0.01
"""


def write_yaml(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_valid_config_round_trip(tmp_path):
    cfg = load_config(write_yaml(tmp_path, VALID_YAML))
    assert cfg.name == "housing_xgb"
    assert cfg.split.test_size == 0.25
    assert cfg.model.params["n_estimators"] == 10
    assert cfg.gates["rmse"].max == 0.6
    assert cfg.champion.tolerance == 0.01


def test_rejects_bad_task(tmp_path):
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, VALID_YAML.replace("task: regression", "task: ranking")))


def test_rejects_bad_direction(tmp_path):
    with pytest.raises(ValidationError):
        load_config(
            write_yaml(tmp_path, VALID_YAML.replace("direction: minimize", "direction: down"))
        )


def test_rejects_empty_gate(tmp_path):
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, VALID_YAML.replace("rmse: {max: 0.6}", "rmse: {}")))


def test_rejects_unknown_field(tmp_path):
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, VALID_YAML + "\nunknown_field: 1\n"))
