import pytest

from mlops_lab.config import (
    ChampionConfig,
    Direction,
    GateSpec,
    ModelConfig,
    ModelSpec,
    SplitConfig,
    TaskType,
)


def make_config(**overrides) -> ModelConfig:
    base = dict(
        name="test_model",
        task=TaskType.regression,
        dataset="california_housing",
        registered_model="test-registered",
        split=SplitConfig(test_size=0.2, seed=42),
        model=ModelSpec(type="sklearn.Ridge", params={"alpha": 1.0}),
        gates={"rmse": GateSpec(max=1.0)},
        champion=ChampionConfig(
            primary_metric="rmse", direction=Direction.minimize, tolerance=0.02
        ),
    )
    base.update(overrides)
    return ModelConfig.model_validate(base)


@pytest.fixture
def regression_config() -> ModelConfig:
    return make_config()


@pytest.fixture
def classification_config() -> ModelConfig:
    return make_config(
        name="test_clf",
        task=TaskType.classification,
        dataset="breast_cancer",
        model=ModelSpec(type="sklearn.LogisticRegression", params={"max_iter": 1000}),
        gates={"f1": GateSpec(min=0.5)},
        champion=ChampionConfig(primary_metric="f1", direction=Direction.maximize, tolerance=0.01),
    )
