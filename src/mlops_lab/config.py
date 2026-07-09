"""Model candidate configuration: one YAML file per candidate, validated with pydantic."""

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class TaskType(StrEnum):
    regression = "regression"
    classification = "classification"


class Direction(StrEnum):
    minimize = "minimize"
    maximize = "maximize"


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_size: float = 0.2
    seed: int = 42


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    type: str
    params: dict = {}


class GateSpec(BaseModel):
    """Absolute threshold on a metric: at least one of min/max must be set."""

    model_config = ConfigDict(extra="forbid")

    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def at_least_one_bound(self) -> "GateSpec":
        if self.min is None and self.max is None:
            raise ValueError("gate must set at least one of min/max")
        return self


class ChampionConfig(BaseModel):
    """How a candidate is compared against the current @production champion."""

    model_config = ConfigDict(extra="forbid")

    primary_metric: str
    direction: Direction
    tolerance: float = 0.0


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str
    task: TaskType
    dataset: str
    registered_model: str
    split: SplitConfig = SplitConfig()
    model: ModelSpec
    gates: dict[str, GateSpec]
    champion: ChampionConfig


def load_config(path: Path | str) -> ModelConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return ModelConfig.model_validate(raw)
