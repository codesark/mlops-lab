"""Preprocessing + estimator pipeline construction."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops_lab.config import ModelConfig
from mlops_lab.models import NEEDS_SCALING, build_estimator


def build_pipeline(cfg: ModelConfig) -> Pipeline:
    steps = []
    if cfg.model.type in NEEDS_SCALING:
        steps.append(("scaler", StandardScaler()))
    steps.append(("estimator", build_estimator(cfg.model.type, cfg.model.params)))
    return Pipeline(steps)
