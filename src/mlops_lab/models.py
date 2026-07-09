"""Estimator factory: explicit whitelist, no dynamic imports."""

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, Ridge
from xgboost import XGBClassifier, XGBRegressor

ESTIMATORS: dict[str, type[BaseEstimator]] = {
    "sklearn.Ridge": Ridge,
    "sklearn.LogisticRegression": LogisticRegression,
    "xgboost.XGBRegressor": XGBRegressor,
    "xgboost.XGBClassifier": XGBClassifier,
}

# Estimators that benefit from feature scaling; tree models get passthrough.
NEEDS_SCALING = {"sklearn.Ridge", "sklearn.LogisticRegression"}


def build_estimator(type_: str, params: dict) -> BaseEstimator:
    if type_ not in ESTIMATORS:
        raise ValueError(f"unknown estimator {type_!r}; known: {sorted(ESTIMATORS)}")
    return ESTIMATORS[type_](**params)
