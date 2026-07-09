"""Dataset loading with deterministic train/test splits."""

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Splits:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def _california_housing() -> tuple[pd.DataFrame, pd.Series]:
    bunch = fetch_california_housing(as_frame=True)
    return bunch.data, bunch.target


def _breast_cancer() -> tuple[pd.DataFrame, pd.Series]:
    bunch = load_breast_cancer(as_frame=True)
    return bunch.data, bunch.target


DATASETS = {
    "california_housing": _california_housing,
    "breast_cancer": _breast_cancer,
}

# Classification datasets stratify their split so class balance is stable across seeds.
CLASSIFICATION_DATASETS = {"breast_cancer"}


def load_dataset(name: str, test_size: float = 0.2, seed: int = 42) -> Splits:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    X, y = DATASETS[name]()
    stratify = y if name in CLASSIFICATION_DATASETS else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )
    return Splits(X_train, X_test, y_train, y_test)


def sample_rows(name: str, n: int = 5, seed: int = 42) -> pd.DataFrame:
    """A few feature rows for smoke-testing a deployed model."""
    X, _ = DATASETS[name]()
    return X.sample(n=n, random_state=seed)
