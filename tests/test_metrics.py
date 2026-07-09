import numpy as np
import pytest

from mlops_lab.config import TaskType
from mlops_lab.evaluate import compute_metrics


def test_perfect_regression():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = compute_metrics(TaskType.regression, y, y)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_known_regression_values():
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([1.0, -1.0, 1.0, -1.0])
    metrics = compute_metrics(TaskType.regression, y_true, y_pred)
    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(1.0)


def test_known_classification_values():
    # TP=2, FP=1, FN=1, TN=2 -> precision=2/3, recall=2/3, f1=2/3
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0])
    metrics = compute_metrics(TaskType.classification, y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert "roc_auc" not in metrics  # no probabilities given


def test_roc_auc_with_proba():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics(TaskType.classification, y_true, (y_proba > 0.5).astype(int), y_proba)
    assert metrics["roc_auc"] == pytest.approx(1.0)
