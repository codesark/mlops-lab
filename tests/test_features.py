import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from mlops_lab.features import build_pipeline
from tests.conftest import make_config


def test_linear_model_gets_scaler(regression_config):
    pipeline = build_pipeline(regression_config)
    assert [name for name, _ in pipeline.steps] == ["scaler", "estimator"]


def test_tree_model_skips_scaler():
    cfg = make_config(model={"type": "xgboost.XGBRegressor", "params": {"n_estimators": 5}})
    pipeline = build_pipeline(cfg)
    assert [name for name, _ in pipeline.steps] == ["estimator"]


def test_regression_pipeline_fit_predict(regression_config):
    X, y = make_regression(n_samples=100, n_features=5, random_state=0)
    X = pd.DataFrame(X)
    pipeline = build_pipeline(regression_config)
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert preds.shape == (100,)
    assert np.all(np.isfinite(preds))


def test_classification_pipeline_fit_predict(classification_config):
    X, y = make_classification(n_samples=100, n_features=5, random_state=0)
    X = pd.DataFrame(X)
    pipeline = build_pipeline(classification_config)
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)[:, 1]
    assert proba.shape == (100,)
    assert np.all((proba >= 0) & (proba <= 1))


def test_unknown_estimator_rejected():
    import pytest

    from mlops_lab.models import build_estimator

    with pytest.raises(ValueError, match="unknown estimator"):
        build_estimator("os.system", {})
