"""Train a candidate model from its config and log the run to MLflow."""

from dataclasses import asdict, dataclass

import mlflow

from mlops_lab.config import ModelConfig, TaskType
from mlops_lab.data import load_dataset
from mlops_lab.evaluate import compute_metrics
from mlops_lab.features import build_pipeline


@dataclass(frozen=True)
class TrainResult:
    name: str
    run_id: str
    model_uri: str
    metrics: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def run_training(cfg: ModelConfig) -> TrainResult:
    splits = load_dataset(cfg.dataset, cfg.split.test_size, cfg.split.seed)
    pipeline = build_pipeline(cfg)

    mlflow.set_experiment(cfg.registered_model)
    with mlflow.start_run(run_name=cfg.name) as run:
        mlflow.log_params(
            {
                "model_type": cfg.model.type,
                "dataset": cfg.dataset,
                "test_size": cfg.split.test_size,
                "seed": cfg.split.seed,
                **{f"model.{k}": v for k, v in cfg.model.params.items()},
            }
        )
        pipeline.fit(splits.X_train, splits.y_train)

        y_pred = pipeline.predict(splits.X_test)
        y_proba = None
        if cfg.task == TaskType.classification:
            y_proba = pipeline.predict_proba(splits.X_test)[:, 1]
        metrics = compute_metrics(cfg.task, splits.y_test, y_pred, y_proba)
        mlflow.log_metrics(metrics)

        # MLflow 2.x API (artifact_path, not name=): Azure ML doesn't implement
        # the MLflow 3 logged-models endpoints.
        model_info = mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            input_example=splits.X_train.head(5),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
    return TrainResult(
        name=cfg.name,
        run_id=run.info.run_id,
        model_uri=model_info.model_uri,
        metrics=metrics,
    )
