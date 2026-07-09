"""Metric computation, absolute quality gates, and champion comparison."""

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from mlops_lab.config import ChampionConfig, Direction, GateSpec, TaskType


def compute_metrics(task: TaskType, y_true, y_pred, y_proba=None) -> dict[str, float]:
    if task == TaskType.regression:
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return metrics


@dataclass
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)


def check_gates(metrics: dict[str, float], gates: dict[str, GateSpec]) -> GateResult:
    failures = []
    for metric, gate in gates.items():
        if metric not in metrics:
            failures.append(f"{metric}: not computed")
            continue
        value = metrics[metric]
        if gate.min is not None and value < gate.min:
            failures.append(f"{metric}={value:.4f} below min {gate.min}")
        if gate.max is not None and value > gate.max:
            failures.append(f"{metric}={value:.4f} above max {gate.max}")
    return GateResult(passed=not failures, failures=failures)


@dataclass
class ComparisonResult:
    passed: bool
    reason: str
    candidate_value: float | None = None
    champion_value: float | None = None


def compare_to_champion(
    metrics: dict[str, float],
    champion_metrics: dict[str, float] | None,
    cfg: ChampionConfig,
) -> ComparisonResult:
    """Candidate must not be worse than the champion beyond the relative tolerance.

    champion_metrics=None means no champion exists yet (bootstrap): auto-pass.
    """
    candidate = metrics.get(cfg.primary_metric)
    if candidate is None:
        return ComparisonResult(False, f"candidate missing metric {cfg.primary_metric!r}")
    if champion_metrics is None:
        return ComparisonResult(True, "no champion (bootstrap)", candidate_value=candidate)
    champion = champion_metrics.get(cfg.primary_metric)
    if champion is None:
        return ComparisonResult(
            True,
            f"champion has no {cfg.primary_metric!r} metric (bootstrap)",
            candidate_value=candidate,
        )

    if cfg.direction == Direction.minimize:
        limit = champion * (1 + cfg.tolerance)
        passed = candidate <= limit
    else:
        limit = champion * (1 - cfg.tolerance)
        passed = candidate >= limit
    verdict = "within tolerance of" if passed else "worse than"
    reason = (
        f"{cfg.primary_metric}={candidate:.4f} {verdict} champion "
        f"{champion:.4f} (tolerance {cfg.tolerance:.0%})"
    )
    return ComparisonResult(passed, reason, candidate_value=candidate, champion_value=champion)
