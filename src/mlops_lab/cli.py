"""mlops CLI: the interface used by developers, Makefile, and CI workflows."""

import json
import os
from pathlib import Path

import typer
from mlflow import MlflowClient

from mlops_lab.config import Direction, ModelConfig, load_config
from mlops_lab.deploy import mock_deploy
from mlops_lab.evaluate import ComparisonResult, GateResult, check_gates, compare_to_champion
from mlops_lab.registry import PRODUCTION, STAGING, get_champion_metrics, register_and_stage
from mlops_lab.registry import promote as promote_alias
from mlops_lab.train import run_training

app = typer.Typer(help="MLOps lab pipeline commands", add_completion=False)


def _require_tracking_uri() -> None:
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri.startswith("http"):
        typer.echo(
            "MLFLOW_TRACKING_URI must point at an mlflow server (http://...). "
            "Run `make mlflow-up` and export the URI — never use a file/sqlite URI "
            "directly, or artifact paths become machine-specific.",
            err=True,
        )
        raise typer.Exit(2)


def _default_metrics_path(cfg: ModelConfig) -> Path:
    return Path("metrics") / f"{cfg.name}.json"


@app.command()
def train(
    config: Path = typer.Option(..., help="Path to the candidate YAML config"),
    metrics_out: Path = typer.Option(None, help="Where to write run metadata JSON"),
) -> None:
    """Train one candidate, log the run to MLflow, write metrics JSON for later steps."""
    _require_tracking_uri()
    cfg = load_config(config)
    result = run_training(cfg)
    out = metrics_out or _default_metrics_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    typer.echo(f"trained {cfg.name}: run_id={result.run_id}")
    for name, value in sorted(result.metrics.items()):
        typer.echo(f"  {name} = {value:.4f}")


def _render_markdown(
    cfg: ModelConfig,
    metrics: dict[str, float],
    gate: GateResult,
    comparison: ComparisonResult | None,
) -> str:
    status = "✅ PASS" if gate.passed and (comparison is None or comparison.passed) else "❌ FAIL"
    lines = [
        f"### `{cfg.name}` → `{cfg.registered_model}` — {status}",
        "",
        "| metric | value | gate |",
        "|---|---|---|",
    ]
    for name, value in sorted(metrics.items()):
        gate_spec = cfg.gates.get(name)
        bounds = "—"
        if gate_spec is not None:
            parts = []
            if gate_spec.min is not None:
                parts.append(f"min {gate_spec.min}")
            if gate_spec.max is not None:
                parts.append(f"max {gate_spec.max}")
            bounds = ", ".join(parts)
        lines.append(f"| {name} | {value:.4f} | {bounds} |")
    if gate.failures:
        lines += ["", "**Gate failures:**"] + [f"- {f}" for f in gate.failures]
    if comparison is not None:
        icon = "✅" if comparison.passed else "❌"
        lines += ["", f"**Champion comparison:** {icon} {comparison.reason}"]
    return "\n".join(lines) + "\n"


@app.command()
def evaluate(
    config: Path = typer.Option(...),
    metrics: Path = typer.Option(None, help="Metrics JSON from `mlops train`"),
    champion_from_registry: bool = typer.Option(
        False, help="Also compare against the @production champion in the registry"
    ),
    output_md: Path = typer.Option(None, help="Write the markdown report here"),
) -> None:
    """Check absolute quality gates (and optionally the champion comparison). Exit 1 on failure."""
    cfg = load_config(config)
    metrics_path = metrics or _default_metrics_path(cfg)
    result = json.loads(metrics_path.read_text())

    gate = check_gates(result["metrics"], cfg.gates)
    comparison = None
    if champion_from_registry:
        _require_tracking_uri()
        champion_metrics = get_champion_metrics(MlflowClient(), cfg.registered_model)
        comparison = compare_to_champion(result["metrics"], champion_metrics, cfg.champion)

    report = _render_markdown(cfg, result["metrics"], gate, comparison)
    typer.echo(report)
    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(report)

    if not gate.passed or (comparison is not None and not comparison.passed):
        raise typer.Exit(1)


@app.command()
def register(
    model: str = typer.Option(..., help="Registered model name, e.g. housing-price"),
    metrics_dir: Path = typer.Option(Path("metrics")),
    configs_dir: Path = typer.Option(Path("configs")),
    git_sha: str = typer.Option(None),
) -> None:
    """Register the best gate-passing candidate for a registered model and set @staging."""
    _require_tracking_uri()
    candidates = []
    for config_path in sorted(configs_dir.glob("*.yaml")):
        cfg = load_config(config_path)
        if cfg.registered_model != model:
            continue
        metrics_path = metrics_dir / f"{cfg.name}.json"
        if not metrics_path.exists():
            typer.echo(f"skipping {cfg.name}: no metrics at {metrics_path}", err=True)
            continue
        result = json.loads(metrics_path.read_text())
        gate = check_gates(result["metrics"], cfg.gates)
        if not gate.passed:
            typer.echo(f"skipping {cfg.name}: gates failed ({'; '.join(gate.failures)})")
            continue
        candidates.append((cfg, result))

    if not candidates:
        typer.echo(f"no gate-passing candidates for {model}", err=True)
        raise typer.Exit(1)

    champion_cfg = candidates[0][0].champion
    best_cfg, best = (
        min(candidates, key=lambda c: c[1]["metrics"][champion_cfg.primary_metric])
        if champion_cfg.direction == Direction.minimize
        else max(candidates, key=lambda c: c[1]["metrics"][champion_cfg.primary_metric])
    )
    version = register_and_stage(
        MlflowClient(),
        model_uri=best["model_uri"],
        model_name=model,
        metrics=best["metrics"],
        dataset=best_cfg.dataset,
        git_sha=git_sha,
    )
    typer.echo(
        f"registered {model} v{version.version} from {best_cfg.name} "
        f"(run {best['run_id']}) → @{STAGING}"
    )


@app.command()
def promote(
    model: str = typer.Option(...),
    from_alias: str = typer.Option(STAGING),
    to_alias: str = typer.Option(PRODUCTION),
) -> None:
    """Reassign an alias, e.g. @staging version becomes @production."""
    _require_tracking_uri()
    version = promote_alias(MlflowClient(), model, from_alias, to_alias)
    typer.echo(f"promoted {model} v{version.version}: @{from_alias} → @{to_alias}")


@app.command()
def deploy(
    model: str = typer.Option(...),
    alias: str = typer.Option(STAGING),
    env: str = typer.Option(...),
    registry_root: Path = typer.Option(Path("registry")),
    actor: str = typer.Option(None),
    git_sha: str = typer.Option(None),
) -> None:
    """Mock-deploy a registry model to an environment: smoke test + manifest + history."""
    _require_tracking_uri()
    manifest = mock_deploy(model, alias, env, registry_root, actor=actor, git_sha=git_sha)
    typer.echo(
        f"deployed {model} v{manifest['version']} (@{alias}) to {env}; "
        f"manifest at {registry_root}/deployments/{env}/latest.json"
    )


if __name__ == "__main__":
    app()
