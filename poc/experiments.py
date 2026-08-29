from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Callable

from .engine import load_experiment_presets, load_scenarios, run_scenario


SUPPORTED_PROVIDERS = ("rule_based", "ollama")
AGGREGATED_METRICS = (
    "relationship_count",
    "relationship_continuity",
    "relationship_bias",
    "onsite_new_relationship_count",
    "online_to_onsite_activation_count",
)


@dataclass(frozen=True)
class ExperimentSpec:
    scenario_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    provider: str = "rule_based"
    preset: str = "poc"
    steps: int | None = None

    @classmethod
    def from_payload(cls, payload: object) -> ExperimentSpec:
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")

        scenario_ids = payload.get("scenario_ids")
        if not isinstance(scenario_ids, list) or not scenario_ids:
            raise ValueError("scenario_ids must be a non-empty array")
        if any(not isinstance(value, str) or not value for value in scenario_ids):
            raise ValueError("scenario_ids must contain only non-empty strings")
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("scenario_ids must not contain duplicates")

        known_scenarios = load_scenarios()
        unknown_scenarios = [value for value in scenario_ids if value not in known_scenarios]
        if unknown_scenarios:
            raise ValueError(f"Unknown scenario_ids: {unknown_scenarios}")

        seeds = payload.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise ValueError("seeds must be a non-empty array")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in seeds):
            raise ValueError("seeds must contain only integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must not contain duplicates")

        provider = payload.get("provider", "rule_based")
        if not isinstance(provider, str) or provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"provider must be one of: {', '.join(SUPPORTED_PROVIDERS)}"
            )

        preset = payload.get("preset", "poc")
        presets = load_experiment_presets()
        if not isinstance(preset, str) or preset not in presets:
            raise ValueError(f"Unknown experiment preset: {preset}")

        steps = payload.get("steps")
        if steps is not None and (
            not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0
        ):
            raise ValueError("steps must be a positive integer")

        return cls(
            scenario_ids=tuple(scenario_ids),
            seeds=tuple(seeds),
            provider=provider,
            preset=preset,
            steps=steps,
        )

    def public_dict(self) -> dict:
        value = {
            "scenario_ids": list(self.scenario_ids),
            "seeds": list(self.seeds),
            "provider": self.provider,
            "preset": self.preset,
        }
        if self.steps is not None:
            value["steps"] = self.steps
        return value


def run_experiment(
    spec: ExperimentSpec,
    *,
    runner: Callable[..., dict] | None = None,
) -> dict:
    """Run every scenario/seed pair and return UI-independent aggregate data."""
    scenario_runner = runner or run_scenario
    scenarios = load_scenarios()
    conditions = []
    for scenario_id in spec.scenario_ids:
        runs = [
            scenario_runner(
                scenario_id,
                seed=seed,
                provider=spec.provider,
                preset=spec.preset,
                steps=spec.steps,
            )
            for seed in spec.seeds
        ]
        representative = _select_representative(runs)
        conditions.append(
            {
                "scenario_id": scenario_id,
                "label": scenarios[scenario_id].label,
                "run_count": len(runs),
                "metrics": {
                    metric_name: _aggregate_metric(runs, metric_name)
                    for metric_name in AGGREGATED_METRICS
                },
                "runs": [
                    {"seed": run["seed"], "metrics": run.get("metrics", {})}
                    for run in runs
                ],
                "representative_seed": representative["seed"],
                "representative_run": representative,
            }
        )

    return {
        "schema_version": 1,
        "spec": spec.public_dict(),
        "conditions": conditions,
    }


def _aggregate_metric(runs: list[dict], metric_name: str) -> dict:
    values = [run.get("metrics", {}).get(metric_name, 0) for run in runs]
    if not values:
        return {"mean": 0, "min": 0, "max": 0}
    return {
        "mean": round(sum(values) / len(values), 2),
        "min": min(values),
        "max": max(values),
    }


def _select_representative(runs: list[dict]) -> dict:
    if not runs:
        raise ValueError("Cannot select a representative from zero runs")
    median_count = median(
        run.get("metrics", {}).get("relationship_count", 0) for run in runs
    )
    return min(
        runs,
        key=lambda run: (
            abs(run.get("metrics", {}).get("relationship_count", 0) - median_count),
            run["seed"],
        ),
    )
