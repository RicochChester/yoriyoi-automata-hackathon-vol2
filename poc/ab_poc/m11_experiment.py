"""Small, provider-neutral aggregation for the M11 online-prior A/B run.

M11 deliberately keeps execution separate from presentation and from any LLM
transport.  The default runner is the deterministic ``run_ab`` kernel; a
future experiment can inject an equivalent runner after its provider has
passed its own quality gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from statistics import mean
from typing import Any, Callable, Mapping, Sequence

from .config import load_legacy_config, load_m11_config
from .engine import run_ab, run_condition


SCHEMA_VERSION = "ab_poc.m11_experiment.v1"
DEFAULT_SEEDS = (26, 27)
M11_CONDITION_IDS = ("N0", "P2", "P3", "G2", "G3")
_METRIC_NAMES = (
    "new_acquaintance_count",
    "new_acquaintance_rate",
    "new_familiar_count",
    "isolated_participant_count",
    "third_party_observation_path_count",
    "conversation_concentration",
    "max_target_receive_count",
    "targeted_participant_count",
)
_STAGE_RANK = {"未形成": 0, "顔見知り": 1, "親しみがある": 2}


@dataclass(frozen=True)
class M11ExperimentSpec:
    """The fixed, paired seed set for one M11 experiment."""

    seeds: tuple[int, ...] = DEFAULT_SEEDS
    scenario_version: str = "m10.3.scenario.v1"
    condition_ids: tuple[str, ...] = ("A", "B")

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("M11 seeds must not be empty")
        if any(type(seed) is not int for seed in self.seeds):
            raise ValueError("M11 seeds must contain integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("M11 seeds must not contain duplicates")
        if not isinstance(self.scenario_version, str) or not self.scenario_version:
            raise ValueError("M11 scenario_version must be non-empty")
        if not self.condition_ids or len(set(self.condition_ids)) != len(self.condition_ids):
            raise ValueError("M11 condition_ids must be non-empty and unique")
        supported = {"A", "B", *M11_CONDITION_IDS}
        if any(condition not in supported for condition in self.condition_ids):
            raise ValueError("M11 condition_ids contains an unsupported condition")

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": "m11_online_prior_ab",
            "version": "m11.experiment.v1",
            "seeds": list(self.seeds),
            "conditions": list(self.condition_ids),
            "scenario_version": self.scenario_version,
        }


def _observations(run: Mapping[str, Any]) -> dict[str, Any]:
    metrics = run.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("M11 run is missing metrics")
    isolated = metrics.get("isolated_participants", [])
    paths = metrics.get("third_party_observation_paths", [])
    known_pair = metrics.get("known_pair_onsite")
    most_targeted = metrics.get("most_targeted_participants", [])
    if not isinstance(isolated, list) or not all(isinstance(item, str) for item in isolated):
        raise ValueError("M11 run has invalid isolated_participants")
    if paths is not None and not isinstance(paths, list):
        raise ValueError("M11 run has invalid third_party_observation_paths")
    if known_pair is not None and not isinstance(known_pair, Mapping):
        raise ValueError("M11 run has invalid known_pair_onsite")
    if not isinstance(most_targeted, list) or not all(isinstance(item, str) for item in most_targeted):
        raise ValueError("M11 run has invalid most_targeted_participants")
    turns = run.get("turns")
    if not isinstance(turns, list):
        raise ValueError("M11 run is missing turns")
    known_conversations = 0
    for turn in turns:
        if not isinstance(turn, Mapping):
            raise ValueError("M11 run has an invalid turn")
        condition = run.get("condition", {})
        known_pairs = condition.get("online_known_pairs", []) if isinstance(condition, Mapping) else []
        pair = tuple(sorted((str(turn.get("actor")), str(turn.get("target")))))
        if any(tuple(sorted(item)) == pair for item in known_pairs):
            known_conversations += 1
    stage = None
    if isinstance(known_pair, Mapping):
        stage = known_pair.get("stage")
        if stage not in _STAGE_RANK:
            raise ValueError("M11 run has an invalid known pair stage")
    values = {
        "new_acquaintance_count": int(metrics.get("new_acquaintance_count", 0)),
        "new_acquaintance_rate": float(metrics.get("new_acquaintance_rate", 0.0)),
        "new_familiar_count": int(metrics.get("new_familiar_count", 0)),
        "isolated_participant_count": len(isolated),
        "third_party_observation_path_count": len(paths or []),
        "conversation_concentration": float(metrics.get("conversation_concentration", 0.0)),
        "max_target_receive_count": int(metrics.get("max_target_receive_count", 0)),
        "targeted_participant_count": int(metrics.get("targeted_participant_count", 0)),
        "most_targeted_participants": most_targeted,
    }
    return {
        **values,
        "online_known_pair_conversation_count": known_conversations,
        "known_pair_onsite_stage": stage,
    }


def _summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observations = [_observations(run) for run in runs]
    return {
        "run_count": len(runs),
        "metrics": {
            name: {
                "values": [item[name] for item in observations],
                "mean": round(mean(item[name] for item in observations), 4),
                "min": min(item[name] for item in observations),
                "max": max(item[name] for item in observations),
            }
            for name in _METRIC_NAMES
        },
        "online_known_pair_conversation_counts": [
            item["online_known_pair_conversation_count"] for item in observations
        ],
        "known_pair_onsite_stages": [item["known_pair_onsite_stage"] for item in observations],
        "most_targeted_participants": [item["most_targeted_participants"] for item in observations],
    }


def _provider_metadata(run: Mapping[str, Any], field: str) -> dict[str, str]:
    value = run.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"M11 run is missing {field} metadata")
    name, version = value.get("name"), value.get("version")
    if not all(isinstance(item, str) and item for item in (name, version)):
        raise ValueError(f"M11 run has invalid {field} metadata")
    return {"name": name, "version": version}


def _invoke_runner(runner: Callable[..., dict[str, Any]], *, seed: int, condition_id: str, config: Any) -> Mapping[str, Any]:
    """Call either the legacy seed-only seam or the condition-aware seam."""

    parameters = inspect.signature(runner).parameters
    accepts_kwargs = any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values())
    kwargs: dict[str, Any] = {"seed": seed}
    if accepts_kwargs or "condition_id" in parameters:
        kwargs["condition_id"] = condition_id
    if accepts_kwargs or "config" in parameters:
        kwargs["config"] = config
    result = runner(**kwargs)
    if not isinstance(result, Mapping):
        raise ValueError("M11 runner must return a mapping")
    return result


def run_m11(
    spec: M11ExperimentSpec | None = None,
    *,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run paired A/B seeds and return full runs plus compact comparisons.

    ``runner`` is a test seam and a future provider seam.  It must accept a
    keyword ``seed`` and return the established ``run_ab`` shape.  No LLM or
    network provider is constructed here.
    """

    spec = spec or M11ExperimentSpec()
    condition_ids = tuple(spec.condition_ids)
    if set(condition_ids) == {"A", "B"} and len(condition_ids) == 2:
        config = load_legacy_config()
    elif condition_ids == M11_CONDITION_IDS:
        config = load_m11_config()
    else:
        raise ValueError("M11 condition_ids must be A/B or the five standard M11 conditions")
    paired: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, str]] | None = None
    for seed in spec.seeds:
        runs: dict[str, dict[str, Any]] = {}
        if set(condition_ids) == {"A", "B"} and len(condition_ids) == 2:
            result = run_ab(seed=seed, config=config) if runner is None else _invoke_runner(
                runner, seed=seed, condition_id="A", config=config
            )
            raw_runs = result.get("runs") if isinstance(result, Mapping) else None
            if not isinstance(raw_runs, Mapping):
                raise ValueError("M11 runner must return a run_ab-shaped result")
            runs = {condition: dict(raw_runs[condition]) for condition in condition_ids}
        else:
            for condition_id in condition_ids:
                if runner is None:
                    runs[condition_id] = run_condition(condition_id, seed=seed, config=config)
                else:
                    result = _invoke_runner(runner, seed=seed, condition_id=condition_id, config=config)
                    raw_runs = result.get("runs")
                    if isinstance(raw_runs, Mapping) and condition_id in raw_runs:
                        runs[condition_id] = dict(raw_runs[condition_id])
                    elif "condition_id" in result or "turns" in result:
                        runs[condition_id] = dict(result)
                    else:
                        raise ValueError("M11 runner must return a run-shaped result")
        seed_metadata = {
            field: _provider_metadata(runs[condition_ids[0]], field)
            for field in ("participant_provider", "relationship_evaluator")
        }
        for condition_id, run in runs.items():
            for field, value in seed_metadata.items():
                if value != _provider_metadata(run, field):
                    raise ValueError(f"M11 {field} metadata differs between conditions")
            if metadata is not None:
                for field, value in seed_metadata.items():
                    if value != metadata[field]:
                        raise ValueError(f"M11 {field} metadata differs across seeds")
        metadata = seed_metadata
        paired.append({"seed": seed, **runs})

    condition_runs = {condition: [item[condition] for item in paired] for condition in condition_ids}
    deltas: dict[str, Any] = {}
    baseline = condition_ids[0]
    for condition in condition_ids[1:]:
        deltas[condition] = {
            name: {
                "values": [
                    round(float(_observations(item[condition])[name]) - float(_observations(item[baseline])[name]), 4)
                    for item in paired
                ],
            }
            for name in _METRIC_NAMES
        }
        for value in deltas[condition].values():
            values = value["values"]
            value.update({"mean": round(mean(values), 4), "min": min(values), "max": max(values)})

    experiment = spec.public_dict()
    experiment["scenario_version"] = config.scenario_version
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment,
        "controlled_fields": {
            "changed_field": "condition.online_experience",
            "conditions": list(condition_ids),
            "same_seed_per_pair": True,
            "provider": metadata["participant_provider"],
            "relationship_evaluator": metadata["relationship_evaluator"],
        },
        "condition_summaries": {condition: _summary(runs) for condition, runs in condition_runs.items()},
        "runs": paired,
        "interpretation_note": (
            "この出力は小規模なA/B観察用であり、使用したProviderの結果を一般化したものではない。"
            "効果・因果・人間行動の予測を示さない。"
        ),
    }
    if len(condition_ids) == 2:
        result["paired_deltas"] = {
            name: {
                "values_B_minus_A": deltas[condition_ids[1]][name]["values"],
                "mean": deltas[condition_ids[1]][name]["mean"],
                "min": deltas[condition_ids[1]][name]["min"],
                "max": deltas[condition_ids[1]][name]["max"],
            }
            for name in _METRIC_NAMES
        }
    else:
        result["baseline_condition"] = baseline
        result["condition_deltas"] = deltas
    return result


__all__ = ["DEFAULT_SEEDS", "M11_CONDITION_IDS", "M11ExperimentSpec", "SCHEMA_VERSION", "run_m11"]
