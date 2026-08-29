"""M13 pilot: compare online direct-tie topologies.

M13 is deliberately a small boundary around the established ``run_condition``
kernel.  It reuses the M11 participants and rules, but replaces only the
condition list.  In particular, no ``OnlineExperience`` (and therefore no
topic, wording, or memory) is attached to any M13 condition.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
import math
from statistics import mean
from typing import Any, Callable, Mapping, Sequence

from .config import load_m11_config
from .domain import Condition, OnsiteRelationship, Pair, PocConfig, canonical_pair
from .engine import run_condition


M13_SCHEMA_VERSION = "ab_poc.m13_topology.v1"
M13_CONFIG_VERSION = "m13_topology_v1"
M13_SCENARIO_VERSION = "m13.topology.v1"
M13_CONDITION_IDS = ("T0", "T1", "T2", "TC", "TH", "TF")
M13_DEFAULT_SEEDS = (26,)
_PARTICIPANTS = ("akane", "koharu", "midori", "kurumi")
_ALL_PAIRS = frozenset(
    canonical_pair(left, right)
    for index, left in enumerate(_PARTICIPANTS)
    for right in _PARTICIPANTS[index + 1 :]
)

_TOPOLOGIES: dict[str, tuple[str, tuple[Pair, ...]]] = {
    "T0": ("none", ()),
    "T1": ("one_pair", (canonical_pair("akane", "midori"),)),
    "T2": (
        "two_pairs",
        (canonical_pair("akane", "koharu"), canonical_pair("midori", "kurumi")),
    ),
    "TC": (
        "chain",
        (
            canonical_pair("akane", "koharu"),
            canonical_pair("koharu", "midori"),
            canonical_pair("midori", "kurumi"),
        ),
    ),
    "TH": (
        "hub",
        (
            canonical_pair("akane", "koharu"),
            canonical_pair("akane", "midori"),
            canonical_pair("akane", "kurumi"),
        ),
    ),
    "TF": ("full_mesh", tuple(sorted(_ALL_PAIRS))),
}


def _validate_pair_set(condition_id: str, pairs: Sequence[Pair]) -> tuple[Pair, ...]:
    if condition_id not in _TOPOLOGIES:
        raise ValueError(f"unsupported M13 condition: {condition_id}")
    canonical = tuple(canonical_pair(*pair) for pair in pairs)
    if len(set(canonical)) != len(canonical):
        raise ValueError(f"M13 condition {condition_id} contains duplicate pairs")
    expected = _TOPOLOGIES[condition_id][1]
    if set(canonical) != set(expected) or len(canonical) != len(expected):
        raise ValueError(f"M13 condition {condition_id} has an invalid pair set")
    return tuple(sorted(canonical))


def m13_conditions() -> tuple[Condition, ...]:
    """Return fresh immutable conditions for the six topology arms."""

    return tuple(
        Condition(
            condition_id,
            label,
            _validate_pair_set(condition_id, pairs),
            online_experience=None,
        )
        for condition_id, (label, pairs) in _TOPOLOGIES.items()
    )


@dataclass(frozen=True)
class M13TopologySpec:
    """Validation and execution metadata for a topology pilot."""

    seeds: tuple[int, ...] = M13_DEFAULT_SEEDS
    condition_ids: tuple[str, ...] = M13_CONDITION_IDS
    scenario_version: str = M13_SCENARIO_VERSION

    def __post_init__(self) -> None:
        if not self.seeds or any(type(seed) is not int for seed in self.seeds):
            raise ValueError("M13 seeds must be a non-empty tuple of integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("M13 seeds must not contain duplicates")
        if tuple(self.condition_ids) != M13_CONDITION_IDS:
            raise ValueError("M13 condition_ids must contain the six topology conditions in order")
        if not isinstance(self.scenario_version, str) or not self.scenario_version:
            raise ValueError("M13 scenario_version must be non-empty")

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": "m13_online_topology_pilot",
            "version": "m13.experiment.v1",
            "seeds": list(self.seeds),
            "conditions": list(self.condition_ids),
            "scenario_version": self.scenario_version,
            "online_experience": None,
        }


def validate_m13_config(config: PocConfig) -> PocConfig:
    """Validate the narrow M13 replacement boundary and return ``config``."""

    if tuple(person.id for person in config.participants) != _PARTICIPANTS:
        raise ValueError("M13 requires the established four participants in order")
    if config.config_version != M13_CONFIG_VERSION:
        raise ValueError("M13 config must use the M13 config version")
    if config.rules.turns != 8:
        raise ValueError("M13 requires exactly 8 turns")
    if tuple(condition.id for condition in config.conditions) != M13_CONDITION_IDS:
        raise ValueError("M13 config must contain exactly the six topology conditions")
    for condition in config.conditions:
        if condition.online_experience is not None:
            raise ValueError("M13 must not inject online_experience or online memory")
        _validate_pair_set(condition.id, condition.online_known_pairs)
    return config


def build_m13_config(base_config: PocConfig | None = None) -> PocConfig:
    """Clone the established config while replacing only conditions/metadata."""

    base = base_config or load_m11_config()
    config = replace(
        base,
        config_version=M13_CONFIG_VERSION,
        conditions=m13_conditions(),
        scenario_version=M13_SCENARIO_VERSION,
    )
    return validate_m13_config(config)


def _invoke_runner(
    runner: Callable[..., Mapping[str, Any]], *, condition_id: str, seed: int, config: PocConfig
) -> Mapping[str, Any]:
    parameters = inspect.signature(runner).parameters
    accepts_kwargs = any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values())
    kwargs: dict[str, Any] = {"seed": seed}
    if accepts_kwargs or "condition_id" in parameters:
        kwargs["condition_id"] = condition_id
    if accepts_kwargs or "config" in parameters:
        kwargs["config"] = config
    result = runner(**kwargs)
    if not isinstance(result, Mapping):
        raise ValueError("M13 runner must return a mapping")
    return result


def _stage(value: Any) -> OnsiteRelationship:
    if isinstance(value, OnsiteRelationship):
        return value
    return OnsiteRelationship(str(value))


def topology_observation(run: Mapping[str, Any]) -> dict[str, Any]:
    """Extract topology-neutral observations without changing RunResult."""

    condition = run.get("condition")
    if not isinstance(condition, Mapping):
        raise ValueError("M13 run is missing condition")
    condition_id = condition.get("id")
    if condition_id not in M13_CONDITION_IDS:
        raise ValueError("M13 run has an unsupported condition")
    known_pairs = condition.get("online_known_pairs")
    if not isinstance(known_pairs, list):
        raise ValueError("M13 run has invalid online_known_pairs")
    def _known_pair(value: Any) -> Pair:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("M13 run has an invalid online pair")
        try:
            return canonical_pair(str(value[0]), str(value[1]))
        except (TypeError, ValueError) as exc:
            raise ValueError("M13 run has an invalid online pair") from exc

    online_pairs = tuple(_known_pair(pair) for pair in known_pairs)
    metrics = run.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("M13 run is missing metrics")
    new_acquaintance_count = metrics.get("new_acquaintance_count")
    eligible_count = metrics.get("new_acquaintance_eligible_count")
    new_acquaintance_rate = metrics.get("new_acquaintance_rate")
    if type(new_acquaintance_count) is not int or new_acquaintance_count < 0:
        raise ValueError("M13 run has invalid new_acquaintance_count")
    if type(eligible_count) is not int or eligible_count < 0:
        raise ValueError("M13 run has invalid new_acquaintance_eligible_count")
    if (
        isinstance(new_acquaintance_rate, bool)
        or not isinstance(new_acquaintance_rate, (int, float))
        or not math.isfinite(float(new_acquaintance_rate))
        or not 0 <= new_acquaintance_rate <= 1
    ):
        raise ValueError("M13 run has invalid new_acquaintance_rate")
    turns = run.get("turns")
    if not isinstance(turns, list) or len(turns) != 8:
        raise ValueError("M13 run must contain exactly 8 turns")
    final = run.get("final_relationships")
    if not isinstance(final, list):
        raise ValueError("M13 run is missing final_relationships")
    onsite_edges = 0
    familiar_edges = 0
    final_by_pair: dict[Pair, OnsiteRelationship] = {}
    for state in final:
        if not isinstance(state, Mapping):
            raise ValueError("M13 run contains an invalid final pair")
        stage = _stage(state.get("onsite_relationship"))
        try:
            pair = canonical_pair(str(state["left_id"]), str(state["right_id"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("M13 run contains an invalid final pair identity") from exc
        final_by_pair[pair] = stage
        onsite_edges += stage is not OnsiteRelationship.NOT_FORMED
        familiar_edges += stage is OnsiteRelationship.FAMILIAR
    acquaintance_rank = {
        OnsiteRelationship.NOT_FORMED: 0,
        OnsiteRelationship.ACQUAINTANCE: 1,
        OnsiteRelationship.FAMILIAR: 2,
    }
    online_known_onsite_count = sum(
        acquaintance_rank.get(final_by_pair.get(pair, OnsiteRelationship.NOT_FORMED), 0) >= 1
        for pair in online_pairs
    )
    targets: dict[str, int] = {person: 0 for person in _PARTICIPANTS}
    for turn in turns:
        if not isinstance(turn, Mapping) or not isinstance(turn.get("target"), str):
            raise ValueError("M13 run contains an invalid turn target")
        if turn["target"] in targets:
            targets[turn["target"]] += 1
    max_receive = max(targets.values(), default=0)
    return {
        "condition": condition_id,
        "online_edge_count": len(online_pairs),
        "online_known_onsite_count": int(online_known_onsite_count),
        "new_acquaintance_count": new_acquaintance_count,
        "new_acquaintance_eligible_count": eligible_count,
        "new_acquaintance_rate": new_acquaintance_rate,
        "onsite_edge_count": int(onsite_edges),
        "onsite_familiar_edge_count": int(familiar_edges),
        "isolated_participant_count": len((run.get("metrics") or {}).get("isolated_participants", [])),
        "conversation_concentration": max_receive / len(turns),
        "max_target_receive_count": max_receive,
        "most_targeted_participants": [person for person, count in targets.items() if count == max_receive],
    }


def _summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observations = [topology_observation(run) for run in runs]
    fields = (
        "online_edge_count", "online_known_onsite_count",
        "new_acquaintance_count", "new_acquaintance_eligible_count", "new_acquaintance_rate",
        "onsite_edge_count", "onsite_familiar_edge_count",
        "isolated_participant_count", "conversation_concentration", "max_target_receive_count",
    )
    return {
        "run_count": len(runs),
        "metrics": {
            field: {
                "values": [item[field] for item in observations],
                "mean": round(mean(float(item[field]) for item in observations), 4),
                "min": min(item[field] for item in observations),
                "max": max(item[field] for item in observations),
            }
            for field in fields
        },
        "most_targeted_participants": [item["most_targeted_participants"] for item in observations],
    }


def run_m13(
    spec: M13TopologySpec | None = None,
    *,
    config: PocConfig | None = None,
    runner: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run all configured topology arms with paired seeds and aggregate them."""

    spec = spec or M13TopologySpec()
    runtime_config = build_m13_config(config)
    paired: list[dict[str, Any]] = []
    for seed in spec.seeds:
        row: dict[str, Any] = {"seed": seed}
        for condition_id in spec.condition_ids:
            result = (
                run_condition(condition_id, seed=seed, config=runtime_config)
                if runner is None else _invoke_runner(runner, condition_id=condition_id, seed=seed, config=runtime_config)
            )
            if result.get("condition_id") != condition_id:
                raise ValueError("M13 runner returned the wrong condition")
            row[condition_id] = dict(result)
        paired.append(row)
    runs_by_condition = {condition: [row[condition] for row in paired] for condition in spec.condition_ids}
    return {
        "schema_version": M13_SCHEMA_VERSION,
        "experiment": spec.public_dict(),
        "controlled_fields": {
            "changed_field": "condition.online_known_pairs",
            "same_seed_across_conditions": True,
            "online_experience": None,
            "turns": 8,
        },
        "condition_summaries": {condition: _summary(runs) for condition, runs in runs_by_condition.items()},
        "runs": paired,
        "interpretation_note": "小規模なトポロジー観察用であり、効果・因果・人間行動の予測を示さない。",
    }


__all__ = [
    "M13_CONFIG_VERSION", "M13_CONDITION_IDS", "M13_DEFAULT_SEEDS", "M13_SCHEMA_VERSION",
    "M13_SCENARIO_VERSION", "M13TopologySpec", "build_m13_config",
    "m13_conditions", "run_m13", "topology_observation", "validate_m13_config",
]
