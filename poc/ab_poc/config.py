"""Strict loader for the versioned A/B POC configuration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from .domain import (
    Condition,
    OnlineExperience,
    Participant,
    PocConfig,
    ReactionPoints,
    RelationshipThresholds,
    RuleConfig,
    SelectionWeights,
    Stance,
    to_jsonable,
)


SCHEMA_VERSION = "ab_poc.schema.v1"
CONFIG_VERSION = "ab_poc_v2"
LEGACY_CONFIG_VERSION = "ab_poc_v1"
M11_CONFIG_VERSION = "m11_online_prior_v1"
SCENARIO_VERSION = "m10.3.scenario.v1"
PARTICIPANT_IDS = ("akane", "koharu", "midori", "kurumi")
CONDITION_IDS = ("A", "B")
_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "ab_poc_v2.json"
_M11_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "m11_online_prior_v1.json"


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing {sorted(missing)}")
        if extra:
            detail.append(f"unknown {sorted(extra)}")
        raise ValueError(f"{name} has invalid keys ({'; '.join(detail)})")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be {'positive and finite' if positive else 'finite'}")
    return result


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _pair(value: Any, name: str, participant_ids: set[str]) -> tuple[str, str]:
    if not isinstance(value, list) or len(value) != 2 or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a two-item string array")
    left, right = value
    if left not in participant_ids or right not in participant_ids:
        raise ValueError(f"{name} references an unknown participant")
    if left == right:
        raise ValueError(f"{name} cannot reference the same participant twice")
    return tuple(sorted((left, right)))  # type: ignore[return-value]


def _parse_participants(value: Any) -> tuple[Participant, ...]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("participants must contain exactly four participants")
    expected = {"id", "name", "interests", "stance", "purpose", "conversation_style", "opens_up_when"}
    result: list[Participant] = []
    for index, raw in enumerate(value):
        item = _object(raw, f"participants[{index}]")
        # Preserve compatibility with pre-persona mappings supplied by
        # callers while making the new card explicit in loaded config.
        item = dict(item)
        item.setdefault("purpose", "")
        item.setdefault("conversation_style", "")
        item.setdefault("opens_up_when", "")
        _keys(item, expected, f"participants[{index}]")
        for persona_key in ("purpose", "conversation_style", "opens_up_when"):
            if not isinstance(item[persona_key], str):
                raise ValueError(f"participants[{index}].{persona_key} must be a string")
        interests = item["interests"]
        if not isinstance(interests, list) or not interests or any(not isinstance(x, str) or not x for x in interests):
            raise ValueError(f"participants[{index}].interests must be a non-empty string array")
        try:
            stance = Stance(item["stance"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"participants[{index}].stance is invalid") from exc
        result.append(Participant(
            _string(item["id"], f"participants[{index}].id"),
            _string(item["name"], f"participants[{index}].name"),
            tuple(interests),
            stance,
            _string(item["purpose"], f"participants[{index}].purpose") if item["purpose"] else "",
            _string(item["conversation_style"], f"participants[{index}].conversation_style") if item["conversation_style"] else "",
            _string(item["opens_up_when"], f"participants[{index}].opens_up_when") if item["opens_up_when"] else "",
        ))
    ids = tuple(item.id for item in result)
    if ids != PARTICIPANT_IDS:
        raise ValueError(f"participants must be ordered as {list(PARTICIPANT_IDS)}")
    return tuple(result)


def _parse_online_experience(value: Any, name: str, participant_ids: set[str]) -> OnlineExperience:
    item = _object(value, name)
    _keys(item, {"id", "version", "kind", "participants", "topic", "exchange_count", "memory_summary"}, name)
    participants_value = item["participants"]
    if not isinstance(participants_value, list) or any(not isinstance(x, str) for x in participants_value):
        raise ValueError(f"{name}.participants must be a string array")
    participants = tuple(participants_value)
    if any(x not in participant_ids for x in participants):
        raise ValueError(f"{name}.participants references an unknown participant")
    topic = item["topic"]
    if topic is not None and (not isinstance(topic, str) or not topic):
        raise ValueError(f"{name}.topic must be a non-empty string or null")
    experience = OnlineExperience(
        _string(item["id"], f"{name}.id"),
        _string(item["version"], f"{name}.version"),
        _string(item["kind"], f"{name}.kind"),
        participants,
        topic,
        _integer(item["exchange_count"], f"{name}.exchange_count", minimum=0),
        _string(item["memory_summary"], f"{name}.memory_summary"),
    )
    return experience


def _parse_conditions(value: Any, participant_ids: set[str], *, config_version: str) -> tuple[Condition, ...]:
    expected_condition_ids = (
        ("A", "B")
        if config_version in {CONFIG_VERSION, LEGACY_CONFIG_VERSION}
        else ("N0", "P2", "P3", "G2", "G3")
    )
    if not isinstance(value, list) or len(value) != len(expected_condition_ids):
        raise ValueError(f"conditions must contain exactly {', '.join(expected_condition_ids)}")
    expected = {"id", "label", "online_known_pairs", "online_experience"}
    result: list[Condition] = []
    for index, raw in enumerate(value):
        item = _object(raw, f"conditions[{index}]")
        item = dict(item)
        item.setdefault("online_experience", None)
        _keys(item, expected, f"conditions[{index}]")
        pairs_value = item["online_known_pairs"]
        if not isinstance(pairs_value, list):
            raise ValueError(f"conditions[{index}].online_known_pairs must be an array")
        pairs = tuple(_pair(pair, f"conditions[{index}].online_known_pairs[{pair_index}]", participant_ids) for pair_index, pair in enumerate(pairs_value))
        experience_value = item.get("online_experience")
        if experience_value is None:
            # Accept the pre-M10.3 config shape.  ``online_known`` remains a
            # compatibility projection while the new memory stays generic.
            if pairs:
                experience = OnlineExperience(
                    f"legacy_direct_{pairs[0][0]}_{pairs[0][1]}",
                    "legacy.online_experience.v1", "direct_exchange", pairs[0],
                    "技術", 1, "技術について1往復した",
                )
            else:
                experience = OnlineExperience(
                    "none", "legacy.online_experience.v1", "none", (), None, 0,
                    "直接交流なし",
                )
        else:
            experience = _parse_online_experience(experience_value, f"conditions[{index}].online_experience", participant_ids)
        if experience.kind in {"direct_exchange", "pair_collaborative_task"} and (tuple(sorted(experience.participants)),) != pairs:
            raise ValueError(f"condition {item.get('id', index)} online_known_pairs must match online_experience participants")
        if experience.kind in {"group_thematic_exchange", "group_collaborative_task"} and pairs:
            raise ValueError(f"condition {item.get('id', index)} group online_experience cannot have online_known_pairs")
        if experience.kind == "none" and pairs:
            raise ValueError(f"conditions[{index}] none online_experience cannot have online_known_pairs")
        result.append(Condition(_string(item["id"], f"conditions[{index}].id"), _string(item["label"], f"conditions[{index}].label"), pairs, experience))
    if tuple(item.id for item in result) != expected_condition_ids:
        raise ValueError(f"conditions must be ordered as {', '.join(expected_condition_ids)}")
    if config_version == M11_CONFIG_VERSION:
        expected_kinds = {
            "N0": "none",
            "P2": "direct_exchange",
            "P3": "pair_collaborative_task",
            "G2": "group_thematic_exchange",
            "G3": "group_collaborative_task",
        }
        for item in result:
            if item.online_experience is None or item.online_experience.kind != expected_kinds[item.id]:
                raise ValueError(f"condition {item.id} has an invalid online experience kind")
        if result[1].online_experience is None or result[1].online_experience.participants != ("akane", "midori"):
            raise ValueError("condition P2 must contain only the akane-midori pair")
        if result[2].online_experience is None or result[2].online_experience.participants != ("akane", "midori"):
            raise ValueError("condition P3 must contain only the akane-midori pair")
        if any(item.online_experience is None or item.online_experience.participants != tuple(sorted(PARTICIPANT_IDS)) for item in result[3:]):
            raise ValueError("conditions G2 and G3 must contain all participants")
        return tuple(result)
    if result[0].online_known_pairs:
        raise ValueError("condition A must have no direct online known pairs")
    if result[1].online_known_pairs != (("akane", "midori"),):
        raise ValueError("condition B must contain only the akane-midori pair")
    experience_a = result[0].online_experience
    experience_b = result[1].online_experience
    if experience_a is None or experience_a.kind != "none":
        raise ValueError("condition A must have no online prior experience")
    if (
        experience_b is None
        or experience_b.kind != "direct_exchange"
        or experience_b.participants != ("akane", "midori")
        or experience_b.topic != "技術"
        or experience_b.exchange_count != 1
    ):
        raise ValueError("condition B must contain one akane-midori 技術 exchange")
    return tuple(result)


def _parse_rules(value: Any, *, config_version: str) -> RuleConfig:
    raw = _object(value, "rules")
    _keys(raw, {"turns", "common_topic", "selection", "reaction", "relationship"}, "rules")
    turns = _integer(raw["turns"], "rules.turns", minimum=1)
    expected_turns = 12 if config_version == CONFIG_VERSION else 8
    if turns != expected_turns:
        raise ValueError(f"rules.turns must be fixed at {expected_turns} for {config_version}")
    selection_raw = _object(raw["selection"], "rules.selection")
    _keys(selection_raw, {"base", "online_known_bonus", "shared_interest_bonus", "prior_positive_bonus", "repeated_target_multiplier"}, "rules.selection")
    selection = SelectionWeights(*(_number(selection_raw[key], f"rules.selection.{key}", positive=True) for key in ("base", "online_known_bonus", "shared_interest_bonus", "prior_positive_bonus", "repeated_target_multiplier")))
    reaction_raw = _object(raw["reaction"], "rules.reaction")
    _keys(reaction_raw, {"base", "online_known_bonus", "topic_match_bonus", "prior_positive_bonus", "proactive_actor_bonus", "cautious_actor_penalty", "cautious_target_penalty", "jitter_values", "positive_threshold", "neutral_threshold"}, "rules.reaction")
    jitter = reaction_raw["jitter_values"]
    if not isinstance(jitter, list) or not jitter or any(isinstance(x, bool) or not isinstance(x, int) for x in jitter):
        raise ValueError("rules.reaction.jitter_values must be a non-empty integer array")
    reaction = ReactionPoints(
        _integer(reaction_raw["base"], "rules.reaction.base"),
        _integer(reaction_raw["online_known_bonus"], "rules.reaction.online_known_bonus"),
        _integer(reaction_raw["topic_match_bonus"], "rules.reaction.topic_match_bonus"),
        _integer(reaction_raw["prior_positive_bonus"], "rules.reaction.prior_positive_bonus"),
        _integer(reaction_raw["proactive_actor_bonus"], "rules.reaction.proactive_actor_bonus"),
        _integer(reaction_raw["cautious_actor_penalty"], "rules.reaction.cautious_actor_penalty"),
        _integer(reaction_raw["cautious_target_penalty"], "rules.reaction.cautious_target_penalty"),
        tuple(jitter),
        _integer(reaction_raw["positive_threshold"], "rules.reaction.positive_threshold"),
        _integer(reaction_raw["neutral_threshold"], "rules.reaction.neutral_threshold"),
    )
    if reaction.positive_threshold <= reaction.neutral_threshold:
        raise ValueError("positive threshold must exceed neutral threshold")
    relationship_raw = _object(raw["relationship"], "rules.relationship")
    _keys(relationship_raw, {"acquaintance_conversations", "acquaintance_positive", "familiar_conversations", "familiar_positive", "familiar_requires_bidirectional"}, "rules.relationship")
    if not isinstance(relationship_raw["familiar_requires_bidirectional"], bool):
        raise ValueError("rules.relationship.familiar_requires_bidirectional must be boolean")
    relationship = RelationshipThresholds(
        _integer(relationship_raw["acquaintance_conversations"], "rules.relationship.acquaintance_conversations", minimum=1),
        _integer(relationship_raw["acquaintance_positive"], "rules.relationship.acquaintance_positive", minimum=1),
        _integer(relationship_raw["familiar_conversations"], "rules.relationship.familiar_conversations", minimum=1),
        _integer(relationship_raw["familiar_positive"], "rules.relationship.familiar_positive", minimum=1),
        relationship_raw["familiar_requires_bidirectional"],
    )
    if relationship.acquaintance_positive > relationship.acquaintance_conversations:
        raise ValueError("acquaintance positive threshold cannot exceed conversation threshold")
    if relationship.familiar_positive > relationship.familiar_conversations:
        raise ValueError("familiar positive threshold cannot exceed conversation threshold")
    if relationship.familiar_conversations < relationship.acquaintance_conversations or relationship.familiar_positive < relationship.acquaintance_positive:
        raise ValueError("familiar thresholds must not be below acquaintance thresholds")
    return RuleConfig(turns, _string(raw["common_topic"], "rules.common_topic"), selection, reaction, relationship)


def parse_config(data: Mapping[str, Any]) -> PocConfig:
    """Validate and parse an already-decoded configuration mapping."""

    raw = _object(data, "config")
    expected = {"schema_version", "config_version", "participants", "conditions", "rules", "scenario_version"}
    # Older callers may still provide an ab_poc_v1 mapping without the new
    # scenario metadata; the loader supplies the stable default.
    validated = dict(raw)
    validated.setdefault("scenario_version", SCENARIO_VERSION)
    _keys(validated, expected, "config")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    if raw["config_version"] not in {CONFIG_VERSION, LEGACY_CONFIG_VERSION, M11_CONFIG_VERSION}:
        raise ValueError(
            f"config_version must be {CONFIG_VERSION!r}, {LEGACY_CONFIG_VERSION!r}, or {M11_CONFIG_VERSION!r}"
        )
    participants = _parse_participants(raw["participants"])
    conditions = _parse_conditions(raw["conditions"], {item.id for item in participants}, config_version=raw["config_version"])
    rules = _parse_rules(raw["rules"], config_version=raw["config_version"])
    # A/B are intentionally identical apart from the online direct-pair list;
    # participants and rules are shared objects after parsing.
    return PocConfig(SCHEMA_VERSION, raw["config_version"], participants, conditions, rules, _string(validated["scenario_version"], "scenario_version"))


def load_config(path: str | Path | None = None) -> PocConfig:
    config_path = Path(path) if path is not None else _DEFAULT_PATH
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"unable to read config: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON config: {config_path}") from exc
    return parse_config(data)


def load_legacy_config(path: str | Path | None = None) -> PocConfig:
    """Load the preserved eight-turn A/B configuration used by legacy runs."""

    config_path = Path(path) if path is not None else Path(__file__).resolve().parents[1] / "config" / "ab_poc_v1.json"
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"unable to read config: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON config: {config_path}") from exc
    return parse_config(data)


def load_m11_config(path: str | Path | None = None) -> PocConfig:
    """Load the five-condition M11 online-prior configuration."""

    config_path = Path(path) if path is not None else _M11_DEFAULT_PATH
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"unable to read config: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON config: {config_path}") from exc
    return parse_config(data)


def config_to_dict(config: PocConfig) -> dict[str, Any]:
    return to_jsonable(config)


__all__ = ["CONFIG_VERSION", "LEGACY_CONFIG_VERSION", "M11_CONFIG_VERSION", "SCENARIO_VERSION", "SCHEMA_VERSION", "config_to_dict", "load_config", "load_legacy_config", "load_m11_config", "parse_config"]
