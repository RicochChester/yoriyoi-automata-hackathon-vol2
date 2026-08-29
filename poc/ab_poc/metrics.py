"""Metrics for the deterministic A/B relationship POC.

The functions in this module deliberately use the final onsite relationship
layer only.  An online direct tie is an eligibility/stratification field, not
an onsite relationship edge.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .domain import (
    OnsiteRelationship,
    Pair,
    PairState,
    Participant,
    canonical_pair,
)


_RANK = {
    OnsiteRelationship.NOT_FORMED: 0,
    OnsiteRelationship.ACQUAINTANCE: 1,
    OnsiteRelationship.FAMILIAR: 2,
}
_KNOWN_PAIR = canonical_pair("akane", "midori")


def _turn_target(turn: Any) -> str | None:
    """Read a target from an Event/turn log without coupling to the engine."""

    if isinstance(turn, Mapping):
        value = turn.get("target")
    else:
        value = getattr(turn, "target", None)
    return value if isinstance(value, str) else None


def _stage(value: Any) -> OnsiteRelationship:
    if isinstance(value, OnsiteRelationship):
        return value
    return OnsiteRelationship(value)


def _pair(value: Any) -> Pair:
    if isinstance(value, PairState):
        return value.pair
    if isinstance(value, Mapping):
        return canonical_pair(str(value["left_id"]), str(value["right_id"]))
    left, right = value
    return canonical_pair(str(left), str(right))


def _state_values(final_pairs: Mapping[Pair, PairState] | Sequence[PairState] | Sequence[Mapping[str, Any]]) -> dict[Pair, Any]:
    if isinstance(final_pairs, Mapping):
        return {_pair(key): value for key, value in final_pairs.items()}
    return {_pair(value): value for value in final_pairs}


def _state_stage(state: Any) -> OnsiteRelationship:
    if state is None:
        return OnsiteRelationship.NOT_FORMED
    if isinstance(state, PairState):
        return state.onsite_relationship
    return _stage(state["onsite_relationship"])


def _participants_ids(participants: Sequence[Participant] | Sequence[str]) -> tuple[str, ...]:
    return tuple(item.id if isinstance(item, Participant) else str(item) for item in participants)


def calculate_metrics(
    *,
    condition_id: str,
    final_pairs: Mapping[Pair, PairState] | Sequence[PairState] | Sequence[Mapping[str, Any]],
    participants: Sequence[Participant] | Sequence[str] = ("akane", "koharu", "midori", "kurumi"),
    online_known_pairs: Sequence[Pair] = (),
    turns: Sequence[Any] = (),
) -> dict[str, Any]:
    """Calculate relationship metrics and target-reception observations."""

    states = _state_values(final_pairs)
    people = _participants_ids(participants)
    online = {canonical_pair(*pair) for pair in online_known_pairs}
    eligible_pairs = tuple(
        canonical_pair(left, right)
        for index, left in enumerate(people)
        for right in people[index + 1 :]
        if canonical_pair(left, right) not in online
    )
    new_acquaintance_count = sum(
        _RANK[_state_stage(states.get(pair))] >= _RANK[OnsiteRelationship.ACQUAINTANCE]
        for pair in eligible_pairs
    )
    new_familiar_count = sum(
        _state_stage(states.get(pair)) is OnsiteRelationship.FAMILIAR for pair in eligible_pairs
    )
    eligible_count = len(eligible_pairs)

    onsite_edges = {
        pair
        for pair, state in states.items()
        if _RANK[_state_stage(state)] >= _RANK[OnsiteRelationship.ACQUAINTANCE]
    }
    degree = {person: 0 for person in people}
    for left, right in onsite_edges:
        if left in degree:
            degree[left] += 1
        if right in degree:
            degree[right] += 1
    isolated = [person for person in people if degree[person] == 0]

    # ``known_pair_onsite`` is meaningful only when the configured online
    # experience actually names the two-person reference pair.  In
    # particular, group conditions intentionally emit null rather than a
    # misleading/dummy pair.
    if _KNOWN_PAIR not in online:
        known_pair_onsite: Any = None
        paths: Any = None
    else:
        known_state = states.get(_KNOWN_PAIR)
        known_pair_onsite = {
            "pair": list(_KNOWN_PAIR),
            "stage": _state_stage(known_state).value if known_state is not None else OnsiteRelationship.NOT_FORMED.value,
        }
        paths = []
        for endpoint in _KNOWN_PAIR:
            for third_party in people:
                if third_party in _KNOWN_PAIR:
                    continue
                edge = canonical_pair(endpoint, third_party)
                if edge in onsite_edges:
                    paths.append(
                        {
                            "from_known_endpoint": endpoint,
                            "third_party": third_party,
                            "pair": list(edge),
                            "stage": _state_stage(states.get(edge)).value,
                        }
                    )

    target_counts = {person: 0 for person in people}
    for turn in turns:
        target = _turn_target(turn)
        if target in target_counts:
            target_counts[target] += 1
    total_conversations = sum(target_counts.values())
    max_target_receive_count = max(target_counts.values(), default=0)
    most_targeted_participants = (
        [person for person in people if target_counts[person] == max_target_receive_count]
        if max_target_receive_count > 0 else []
    )

    result = {
        "new_acquaintance_count": int(new_acquaintance_count),
        "new_acquaintance_eligible_count": eligible_count,
        "new_acquaintance_rate": (new_acquaintance_count / eligible_count if eligible_count else 0.0),
        "new_familiar_count": int(new_familiar_count),
        "known_pair_onsite": known_pair_onsite,
        "isolated_participants": isolated,
        "third_party_observation_paths": paths,
    }
    # These are M11 observation fields.  Keep the established A/B result
    # schema byte-for-byte compatible for existing replay fixtures.
    if condition_id in {"N0", "P2", "P3", "G2", "G3"}:
        result.update({
            "conversation_concentration": (
                max_target_receive_count / total_conversations
                if total_conversations else 0.0
            ),
            "most_targeted_participants": most_targeted_participants,
            "max_target_receive_count": int(max_target_receive_count),
            "targeted_participant_count": sum(value > 0 for value in target_counts.values()),
        })
    return result


def compute_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    """Convenience adapter for a completed JSON run result."""

    condition = result.get("condition", {})
    if isinstance(condition, str):
        condition_id = condition
        online = result.get("online_known_pairs", ())
    else:
        condition_id = str(condition.get("id", result.get("condition_id", "A")))
        online = condition.get("online_known_pairs", ())
    snapshots = result.get("snapshots", ())
    final = next((item for item in reversed(snapshots) if item.get("phase") == "end"), None)
    if final is None and snapshots:
        final = snapshots[-1]
    pairs = {} if final is None else final.get("pairs", {})
    participants = result.get("participants", ("akane", "koharu", "midori", "kurumi"))
    if participants and isinstance(participants[0], Mapping):
        participants = [item["id"] for item in participants]
    turns = result.get("turns", ())
    return calculate_metrics(
        condition_id=condition_id,
        final_pairs=pairs,
        participants=participants,
        online_known_pairs=online,
        turns=turns if isinstance(turns, Sequence) and not isinstance(turns, (str, bytes)) else (),
    )


# A descriptive alias is useful to callers that do not know the internal
# naming convention, while retaining one implementation and one output shape.
metrics_for_result = compute_metrics


__all__ = ["calculate_metrics", "compute_metrics", "metrics_for_result"]
