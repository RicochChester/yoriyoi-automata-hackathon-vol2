"""Immutable domain types for the four-person online-community POC.

The types in this module intentionally keep the three relationship layers
separate: an online direct tie, an onsite contact history, and the evaluated
onsite relationship stage.  The simulation engine can therefore replace a
provider without changing the state it records.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Literal, Mapping, TypeVar


ParticipantId = Literal["akane", "koharu", "midori", "kurumi"]
Approach = str
Pair = tuple[str, str]


class OnlineKnown(str, Enum):
    """Compatibility projection for whether a pair was directly known online.

    The M10.3 source of truth is ``OnlineExperience``.  This coarse enum is
    retained for existing providers and metrics; it deliberately carries no
    topic, wording, or private memory content.
    """

    NONE = "なし"
    DIRECT = "あり"
    # Readable aliases for providers that prefer explicit terminology.
    NOT_KNOWN = "なし"
    KNOWN = "あり"


class ContactHistory(str, Enum):
    """The observed onsite contact state for a pair."""

    NONE = "未接触"
    CONVERSATION = "会話経験あり"
    UNTOUCHED = "未接触"
    EXPERIENCED = "会話経験あり"


class OnsiteRelationship(str, Enum):
    """The relationship stage inferred from onsite logs only."""

    NOT_FORMED = "未形成"
    ACQUAINTANCE = "顔見知り"
    FAMILIAR = "親しみがある"


class Stance(str, Enum):
    """A participant's interpersonal stance."""

    PROACTIVE = "積極的"
    NEUTRAL = "ふつう"
    CAUTIOUS = "慎重"


class ResponseKind(str, Enum):
    POSITIVE = "前向き"
    NEUTRAL = "ふつう"
    MISALIGNED = "噛み合わない"


def canonical_pair(left_id: str, right_id: str) -> Pair:
    """Return a stable, unordered pair key and reject self-pairs."""

    if left_id == right_id:
        raise ValueError("a pair cannot contain the same participant twice")
    return tuple(sorted((left_id, right_id)))  # type: ignore[return-value]


@dataclass(frozen=True)
class Participant:
    id: str
    name: str
    interests: tuple[str, ...]
    stance: Stance
    # Optional first-phase persona card.  Defaults keep direct constructors
    # from older callers/tests source-compatible.
    purpose: str = ""
    conversation_style: str = ""
    opens_up_when: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.name:
            raise ValueError("participant id and name are required")
        if not self.interests or any(not interest for interest in self.interests):
            raise ValueError("participant interests must be non-empty")
        if not isinstance(self.stance, Stance):
            object.__setattr__(self, "stance", Stance(self.stance))

    @property
    def persona(self) -> dict[str, str]:
        """Small prompt-safe persona projection for participant providers."""

        return {
            "purpose": self.purpose,
            "conversation_style": self.conversation_style,
            "opens_up_when": self.opens_up_when,
        }


@dataclass(frozen=True)
class OnlineMemory:
    """The actor-private, minimal memory of one prior online experience."""

    experience_id: str
    experience_version: str
    kind: str
    counterpart_id: str | None
    topic: str | None
    exchange_count: int
    summary: str
    participant_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.experience_id or not self.experience_version or not self.kind:
            raise ValueError("online memory identity and kind are required")
        if self.exchange_count < 1 or not self.summary:
            raise ValueError("online memory must identify an exchange")
        participant_ids = tuple(sorted(self.participant_ids))
        if len(set(participant_ids)) != len(participant_ids):
            raise ValueError("online memory participants must be unique")
        if self.counterpart_id is not None and not self.counterpart_id:
            raise ValueError("online memory counterpart_id must be non-empty or null")
        if self.counterpart_id is None and len(participant_ids) < 3:
            raise ValueError("group online memory must identify at least three participants")
        if self.counterpart_id is not None and participant_ids:
            raise ValueError("pair online memory cannot include group participants")
        object.__setattr__(self, "participant_ids", participant_ids)

    def applies_to(self, participant_id: str) -> bool:
        """Whether this private memory is relevant to a participant."""

        return self.counterpart_id == participant_id or participant_id in self.participant_ids


@dataclass(frozen=True)
class OnlineExperience:
    """Versioned scenario data describing an online prior experience.

    Only named participants receive the derived ``OnlineMemory``. A group
    experience remains one group event; it is not expanded into pairwise
    exchanges. The experience stores a safe summary rather than invented
    message text.
    """

    id: str
    version: str
    kind: str
    participants: tuple[str, ...] = ()
    topic: str | None = None
    exchange_count: int = 0
    memory_summary: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.version or self.kind not in {
            "none", "direct_exchange", "pair_collaborative_task", "group_thematic_exchange", "group_collaborative_task"
        }:
            raise ValueError("online experience id, version, and kind are invalid")
        participants = tuple(sorted(self.participants))
        if len(set(participants)) != len(participants):
            raise ValueError("online experience participants must be unique")
        if self.kind == "none":
            if participants or self.topic is not None or self.exchange_count != 0:
                raise ValueError("none online experience must have no exchange data")
        elif self.kind in {"direct_exchange", "pair_collaborative_task"}:
            if len(participants) != 2 or not self.topic or self.exchange_count < 1:
                raise ValueError("direct online experience requires a pair, topic, and exchange")
        else:
            if len(participants) < 3 or not self.topic or self.exchange_count < 1:
                raise ValueError("group online experience requires participants, topic, and exchange")
        if not self.memory_summary:
            raise ValueError("online experience memory_summary is required")
        object.__setattr__(self, "participants", participants)

    def memory_for(self, actor_id: str) -> tuple[OnlineMemory, ...]:
        """Return only the memory owned by ``actor_id`` (never a global map)."""

        if self.kind == "none" or actor_id not in self.participants:
            return ()
        if self.kind in {"direct_exchange", "pair_collaborative_task"}:
            counterpart = next(item for item in self.participants if item != actor_id)
            return (OnlineMemory(
                self.id, self.version, self.kind, counterpart, self.topic,
                self.exchange_count, self.memory_summary,
            ),)
        return (OnlineMemory(
            self.id, self.version, self.kind, None, self.topic,
            self.exchange_count, self.memory_summary, self.participants,
        ),)

    def memory_for_pair(self, actor_id: str, target_id: str) -> OnlineMemory | None:
        """Return a private memory visible to both sides of one response."""

        if self.kind in {"direct_exchange", "pair_collaborative_task"}:
            if actor_id not in self.participants or target_id not in self.participants:
                return None
            return self.memory_for(target_id)[0]
        if self.kind in {"group_thematic_exchange", "group_collaborative_task"}:
            if actor_id in self.participants and target_id in self.participants:
                return self.memory_for(target_id)[0]
        return None


@dataclass(frozen=True)
class Condition:
    id: str
    label: str
    online_known_pairs: tuple[Pair, ...]
    online_experience: OnlineExperience | None = None

    def __post_init__(self) -> None:
        pairs = tuple(canonical_pair(*pair) for pair in self.online_known_pairs)
        if len(set(pairs)) != len(pairs):
            raise ValueError("online known pairs must be unique")
        object.__setattr__(self, "online_known_pairs", pairs)

    def online_known(self, left_id: str, right_id: str) -> OnlineKnown:
        return (
            OnlineKnown.DIRECT
            if canonical_pair(left_id, right_id) in self.online_known_pairs
            else OnlineKnown.NONE
        )


@dataclass(frozen=True)
class SelectionWeights:
    base: float = 1.0
    online_known_bonus: float = 1.0
    shared_interest_bonus: float = 1.0
    prior_positive_bonus: float = 0.5
    repeated_target_multiplier: float = 0.5


@dataclass(frozen=True)
class ReactionPoints:
    base: int = 1
    online_known_bonus: int = 2
    topic_match_bonus: int = 2
    prior_positive_bonus: int = 1
    proactive_actor_bonus: int = 1
    cautious_actor_penalty: int = -1
    cautious_target_penalty: int = -1
    jitter_values: tuple[int, ...] = (-1, 0, 1)
    positive_threshold: int = 3
    neutral_threshold: int = 1


@dataclass(frozen=True)
class RelationshipThresholds:
    acquaintance_conversations: int = 2
    acquaintance_positive: int = 1
    familiar_conversations: int = 3
    familiar_positive: int = 2
    familiar_requires_bidirectional: bool = True


@dataclass(frozen=True)
class RuleConfig:
    turns: int
    common_topic: str
    selection: SelectionWeights
    reaction: ReactionPoints
    relationship: RelationshipThresholds


@dataclass(frozen=True)
class PocConfig:
    schema_version: str
    config_version: str
    participants: tuple[Participant, ...]
    conditions: tuple[Condition, ...]
    rules: RuleConfig
    scenario_version: str = "m10.3.scenario.v1"


@dataclass(frozen=True)
class Event:
    """The deliberately blind evaluation input for one onsite interaction."""

    turn: int
    actor: str
    target: str
    topic: str
    approach: Approach
    response: ResponseKind

    def __post_init__(self) -> None:
        if self.turn < 1:
            raise ValueError("turn must be positive")
        if self.actor == self.target:
            raise ValueError("event actor and target must differ")
        if not self.topic or not self.approach:
            raise ValueError("event topic and approach are required")
        if not isinstance(self.response, ResponseKind):
            object.__setattr__(self, "response", ResponseKind(self.response))


@dataclass(frozen=True)
class PairState:
    """All state for one pair, with online and onsite layers separate."""

    left_id: str
    right_id: str
    online_known: OnlineKnown = OnlineKnown.NONE
    contact_history: ContactHistory = ContactHistory.NONE
    onsite_relationship: OnsiteRelationship = OnsiteRelationship.NOT_FORMED
    conversation_count: int = 0
    positive_count: int = 0
    actor_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        pair = canonical_pair(self.left_id, self.right_id)
        object.__setattr__(self, "left_id", pair[0])
        object.__setattr__(self, "right_id", pair[1])
        for field_name, enum_type in (
            ("online_known", OnlineKnown),
            ("contact_history", ContactHistory),
            ("onsite_relationship", OnsiteRelationship),
        ):
            value = getattr(self, field_name)
            if not isinstance(value, enum_type):
                object.__setattr__(self, field_name, enum_type(value))
        if self.conversation_count < 0 or self.positive_count < 0:
            raise ValueError("pair counters cannot be negative")
        if self.positive_count > self.conversation_count:
            raise ValueError("positive_count cannot exceed conversation_count")

    @property
    def pair(self) -> Pair:
        return self.left_id, self.right_id


@dataclass(frozen=True)
class RelationshipUpdate:
    pair: PairState
    previous_relationship: OnsiteRelationship
    relationship: OnsiteRelationship
    reason: str


@dataclass(frozen=True)
class ParticipantTurn:
    turn: int
    actor: str
    target: str
    topic: str
    approach: Approach
    response: ResponseKind

    def __post_init__(self) -> None:
        if self.turn < 1:
            raise ValueError("turn must be positive")
        if self.actor == self.target:
            raise ValueError("participant turn actor and target must differ")
        if not self.topic or not self.approach:
            raise ValueError("participant turn topic and approach are required")
        if not isinstance(self.response, ResponseKind):
            object.__setattr__(self, "response", ResponseKind(self.response))


@dataclass(frozen=True)
class DialogueTurn:
    """Participant-only record of one exchange, including the reply text.

    This is deliberately separate from ``Event``: relationship evaluation
    remains blind to generated response wording.
    """

    turn: int
    actor: str
    target: str
    topic: str
    approach: str
    reply: str | None
    reaction: str

    def __post_init__(self) -> None:
        if self.turn < 1:
            raise ValueError("dialogue turn must be positive")
        if not self.actor or not self.target or self.actor == self.target:
            raise ValueError("dialogue actor and target must differ")
        if not self.topic or not self.approach:
            raise ValueError("dialogue topic and approach are required")
        if self.reply is not None and (not isinstance(self.reply, str) or not self.reply.strip()):
            raise ValueError("dialogue reply must be empty or a non-empty string")
        if isinstance(self.reaction, ResponseKind):
            object.__setattr__(self, "reaction", {
                ResponseKind.POSITIVE: "positive",
                ResponseKind.NEUTRAL: "neutral",
                ResponseKind.MISALIGNED: "misaligned",
            }[self.reaction])
        if not isinstance(self.reaction, str) or not self.reaction:
            raise ValueError("dialogue reaction is required")


def update_relationship(
    state: PairState,
    event: Event,
    thresholds: RelationshipThresholds | None = None,
) -> RelationshipUpdate:
    """Apply one onsite event and infer the monotonic onsite stage."""

    thresholds = thresholds or RelationshipThresholds()
    if canonical_pair(event.actor, event.target) != state.pair:
        raise ValueError("event does not belong to pair state")
    count = state.conversation_count + 1
    positive = state.positive_count + (event.response is ResponseKind.POSITIVE)
    actor_ids = tuple(dict.fromkeys((*state.actor_ids, event.actor)))
    candidate = state.onsite_relationship
    rank = {
        OnsiteRelationship.NOT_FORMED: 0,
        OnsiteRelationship.ACQUAINTANCE: 1,
        OnsiteRelationship.FAMILIAR: 2,
    }
    if (
        count >= thresholds.familiar_conversations
        and positive >= thresholds.familiar_positive
        and (
            not thresholds.familiar_requires_bidirectional
            or set(actor_ids) >= set(state.pair)
        )
    ):
        candidate = OnsiteRelationship.FAMILIAR
    elif count >= thresholds.acquaintance_conversations and positive >= thresholds.acquaintance_positive:
        if rank[candidate] < rank[OnsiteRelationship.ACQUAINTANCE]:
            candidate = OnsiteRelationship.ACQUAINTANCE
    # Relationship never regresses: neutral and misaligned contacts still count.
    if rank[candidate] < rank[state.onsite_relationship]:
        candidate = state.onsite_relationship
    reason = (
        f"現地会話{count}回、前向き{positive}回"
        f"（{state.onsite_relationship.value}→{candidate.value}）"
    )
    next_state = PairState(
        *state.pair,
        online_known=state.online_known,
        contact_history=ContactHistory.CONVERSATION,
        onsite_relationship=candidate,
        conversation_count=count,
        positive_count=positive,
        actor_ids=actor_ids,
    )
    return RelationshipUpdate(next_state, state.onsite_relationship, candidate, reason)


_T = TypeVar("_T")


def to_jsonable(value: Any) -> Any:
    """Convert domain dataclasses/enums/tuples to JSON-compatible values."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    return value


__all__ = [
    "Approach", "Condition", "ContactHistory", "DialogueTurn", "Event", "OnlineKnown",
    "OnlineExperience", "OnlineMemory",
    "OnsiteRelationship", "PairState", "Participant", "ParticipantId",
    "ParticipantTurn", "ReactionPoints", "RelationshipThresholds",
    "PocConfig", "RelationshipUpdate", "ResponseKind", "RuleConfig", "SelectionWeights",
    "Stance", "canonical_pair", "to_jsonable", "update_relationship",
]
