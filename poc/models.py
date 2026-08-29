from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


ActionName = Literal["talk", "continue", "step_back"]


@dataclass(frozen=True)
class Persona:
    id: str
    name: str
    interests: tuple[str, ...]
    visit_purpose: str
    approachability: int
    receptivity: int
    conversation_style: str
    outfit_color: str

    def public_dict(self) -> dict:
        value = asdict(self)
        value["interests"] = list(self.interests)
        return value


@dataclass
class ParticipantState:
    participant_id: str
    position: tuple[int, int]
    arrived: bool = False
    last_partner_id: str | None = None


@dataclass
class Relationship:
    left_id: str
    right_id: str
    known_before_visit: bool = False
    channel: str = "onsite"
    talk_count: int = 0
    continuation_count: int = 0
    strength: int = 0
    # ``known_before_visit`` remains the legacy field.  These explicit fields
    # let consumers distinguish an online tie that was activated onsite from
    # a relationship formed onsite during this run.
    pre_existing_online_connection: bool = False
    onsite_talk_count: int = 0
    onsite_relationship_formed: bool = False

    def key(self) -> tuple[str, str]:
        return tuple(sorted((self.left_id, self.right_id)))

    def public_dict(self) -> dict:
        return {
            "source": self.left_id,
            "target": self.right_id,
            "known_before_visit": self.known_before_visit,
            "channel": self.channel,
            "talk_count": self.talk_count,
            "continuation_count": self.continuation_count,
            "strength": self.strength,
            "pre_existing_online_connection": (
                self.pre_existing_online_connection or self.known_before_visit
            ),
            "onsite_talk_count": self.onsite_talk_count,
            "onsite_relationship_formed": self.onsite_relationship_formed,
            "online_to_onsite_activation": (
                (self.pre_existing_online_connection or self.known_before_visit)
                and self.onsite_talk_count > 0
            ),
        }


@dataclass(frozen=True)
class Scenario:
    id: str
    label: str
    venue_id: str
    venue_label: str
    venue_description: str
    relationship_mode_id: str
    relationship_mode_label: str
    arrival_steps: tuple[int, ...]
    conversation_bonus: int
    conversation_start_step: int
    shared_interest_bonus: int
    unrelated_pair_penalty: int
    venue_topic_is_primary: bool
    venue_topic: str
    known_pairs: tuple[tuple[str, str], ...]
    online_community: dict | None = None

    def public_dict(self) -> dict:
        value = asdict(self)
        value["arrival_steps"] = list(self.arrival_steps)
        value["known_pairs"] = [list(pair) for pair in self.known_pairs]
        if self.online_community is not None:
            value["online_community"] = self.online_community
        else:
            value.pop("online_community", None)
        return value


@dataclass(frozen=True)
class ExperimentPreset:
    """A selectable simulation size, kept separate from venue scenarios."""

    id: str
    label: str
    description: str
    participant_ids: tuple[str, ...]
    steps: int
    arrival_steps_by_venue: dict[str, tuple[int, ...]]

    def public_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "participant_ids": list(self.participant_ids),
            "participant_count": len(self.participant_ids),
            "steps": self.steps,
        }


@dataclass(frozen=True)
class Candidate:
    participant_id: str
    name: str
    score: int
    shared_interests: tuple[str, ...]
    relation_strength: int
    interests: tuple[str, ...]
    visit_purpose: str
    conversation_style: str


@dataclass(frozen=True)
class DecisionContext:
    step: int
    actor: Persona
    candidates: tuple[Candidate, ...]
    scenario: Scenario


@dataclass(frozen=True)
class Decision:
    action: ActionName
    target_id: str | None
    reason: str


@dataclass(frozen=True)
class Conversation:
    topic: str
    speeches: tuple[dict[str, str], ...]


@dataclass
class TurnRecord:
    step: int
    actor_id: str
    decision: Decision
    conversation: Conversation | None
    positions: dict[str, tuple[int, int]]
    relations: list[dict]
    metrics: dict
    arrivals: list[str] = field(default_factory=list)

    def public_dict(self) -> dict:
        return {
            "step": self.step,
            "actor_id": self.actor_id,
            "decision": asdict(self.decision),
            "conversation": asdict(self.conversation) if self.conversation else None,
            "positions": {key: list(value) for key, value in self.positions.items()},
            "relations": self.relations,
            "metrics": self.metrics,
            "arrivals": self.arrivals,
        }
