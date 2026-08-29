"""Provider contracts for the A/B POC.

The contracts are deliberately small.  A participant provider receives the
state it is allowed to use for one turn, while an evaluator receives only the
onsite event sequence for one pair.  Keeping these two boundaries explicit
makes a future LLM implementation replaceable without changing the engine.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .domain import (
    Event,
    DialogueTurn,
    Pair,
    PairState,
    Participant,
    ParticipantTurn,
    OnlineMemory,
    RelationshipUpdate,
    ResponseKind,
    RuleConfig,
)


@dataclass(frozen=True)
class ConversationUtterance:
    """One utterance in an optional bounded conversation episode.

    ``sequence`` is deliberately an integer rather than a wall-clock value.
    The engine rewrites it to the run-global sequence when serialising a
    result, so replay and rule-mode determinism do not depend on a clock.
    """

    sequence: int
    speaker: str
    text: str

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError("conversation utterance sequence must be positive")
        if type(self.speaker) is not str or not self.speaker.strip():
            raise ValueError("conversation utterance speaker is required")
        if type(self.text) is not str or not self.text.strip():
            raise ValueError("conversation utterance text is required")


@dataclass(frozen=True)
class ConversationEpisode:
    """A bounded optional episode attached to one participant turn."""

    utterances: tuple[ConversationUtterance, ...]
    ended: bool = True
    reaction: str = "neutral"
    end_reason: str = "responder_end"

    def __post_init__(self) -> None:
        if not isinstance(self.utterances, tuple):
            object.__setattr__(self, "utterances", tuple(self.utterances))
        if not 2 <= len(self.utterances) <= 4:
            raise ValueError("conversation episode must contain 2 to 4 utterances")
        if any(not isinstance(item, ConversationUtterance) for item in self.utterances):
            raise ValueError("conversation episode utterances must be ConversationUtterance values")
        expected = tuple(range(1, len(self.utterances) + 1))
        sequences = tuple(item.sequence for item in self.utterances)
        if sequences != expected:
            raise ValueError("conversation episode utterances must be numbered 1..N")
        if type(self.ended) is not bool:
            raise ValueError("conversation episode ended must be a boolean")
        if type(self.reaction) is not str or self.reaction not in {"positive", "neutral", "misaligned"}:
            raise ValueError("conversation episode reaction is invalid")
        if type(self.end_reason) is not str or self.end_reason not in {
            "responder_end", "initiator_end", "utterance_limit", "provider_unavailable"
        }:
            raise ValueError("conversation episode end_reason is invalid")


# The shorter name is useful to provider implementations and keeps the DTO
# discoverable for callers that describe records as episode utterances.
EpisodeUtterance = ConversationUtterance
EPISODE_END_REASONS = frozenset({
    "responder_end", "initiator_end", "utterance_limit", "provider_unavailable"
})


@dataclass(frozen=True)
class VerbalEpisodeEvaluation:
    """Presentation-only assessment of one completed conversation episode.

    This DTO intentionally contains no relationship state or wall-clock data.
    ``evidence_sequences`` points back to the integer utterance sequence in
    the episode transcript, making the result auditable and replay-stable.
    """

    relationship_movement: int
    end_momentum: int
    reason: str
    evidence_sequences: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.relationship_movement) is not int or self.relationship_movement not in {-1, 0, 1}:
            raise ValueError("verbal relationship_movement must be -1, 0, or 1")
        if type(self.end_momentum) is not int or not 0 <= self.end_momentum <= 3:
            raise ValueError("verbal end_momentum must be an integer from 0 to 3")
        if type(self.reason) is not str or not self.reason.strip():
            raise ValueError("verbal evaluation reason is required")
        if not isinstance(self.evidence_sequences, tuple):
            object.__setattr__(self, "evidence_sequences", tuple(self.evidence_sequences))
        if not self.evidence_sequences:
            raise ValueError("verbal evidence_sequences must not be empty")
        if any(type(sequence) is not int or sequence < 1 for sequence in self.evidence_sequences):
            raise ValueError("verbal evidence_sequences must contain positive integers")
        if tuple(dict.fromkeys(self.evidence_sequences)) != self.evidence_sequences:
            raise ValueError("verbal evidence_sequences must be unique and ordered")


@runtime_checkable
class VerbalEpisodeEvaluator(Protocol):
    """Evaluate only the transcript and reaction of a completed episode.

    Implementations may later call a cloud LLM, but the engine does not know
    or depend on that implementation detail.
    """

    name: str
    version: str

    def evaluate(self, episode: ConversationEpisode) -> VerbalEpisodeEvaluation:
        """Return a verbal assessment without mutating simulation state."""


@dataclass(frozen=True)
class Initiation:
    """The part of a participant turn owned by the talk initiator.

    Keeping the target, topic, and wording together makes it impossible for
    a response provider to silently choose a second target or topic.  The
    DTO is intentionally independent from ``ParticipantTurn``: response is
    supplied only after this initiation has been produced.
    """

    target_id: str
    topic: str
    approach: str

    def __post_init__(self) -> None:
        for field_name in ("target_id", "topic", "approach"):
            value = getattr(self, field_name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(
                    f"initiation {field_name} must be a non-empty trimmed string"
                )


_REACTION_LABELS = {
    "positive": ResponseKind.POSITIVE,
    "neutral": ResponseKind.NEUTRAL,
    "misaligned": ResponseKind.MISALIGNED,
}
_REACTION_NAMES = {value: key for key, value in _REACTION_LABELS.items()}


@dataclass(frozen=True)
class ResponseOutput:
    """Response-provider output kept separate from ``ParticipantTurn``."""

    reply: str
    reaction: str
    episode: ConversationEpisode | None = None
    continue_conversation: bool | None = None

    def __post_init__(self) -> None:
        if type(self.reply) is not str or not self.reply.strip():
            raise ValueError("response reply must be a non-empty string")
        reaction = self.reaction
        if isinstance(reaction, ResponseKind):
            label = _REACTION_NAMES[reaction]
        elif type(reaction) is str:
            label = reaction.strip().lower()
            if label not in _REACTION_LABELS:
                try:
                    label = _REACTION_NAMES[ResponseKind(reaction)]
                except (KeyError, ValueError) as exc:
                    raise ValueError(
                        "response reaction must be positive, neutral, or misaligned"
                    ) from exc
        else:
            raise ValueError(
                "response reaction must be positive, neutral, or misaligned"
            )
        object.__setattr__(self, "reaction", label)
        if self.episode is not None and not isinstance(self.episode, ConversationEpisode):
            raise ValueError("response episode must be a ConversationEpisode")
        if self.continue_conversation is not None and type(self.continue_conversation) is not bool:
            raise ValueError("response continue_conversation must be a boolean or None")

    @property
    def kind(self) -> ResponseKind:
        """Return the legacy enum consumed by ``ParticipantTurn``/``Event``."""

        return _REACTION_LABELS[self.reaction]


@dataclass(frozen=True)
class ResponseContext:
    """Narrow private context owned by one response recipient.

    This boundary intentionally contains one pair's observable history, the
    recipient's matching online memory, and only the recipient -> speaker
    affect value.  It is separate from ``ParticipantContext`` so a response
    provider cannot accidentally receive all candidates or all run state.
    """

    responder: Participant
    initiator: Participant
    pair_events: tuple[Event, ...] = ()
    online_memory: OnlineMemory | None = None
    affect: int = 0
    turn: int = 1
    # Participant-only exchange history; Event remains the blind evaluator
    # input and is retained separately for compatibility.
    dialogue_history: tuple[DialogueTurn, ...] = ()
    # Public recent exchange history.  This is intentionally separate from
    # the pair-only history above so autonomous responders can use what was
    # said in the room without receiving another participant's persona.
    public_dialogue_history: tuple[DialogueTurn, ...] = ()
    # Optional in-memory redaction tokens for public logs. This keeps names
    # and IDs of third parties out of an autonomous provider prompt while
    # remaining source-compatible with older constructors.
    public_participant_tokens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.responder, Participant) or not isinstance(self.initiator, Participant):
            raise ValueError("response context must contain responder and initiator participants")
        if self.responder.id == self.initiator.id:
            raise ValueError("response context responder and initiator must differ")
        if type(self.turn) is not int or self.turn < 1:
            raise ValueError("response context turn must be positive")
        if not isinstance(self.pair_events, tuple):
            object.__setattr__(self, "pair_events", tuple(self.pair_events))
        pair = (self.responder.id, self.initiator.id)
        for event in self.pair_events:
            if not isinstance(event, Event):
                raise ValueError("response context pair_events must contain Event values")
            if {event.actor, event.target} != set(pair):
                raise ValueError("response context pair_events must belong to the responder/initiator pair")
        if self.online_memory is not None:
            if not isinstance(self.online_memory, OnlineMemory):
                raise ValueError("response context online_memory must be an OnlineMemory")
            if self.online_memory.counterpart_id != self.initiator.id and not (
                self.online_memory.counterpart_id is None
                and self.responder.id in self.online_memory.participant_ids
                and self.initiator.id in self.online_memory.participant_ids
            ):
                raise ValueError("response context online_memory must belong to the target pair")
        if type(self.affect) is not int or not -2 <= self.affect <= 2:
            raise ValueError("response context affect must be an integer from -2 to +2")
        if not isinstance(self.dialogue_history, tuple):
            object.__setattr__(self, "dialogue_history", tuple(self.dialogue_history))
        for dialogue in self.dialogue_history:
            if not isinstance(dialogue, DialogueTurn):
                raise ValueError("response context dialogue_history must contain DialogueTurn values")
            if {dialogue.actor, dialogue.target} != set(pair):
                raise ValueError("response context dialogue_history must belong to the responder/initiator pair")
        if not isinstance(self.public_dialogue_history, tuple):
            object.__setattr__(self, "public_dialogue_history", tuple(self.public_dialogue_history))
        for dialogue in self.public_dialogue_history:
            if not isinstance(dialogue, DialogueTurn):
                raise ValueError("response context public_dialogue_history must contain DialogueTurn values")
        if not isinstance(self.public_participant_tokens, tuple):
            object.__setattr__(self, "public_participant_tokens", tuple(self.public_participant_tokens))
        if any(type(token) is not str or not token for token in self.public_participant_tokens):
            raise ValueError("response context public_participant_tokens must contain non-empty strings")


def adapt_response(value: Any) -> tuple[ResponseKind, str | None, ResponseOutput | None]:
    """Normalize the M10 DTO while retaining legacy reaction-only providers."""

    if isinstance(value, ResponseOutput):
        return value.kind, value.reply, value
    if isinstance(value, ResponseKind):
        return value, None, None
    if isinstance(value, Mapping):
        if "reply" not in value or "reaction" not in value:
            raise ValueError("response mapping must contain reply and reaction")
        output = ResponseOutput(value["reply"], value["reaction"])
        return output.kind, output.reply, output
    if type(value) is str:
        # Keep the old response-only adapter useful for providers that return
        # the English reaction label directly.
        label = value.strip().lower()
        kind = _REACTION_LABELS.get(label)
        if kind is None:
            try:
                kind = ResponseKind(value)
            except ValueError as exc:
                raise ValueError(
                    "response reaction must be positive, neutral, or misaligned"
                ) from exc
        return kind, None, None
    raise ValueError("response provider must return ResponseOutput or ResponseKind")


@dataclass(frozen=True)
class ParticipantContext:
    """Information available to a participant provider for one turn.

    ``pair_states`` may contain the online layer because participant choice
    intentionally models the comparison condition.  ``events`` is limited
    to events already produced in the current run.  The evaluator does not
    consume this DTO and therefore cannot accidentally receive that layer.
    ``rng`` is a run-owned random stream; providers must not use global random
    state.
    """

    turn: int
    actor: Participant
    candidates: tuple[Participant, ...]
    pair_states: Mapping[Pair, PairState]
    events: tuple[Event, ...]
    last_target: str | None
    rules: RuleConfig
    rng: random.Random
    # This is intentionally a tuple owned by the current actor, rather than
    # a condition-wide map.  It cannot expose another participant's memory.
    online_memories: tuple[OnlineMemory, ...] = ()
    # A narrow, read-only projection of the current actor's private affect.
    # Keys are candidate IDs and values are the actor -> candidate values;
    # no reverse or third-party direction is available at this boundary.
    outgoing_affect: Mapping[str, int] = field(default_factory=dict)
    # Short actor-visible exchange history.  The default keeps older direct
    # ParticipantContext constructors source-compatible.
    dialogue_history: tuple[DialogueTurn, ...] = ()

    def __post_init__(self) -> None:
        if self.turn < 1:
            raise ValueError("participant context turn must be positive")
        if not isinstance(self.outgoing_affect, Mapping):
            raise ValueError("outgoing_affect must be a mapping of candidate IDs to integers")
        else:
            affect = dict(self.outgoing_affect)
        candidate_ids = tuple(candidate.id for candidate in self.candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("participant candidates must have unique IDs")
        if affect and set(affect) != set(candidate_ids):
            raise ValueError("outgoing_affect must contain exactly one value per candidate")
        if any(type(candidate_id) is not str or not candidate_id for candidate_id in affect):
            raise ValueError("outgoing_affect keys must be candidate IDs")
        if any(type(value) is not int or value < -2 or value > 2 for value in affect.values()):
            raise ValueError("outgoing_affect values must be integers from -2 to +2")
        object.__setattr__(self, "outgoing_affect", MappingProxyType(affect))
        if not isinstance(self.dialogue_history, tuple):
            object.__setattr__(self, "dialogue_history", tuple(self.dialogue_history))
        if any(not isinstance(item, DialogueTurn) for item in self.dialogue_history):
            raise ValueError("participant dialogue_history must contain DialogueTurn values")

    @property
    def private_online_projection(self) -> tuple[dict[str, Any], ...]:
        """Safe, actor-owned projection for prompt builders and audits."""

        return tuple(
            {
                "experience_id": memory.experience_id,
                "experience_version": memory.experience_version,
                "kind": memory.kind,
                "counterpart_id": memory.counterpart_id,
                "participant_ids": list(memory.participant_ids),
                "topic": memory.topic,
                "exchange_count": memory.exchange_count,
                "summary": memory.summary,
            }
            for memory in self.online_memories
        )


@runtime_checkable
class ParticipantProvider(Protocol):
    """Choose and render one participant action."""

    name: str
    version: str

    def propose(self, context: ParticipantContext) -> ParticipantTurn:
        """Return the next action using only the supplied context."""


@runtime_checkable
class TalkInitiator(Protocol):
    """Choose who to approach and compose the initial approach."""

    name: str
    version: str

    def initiate(self, context: ParticipantContext) -> Initiation:
        """Return target, topic, and approach for one turn."""


@runtime_checkable
class EpisodeInitiator(Protocol):
    """Optional actor-owned continuation boundary for an episode."""

    def continue_episode(
        self,
        context: ParticipantContext,
        initiation: Initiation,
        episode: ConversationEpisode,
    ) -> tuple[str, bool]:
        """Return the actor's next utterance and whether to continue."""


@runtime_checkable
class EpisodeResponder(Protocol):
    """Optional target-owned continuation boundary for an episode."""

    def continue_episode(
        self,
        context: ResponseContext,
        initiation: Initiation,
        episode: ConversationEpisode,
    ) -> tuple[str, bool]:
        """Return the target's next utterance and whether to continue."""


@runtime_checkable
class ResponseProvider(Protocol):
    """Provide the target's response to one already-composed initiation."""

    name: str
    version: str

    def respond(
        self, context: ParticipantContext | ResponseContext, initiation: Initiation
    ) -> ResponseOutput | ResponseKind:
        """Return a reply/reaction DTO, or a legacy ``ResponseKind``."""


@runtime_checkable
class RelationshipEvaluator(Protocol):
    """Infer onsite relationship from a blind event sequence."""

    name: str
    version: str

    def evaluate(self, events: Sequence[Event]) -> RelationshipUpdate:
        """Recompute the pair state from onsite events only."""


__all__ = [
    "ConversationUtterance",
    "EpisodeUtterance",
    "ConversationEpisode",
    "VerbalEpisodeEvaluation",
    "EPISODE_END_REASONS",
    "Initiation",
    "DialogueTurn",
    "ResponseOutput",
    "ResponseContext",
    "ParticipantContext",
    "ParticipantProvider",
    "RelationshipEvaluator",
    "ResponseProvider",
    "TalkInitiator",
    "EpisodeInitiator",
    "EpisodeResponder",
    "VerbalEpisodeEvaluator",
]
