"""Deterministic, seedable rule providers for the four-person POC."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

from .contracts import Initiation, ParticipantContext, ResponseOutput, adapt_response
from .domain import (
    ContactHistory,
    Event,
    OnlineKnown,
    OnsiteRelationship,
    Pair,
    PairState,
    Participant,
    ParticipantTurn,
    RelationshipThresholds,
    RelationshipUpdate,
    ResponseKind,
    RuleConfig,
    Stance,
    canonical_pair,
)


def _state_for(
    pair_states: Mapping[Pair, PairState], left: str, right: str
) -> PairState | None:
    """Look up a state while tolerating either orientation of mapping keys."""

    pair = canonical_pair(left, right)
    state = pair_states.get(pair)
    if state is not None:
        return state
    # A mapping supplied by an engine may use PairState keys as its source of
    # truth; accepting a reversed key keeps this provider independent of that
    # storage detail.
    return pair_states.get((pair[1], pair[0]))


def _events_for_pair(events: Sequence[Event], left: str, right: str) -> tuple[Event, ...]:
    pair = canonical_pair(left, right)
    return tuple(event for event in events if canonical_pair(event.actor, event.target) == pair)


def _shared_interests(actor: Participant, target: Participant) -> tuple[str, ...]:
    target_interests = set(target.interests)
    return tuple(interest for interest in actor.interests if interest in target_interests)


def _weighted_choice(
    candidates: Sequence[Participant], weights: Sequence[float], rng: random.Random
) -> Participant:
    if not candidates or len(candidates) != len(weights):
        raise ValueError("candidates and weights must be non-empty and have equal length")
    if any(weight <= 0 for weight in weights):
        raise ValueError("all candidate weights must be positive")
    # A cumulative draw is explicit about being weighted random, rather than
    # silently becoming a max-score policy.  It also behaves consistently for
    # a supplied random.Random instance.
    total = sum(weights)
    needle = rng.random() * total
    cumulative = 0.0
    for candidate, weight in zip(candidates, weights):
        cumulative += weight
        if needle < cumulative:
            return candidate
    return candidates[-1]


class RuleTalkInitiator:
    """Deterministic rule implementation for target/topic/approach choice."""

    name = "rule-talk-initiator"
    version = "ab_poc_rules_v1"

    def initiate(self, context: ParticipantContext) -> Initiation:
        actor = context.actor
        candidates = tuple(context.candidates)
        if not candidates:
            raise ValueError("participant context must contain at least one candidate")
        if any(candidate.id == actor.id for candidate in candidates):
            raise ValueError("participant candidates must not include the actor")

        weights = tuple(self.selection_weight(context, candidate) for candidate in candidates)
        target = _weighted_choice(candidates, weights, context.rng)
        pair_events = _events_for_pair(context.events, actor.id, target.id)
        topic = self._choose_topic(context, target, pair_events)
        approach = self._approach(actor.stance)
        return Initiation(target.id, topic, approach)

    def selection_weight(self, context: ParticipantContext, candidate: Participant) -> float:
        """Return the positive weight for one candidate before drawing."""

        actor = context.actor
        state = _state_for(context.pair_states, actor.id, candidate.id)
        pair_events = _events_for_pair(context.events, actor.id, candidate.id)
        weights = context.rules.selection
        weight = weights.base
        if state is not None and state.online_known is OnlineKnown.DIRECT:
            weight += weights.online_known_bonus
        if _shared_interests(actor, candidate):
            weight += weights.shared_interest_bonus
        if any(event.response is ResponseKind.POSITIVE for event in pair_events):
            weight += weights.prior_positive_bonus
        if context.last_target == candidate.id:
            weight *= weights.repeated_target_multiplier
        if weight <= 0:
            raise ValueError("selection rule produced a non-positive weight")
        return weight

    def _choose_topic(
        self, context: ParticipantContext, target: Participant, pair_events: Sequence[Event]
    ) -> str:
        online_topics = tuple(
            memory.topic
            for memory in context.online_memories
            if memory.applies_to(target.id) and memory.topic
        )
        if online_topics:
            return online_topics[0]
        shared = _shared_interests(context.actor, target)
        if shared:
            options = shared
        else:
            prior_topics = tuple(dict.fromkeys(event.topic for event in pair_events))
            if prior_topics:
                options = prior_topics
            elif context.actor.interests:
                options = context.actor.interests
            else:
                options = (context.rules.common_topic,)
        return context.rng.choice(options)

    @staticmethod
    def _approach(stance: Stance) -> str:
        return {
            Stance.PROACTIVE: "よかったら、少し話しませんか？",
            Stance.NEUTRAL: "この話題、どう思いますか？",
            Stance.CAUTIOUS: "もしよければ、聞かせてください。",
        }[stance]


class RuleResponseProvider:
    """Deterministic rule implementation for the target's response."""

    name = "rule-response-provider"
    version = "ab_poc_rules_v1"

    def respond(self, context: ParticipantContext, initiation: Initiation) -> ResponseKind:
        target = self._target(context, initiation.target_id)
        pair_events = _events_for_pair(context.events, context.actor.id, target.id)
        return self._respond_with_events(context, target, initiation.topic, pair_events)

    def respond_output(self, context: ParticipantContext, initiation: Initiation) -> ResponseOutput:
        """Return the M10 response DTO without changing the legacy ``respond`` API."""

        reaction = self.respond(context, initiation)
        reply = {
            ResponseKind.POSITIVE: "その話、もう少し聞いてみたいです。",
            ResponseKind.NEUTRAL: "なるほど、そうなんですね。",
            ResponseKind.MISALIGNED: "少し難しい話かもしれません。",
        }[reaction]
        return ResponseOutput(reply, reaction)

    def _respond_with_events(
        self,
        context: ParticipantContext,
        target: Participant,
        topic: str,
        pair_events: Sequence[Event],
    ) -> ResponseKind:
        points = self.reaction_points(context, target, topic, pair_events)
        reaction = context.rules.reaction
        if points >= reaction.positive_threshold:
            return ResponseKind.POSITIVE
        if points >= reaction.neutral_threshold:
            return ResponseKind.NEUTRAL
        return ResponseKind.MISALIGNED

    @staticmethod
    def _target(context: ParticipantContext, target_id: str) -> Participant:
        for candidate in context.candidates:
            if candidate.id == target_id:
                return candidate
        raise ValueError("initiation target must be one of the participant candidates")

    def reaction_points(
        self,
        context: ParticipantContext,
        target: Participant,
        topic: str,
        pair_events: Sequence[Event],
    ) -> int:
        """Calculate reaction points and consume one jitter draw."""

        rules = context.rules.reaction
        state = _state_for(context.pair_states, context.actor.id, target.id)
        online_known = state is not None and state.online_known is OnlineKnown.DIRECT
        shared = bool(_shared_interests(context.actor, target))
        prior_positive = any(event.response is ResponseKind.POSITIVE for event in pair_events)
        points = rules.base
        if online_known:
            points += rules.online_known_bonus
        if topic in target.interests:
            points += rules.topic_match_bonus
        if prior_positive:
            points += rules.prior_positive_bonus
        if context.actor.stance is Stance.PROACTIVE:
            points += rules.proactive_actor_bonus
        if context.actor.stance is Stance.CAUTIOUS and not (online_known or shared):
            points += rules.cautious_actor_penalty
        if target.stance is Stance.CAUTIOUS and not (online_known or shared or prior_positive):
            points += rules.cautious_target_penalty
        points += context.rng.choice(rules.jitter_values)
        return points


class RuleParticipantProvider:
    """Thin composition preserving the versioned rule provider contract."""

    name = "rule-participant"
    version = "ab_poc_rules_v1"

    def __init__(
        self,
        *,
        initiator: RuleTalkInitiator | None = None,
        responder: RuleResponseProvider | None = None,
    ) -> None:
        self.initiator = initiator or RuleTalkInitiator()
        self.responder = responder or RuleResponseProvider()
        self.last_response_output: ResponseOutput | None = None

    def propose(self, context: ParticipantContext) -> ParticipantTurn:
        # This field is an engine-side compatibility bridge; never let a
        # previous turn's reply be observed as the current one.
        self.last_response_output = None
        initiation = self.initiator.initiate(context)
        raw_response = (
            self.responder.respond_output(context, initiation)
            if callable(getattr(self.responder, "respond_output", None))
            else self.responder.respond(context, initiation)
        )
        response, _, output = adapt_response(raw_response)
        self.last_response_output = output
        return ParticipantTurn(
            context.turn,
            context.actor.id,
            initiation.target_id,
            initiation.topic,
            initiation.approach,
            response,
        )

    def selection_weight(self, context: ParticipantContext, candidate: Participant) -> float:
        return self.initiator.selection_weight(context, candidate)

    def reaction_points(
        self,
        context: ParticipantContext,
        target: Participant,
        topic: str,
        pair_events: Sequence[Event],
    ) -> int:
        return self.responder.reaction_points(context, target, topic, pair_events)

    def _choose_topic(
        self, context: ParticipantContext, target: Participant, pair_events: Sequence[Event]
    ) -> str:
        return self.initiator._choose_topic(context, target, pair_events)

    @staticmethod
    def _approach(stance: Stance) -> str:
        return RuleTalkInitiator._approach(stance)

    def _response(
        self,
        context: ParticipantContext,
        target: Participant,
        topic: str,
        pair_events: Sequence[Event],
    ) -> ResponseKind:
        if isinstance(self.responder, RuleResponseProvider):
            return self.responder._respond_with_events(context, target, topic, pair_events)
        initiation = Initiation(target.id, topic, self._approach(context.actor.stance))
        return self.responder.respond(context, initiation)


class RuleRelationshipEvaluator:
    """Blind evaluator that derives relationship only from a pair's events."""

    name = "rule-relationship"
    version = "ab_poc_rules_v1"

    def __init__(self, thresholds: RelationshipThresholds | RuleConfig | None = None) -> None:
        if isinstance(thresholds, RuleConfig):
            thresholds = thresholds.relationship
        self.thresholds = thresholds or RelationshipThresholds()

    def evaluate(self, events: Sequence[Event]) -> RelationshipUpdate:
        """Recompute all counters and the onsite stage from events alone.

        The returned PairState deliberately has ``online_known=NONE``.  The
        engine may merge its real condition-layer value after this blind
        evaluation; the evaluator itself never observes or infers it.
        """

        observed = tuple(events)
        if not observed:
            raise ValueError("at least one event is required")
        turns = tuple(event.turn for event in observed)
        if turns != tuple(sorted(turns)) or len(set(turns)) != len(turns):
            raise ValueError("evaluation events must have unique turns in ascending order")
        pair = canonical_pair(observed[0].actor, observed[0].target)
        if any(canonical_pair(event.actor, event.target) != pair for event in observed):
            raise ValueError("all evaluation events must belong to one pair")

        previous = self._state(observed[:-1], pair, previous_relationship=True)
        current = self._state(observed, pair, previous_relationship=False)
        relationship = current.onsite_relationship
        return RelationshipUpdate(
            current,
            previous.onsite_relationship,
            relationship,
            self._reason(current, previous.onsite_relationship),
        )

    def _state(
        self,
        events: Sequence[Event],
        pair: Pair,
        *,
        previous_relationship: bool,
    ) -> PairState:
        count = len(events)
        positive = sum(event.response is ResponseKind.POSITIVE for event in events)
        actor_ids = tuple(dict.fromkeys(event.actor for event in events))
        thresholds = self.thresholds
        relationship = OnsiteRelationship.NOT_FORMED
        if (
            count >= thresholds.familiar_conversations
            and positive >= thresholds.familiar_positive
            and (
                not thresholds.familiar_requires_bidirectional
                or set(actor_ids) >= set(pair)
            )
        ):
            relationship = OnsiteRelationship.FAMILIAR
        elif count >= thresholds.acquaintance_conversations and positive >= thresholds.acquaintance_positive:
            relationship = OnsiteRelationship.ACQUAINTANCE
        # ``previous_relationship`` is retained as a named argument to make
        # this helper's intent clear to readers; monotonicity follows from the
        # complete event sequence and therefore needs no state input.
        del previous_relationship
        return PairState(
            *pair,
            online_known=OnlineKnown.NONE,
            contact_history=(ContactHistory.CONVERSATION if count else ContactHistory.NONE),
            onsite_relationship=relationship,
            conversation_count=count,
            positive_count=positive,
            actor_ids=actor_ids,
        )

    @staticmethod
    def _reason(state: PairState, previous: OnsiteRelationship) -> str:
        return (
            f"現地会話{state.conversation_count}回、前向き{state.positive_count}回"
            f"（{previous.value}→{state.onsite_relationship.value}）"
        )


__all__ = [
    "RuleParticipantProvider",
    "RuleRelationshipEvaluator",
    "RuleResponseProvider",
    "RuleTalkInitiator",
]
