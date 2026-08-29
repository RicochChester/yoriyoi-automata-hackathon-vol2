"""Deterministic execution kernel for the four-person A/B POC."""

from __future__ import annotations

import hashlib
import random
from copy import deepcopy
from collections.abc import Mapping
from itertools import combinations
from typing import Any

from .config import load_config, parse_config
from .contracts import (
    ConversationEpisode,
    Initiation,
    ParticipantContext,
    ParticipantProvider,
    RelationshipEvaluator,
    ResponseContext,
    VerbalEpisodeEvaluator,
    VerbalEpisodeEvaluation,
    adapt_response,
)
from .affect import AffectUpdater, DirectionalAffectState
from .domain import (
    Condition,
    DialogueTurn,
    Event,
    OnlineKnown,
    OnsiteRelationship,
    Pair,
    PairState,
    Participant,
    ParticipantTurn,
    PocConfig,
    RelationshipUpdate,
    ResponseKind,
    canonical_pair,
    to_jsonable,
)
from .metrics import calculate_metrics
from .rule_providers import RuleParticipantProvider, RuleRelationshipEvaluator
from .conversation_provider import SplitConversationProvider
from .narrative import LocalNarrativeProvider
from .verbal_evaluator import DeterministicVerbalEpisodeEvaluator


ENGINE_SCHEMA_VERSION = "ab_poc.run.v2"

_RELATIONSHIP_RANK = {
    OnsiteRelationship.NOT_FORMED: 0,
    OnsiteRelationship.ACQUAINTANCE: 1,
    OnsiteRelationship.FAMILIAR: 2,
}


def _provider_name(provider: Any, fallback: str) -> dict[str, str]:
    return {
        "name": str(getattr(provider, "name", provider.__class__.__name__)),
        "version": str(getattr(provider, "version", fallback)),
    }


def _narrative_metadata(provider: Any, *, default_source: str = "local") -> dict[str, str]:
    """Project optional narrative provenance into a small safe DTO.

    Narrative metadata is presentation-only. Restrict it to known strings so
    a custom provider cannot accidentally place API responses or exception
    text into RunResult.
    """

    source_values = {"deepseek", "local_fallback", "local"}
    source = default_source if default_source in source_values else "local"
    describe = getattr(provider, "narrative_metadata", None)
    if callable(describe):
        try:
            value = describe()
        except Exception:
            value = None
        if isinstance(value, Mapping):
            candidate = value.get("source")
            if candidate in source_values:
                source = candidate
            reason = value.get("fallback_reason")
            if source == "local_fallback" and isinstance(reason, str) and reason and len(reason) <= 64:
                return {"source": source, "fallback_reason": reason}
    return {"source": source}


def _legacy_participant_payload(participants: tuple[Participant, ...]) -> list[dict[str, Any]]:
    """Keep the established RunResult participant shape stable.

    Persona cards are configuration/prompt inputs, not a replay-contract
    change, so prior sample runs remain byte-for-byte comparable.
    """

    return [
        {
            "id": person.id,
            "name": person.name,
            "interests": list(person.interests),
            "stance": person.stance.value,
        }
        for person in participants
    ]


def _validate_provider(provider: Any, *, kind: str) -> None:
    """Fail early when an injected provider cannot satisfy the small contract."""

    method = "propose" if kind == "participant" else "evaluate"
    if provider is None or not callable(getattr(provider, method, None)):
        raise ValueError(
            f"{kind} provider must implement {method}(); "
            "use a provider factory that returns a fresh contract-compatible instance"
        )
    for field in ("name", "version"):
        value = getattr(provider, field, None)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"{kind} provider must expose non-empty string {field} metadata"
            )


def _clone_prototype(provider: Any, *, kind: str, factory_name: str) -> tuple[Any, Any]:
    """Clone an explicitly injected provider once for each A/B condition."""

    try:
        first = deepcopy(provider)
        second = deepcopy(provider)
    except Exception as exc:  # pragma: no cover - concrete exception is implementation-specific
        raise ValueError(
            f"{kind} provider could not be copied for A/B isolation; "
            f"pass {factory_name}=callable returning a fresh instance"
        ) from exc
    if first is provider or second is provider or first is second:
        raise ValueError(
            f"{kind} provider copy is not independent for A/B isolation; "
            f"pass {factory_name}=callable returning a fresh instance"
        )
    _validate_provider(first, kind=kind)
    _validate_provider(second, kind=kind)
    if _provider_name(first, "unknown") != _provider_name(second, "unknown"):
        raise ValueError(
            f"{factory_name} must return matching name/version metadata for A and B"
        )
    return first, second


def _factory_pair(factory: Any, *, kind: str, factory_name: str) -> tuple[Any, Any]:
    """Create and validate one independent provider for each condition."""

    if not callable(factory):
        raise ValueError(f"{factory_name} must be a callable returning a fresh {kind} provider")
    try:
        first = factory()
        second = factory()
    except Exception as exc:
        raise ValueError(f"{factory_name} failed while creating A/B providers") from exc
    if first is None or second is None:
        raise ValueError(f"{factory_name} must return a non-None {kind} provider")
    if first is second:
        raise ValueError(f"{factory_name} returned the same {kind} provider instance twice")
    _validate_provider(first, kind=kind)
    _validate_provider(second, kind=kind)
    if _provider_name(first, "unknown") != _provider_name(second, "unknown"):
        raise ValueError(f"{factory_name} must return matching name/version metadata for A and B")
    return first, second


def _validate_conversation_provider(provider: Any) -> None:
    """Validate the small split conversation contract used by M10.6."""

    if provider is None or not callable(getattr(provider, "initiate", None)):
        raise ValueError(
            "conversation provider must implement initiate() and respond()"
        )
    if not callable(getattr(provider, "respond", None)):
        raise ValueError(
            "conversation provider must implement initiate() and respond()"
        )
    if getattr(provider, "mode", None) not in {
        "rule_no_affect", "llm_no_affect", "llm_affect", "deepseek_natural", "deepseek_autonomous"
    }:
        raise ValueError("conversation provider mode is invalid")
    for field in ("name", "version"):
        value = getattr(provider, field, None)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"conversation provider must expose non-empty string {field} metadata"
            )


def _validate_verbal_episode_evaluator(provider: Any) -> None:
    """Validate the replaceable, presentation-only episode evaluator."""

    if provider is None or not callable(getattr(provider, "evaluate", None)):
        raise ValueError("verbal episode evaluator must implement evaluate()")
    for field in ("name", "version"):
        value = getattr(provider, field, None)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"verbal episode evaluator must expose non-empty string {field} metadata"
            )


def _conversation_factory_pair(factory: Any) -> tuple[Any, Any]:
    """Create two independent split providers for A/B."""

    factory_name = "conversation_provider_factory"
    if not callable(factory):
        raise ValueError(
            f"{factory_name} must be a callable returning a fresh conversation provider"
        )
    try:
        first = factory()
        second = factory()
    except Exception as exc:
        raise ValueError(f"{factory_name} failed while creating A/B providers") from exc
    if first is None or second is None:
        raise ValueError(f"{factory_name} must return non-None providers")
    if first is second:
        raise ValueError(f"{factory_name} returned the same provider instance twice")
    _validate_conversation_provider(first)
    _validate_conversation_provider(second)
    if (_provider_name(first, "unknown"), first.mode) != (
        _provider_name(second, "unknown"), second.mode
    ):
        raise ValueError(f"{factory_name} must return matching mode/name/version metadata")
    return first, second


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer (bool is not accepted)")
    return seed


def _stable_provider_seed(root_seed: int, turn: int) -> int:
    # Deliberately avoid Python's process-randomized hash().  The textual
    # representation is part of the deterministic run contract.
    token = f"ab_poc.provider.v1\0{root_seed!r}\0turn:{turn}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:16], "big")


def _condition(config: PocConfig, condition_id: str) -> Condition:
    for condition in config.conditions:
        if condition.id == condition_id:
            return condition
    raise ValueError(f"unknown condition_id: {condition_id!r}")


def _pair_keys(config: PocConfig) -> tuple[Pair, ...]:
    return tuple(
        canonical_pair(left, right)
        for left, right in combinations((participant.id for participant in config.participants), 2)
    )


def _initial_states(config: PocConfig, condition: Condition) -> dict[Pair, PairState]:
    return {
        pair: PairState(*pair, online_known=condition.online_known(*pair))
        for pair in _pair_keys(config)
    }


def _snapshot(turn: int, phase: str, states: Mapping[Pair, PairState], pairs: tuple[Pair, ...]) -> dict[str, Any]:
    return {
        "turn": turn,
        "phase": phase,
        "pairs": [to_jsonable(states[pair]) for pair in pairs],
    }


def _validate_turn(
    proposal: Any,
    *,
    expected_turn: int,
    actor_id: str,
    candidate_ids: set[str],
    participant_ids: set[str],
) -> ParticipantTurn:
    if not isinstance(proposal, ParticipantTurn):
        raise ValueError("participant provider must return ParticipantTurn")
    if proposal.turn != expected_turn:
        raise ValueError("participant provider returned an invalid turn")
    if proposal.actor != actor_id:
        raise ValueError("participant provider returned an invalid actor")
    if proposal.actor not in participant_ids:
        raise ValueError("participant provider returned an unknown actor")
    if proposal.target not in candidate_ids or proposal.target not in participant_ids:
        raise ValueError("participant provider returned an invalid target")
    if proposal.target == proposal.actor:
        raise ValueError("participant provider returned a self-target")
    if not isinstance(proposal.topic, str) or not proposal.topic:
        raise ValueError("participant provider returned an invalid topic")
    if not isinstance(proposal.approach, str) or not proposal.approach:
        raise ValueError("participant provider returned an invalid approach")
    return proposal


def _reaction_label(response: ResponseKind) -> str:
    return {
        ResponseKind.POSITIVE: "positive",
        ResponseKind.NEUTRAL: "neutral",
        ResponseKind.MISALIGNED: "misaligned",
    }[response]


def _provider_turn_audit(provider: Any, turn: int) -> dict[str, Any]:
    """Read only the current turn's optional response provenance."""

    snapshotter = getattr(provider, "audit_snapshot", None)
    if not callable(snapshotter):
        return {}
    try:
        snapshot = snapshotter()
    except Exception:
        return {}
    turns = snapshot.get("turns") if isinstance(snapshot, Mapping) else None
    if not isinstance(turns, (tuple, list)):
        return {}
    for item in turns:
        if isinstance(item, Mapping) and item.get("turn") == turn:
            return dict(item)
    return {}


def _response_audit(provider: Any, turn: int, response: ResponseKind) -> dict[str, Any]:
    """Build a stable response/fallback audit view for old and new providers."""

    item = _provider_turn_audit(provider, turn)
    response_output = item.get("response_output")
    if not isinstance(response_output, Mapping):
        response_output = {}
    if not response_output:
        latest = getattr(provider, "last_response_output", None)
        if latest is not None:
            response_output = {
                "reply": getattr(latest, "reply", None),
                "reaction": getattr(latest, "reaction", None),
            }
    reaction = response_output.get("reaction")
    if not isinstance(reaction, str):
        reaction = _reaction_label(response)
    reply = response_output.get("reply")
    if not isinstance(reply, str):
        # A legacy provider intentionally has no reply channel.  Keep this
        # explicit and JSON-safe instead of inventing an utterance.
        reply = None
    fallback = item.get("fallback")
    if not isinstance(fallback, Mapping):
        fallback = {}
    response_fallback = bool(fallback.get("response", False))
    reason = fallback.get("response_reason")
    if reason is not None and not isinstance(reason, str):
        reason = str(reason)
    return {
        "reply": reply,
        "reaction": reaction,
        "fallback": {
            "response": response_fallback,
            "response_reason": reason if response_fallback else None,
        },
    }


def _transition_payload(transition: Any) -> dict[str, Any]:
    """Serialize one affect transition with explicit direction metadata."""

    payload = to_jsonable(transition)
    payload["reaction"] = _reaction_label(transition.reaction)
    payload["direction"] = {
        "receiver": transition.receiver,
        "initiator": transition.initiator,
    }
    payload["basis_turn"] = transition.turn
    return payload


def _affect_state_payload(state: DirectionalAffectState) -> dict[str, Any]:
    """Serialize affect values as stable, human-readable direction records."""

    return {
        "participants": list(state.participants),
        "values": [
            {"owner": owner, "other": other, "value": state.value(owner, other)}
            for owner in state.participants
            for other in state.participants
            if owner != other
        ],
    }


def _outgoing_affect(
    state: DirectionalAffectState,
    actor: str,
    candidates: tuple[Any, ...],
) -> dict[str, int]:
    """Project only the current actor's private values to candidate IDs."""

    return {candidate.id: state.value(actor, candidate.id) for candidate in candidates}


def _decision_affect_payload(context: ParticipantContext) -> dict[str, Any]:
    """Serialize the exact affect projection used at proposal time once."""

    return {
        "actor": context.actor.id,
        "outgoing": [
            {"candidate": candidate.id, "value": context.outgoing_affect[candidate.id]}
            for candidate in context.candidates
        ],
    }


def _validate_initiation(
    initiation: Any,
    *,
    actor_id: str,
    candidate_ids: set[str],
) -> Initiation:
    if not isinstance(initiation, Initiation):
        raise ValueError("conversation initiator must return Initiation")
    if initiation.target_id not in candidate_ids or initiation.target_id == actor_id:
        raise ValueError("conversation initiation target must be one of the candidates")
    return initiation


def _conversation_response_audit(provider: Any, turn: int, response: ResponseKind, reply: str | None) -> dict[str, Any]:
    item = _provider_turn_audit(provider, turn)
    fallback = item.get("fallback") if isinstance(item.get("fallback"), Mapping) else {}
    response_fallback = bool(fallback.get("responder", False))
    reason = fallback.get("responder_reason")
    return {
        "reply": reply,
        "reaction": _reaction_label(response),
        "fallback": {
            "response": response_fallback,
            "response_reason": reason if response_fallback else None,
        },
    }


def _episode_payload(
    episode: ConversationEpisode | None,
    *,
    next_sequence: int,
) -> tuple[list[dict[str, Any]] | None, int]:
    """Serialize an optional episode with run-global integer sequence IDs."""

    if episode is None:
        return None, next_sequence
    payload: list[dict[str, Any]] = []
    sequence = next_sequence
    for item in episode.utterances:
        payload.append({
            "sequence": sequence,
            "speaker": item.speaker,
            "text": item.text,
        })
        sequence += 1
    return payload, sequence


def _comparison_summary(
    result_turns: list[dict[str, Any]],
    *,
    condition: Condition,
    mode: str,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    known = sum(
        condition.online_known(turn["actor"], turn["target"]) is OnlineKnown.DIRECT
        for turn in result_turns
    )
    known_pairs = [list(pair) for pair in sorted({
        canonical_pair(turn["actor"], turn["target"])
        for turn in result_turns
        if condition.online_known(turn["actor"], turn["target"]) is OnlineKnown.DIRECT
    })]
    non_direct_pairs = [list(pair) for pair in sorted({
        canonical_pair(turn["actor"], turn["target"])
        for turn in result_turns
        if condition.online_known(turn["actor"], turn["target"]) is not OnlineKnown.DIRECT
    })]
    transitions = [turn["affect_transition"] for turn in result_turns if turn.get("affect_transition")]
    changed = sum(item["before"] != item["after"] for item in transitions)
    later: list[dict[str, Any]] = []
    for transition in transitions:
        if transition["before"] == transition["after"]:
            continue
        # Affect direction is receiver -> initiator.  A later decision is
        # aligned only when that receiver selects the same initiator.
        for turn in result_turns:
            if (
                turn["turn"] > transition["turn"]
                and turn["actor"] == transition["receiver"]
                and turn["target"] == transition["initiator"]
            ):
                later.append({
                    "source_turn": transition["turn"],
                    "before": transition["before"],
                    "after": transition["after"],
                    "selection_turn": turn["turn"],
                    "actor": turn["actor"],
                    "target": turn["target"],
                    "topic": turn["topic"],
                    "approach": turn["approach"],
                })
    summary = audit.get("summary") if isinstance(audit.get("summary"), Mapping) else {}
    return {
        "mode": mode,
        "known_pair_conversations": known,
        "known_pair_count": len(known_pairs),
        "known_pairs": known_pairs,
        "non_direct_pair_conversations": len(result_turns) - known,
        "non_direct_pair_count": len(non_direct_pairs),
        "non_direct_pairs": non_direct_pairs,
        "affect_changes": changed,
        "selections_after_affect_change": later,
        "fallbacks": {
            "initiator": int(summary.get("initiator_fallback", 0)),
            "responder": int(summary.get("responder_fallback", 0)),
        },
    }


def run_condition(
    condition_id: str,
    *,
    seed: int,
    participant_provider: ParticipantProvider | None = None,
    conversation_provider: SplitConversationProvider | None = None,
    relationship_evaluator: RelationshipEvaluator | None = None,
    verbal_episode_evaluator: VerbalEpisodeEvaluator | None = None,
    narrative_provider: Any | None = None,
    config: PocConfig | None = None,
) -> dict[str, Any]:
    """Run exactly one condition and return a JSON-serializable result."""

    seed = _validate_seed(seed)
    if config is None:
        config = load_config()
    elif isinstance(config, Mapping):
        config = parse_config(config)
    condition = _condition(config, condition_id)
    if participant_provider is not None and conversation_provider is not None:
        raise ValueError("pass either participant_provider or conversation_provider, not both")
    if participant_provider is None:
        if conversation_provider is None:
            participant_provider = RuleParticipantProvider()
    if relationship_evaluator is None:
        relationship_evaluator = RuleRelationshipEvaluator(config.rules)
    if verbal_episode_evaluator is not None:
        _validate_verbal_episode_evaluator(verbal_episode_evaluator)
    elif conversation_provider is not None:
        # This is a presentation-only default. Rule mode has no episode and
        # therefore never instantiates or records this evaluator.
        verbal_episode_evaluator = DeterministicVerbalEpisodeEvaluator()
    participants = tuple(config.participants)
    participant_by_id = {person.id: person for person in participants}
    participant_ids = set(participant_by_id)
    pairs = _pair_keys(config)

    actor_order = list(participants)
    random.Random(seed).shuffle(actor_order)
    # The standard AB scenario runs one complete actor order per configured
    # cycle.  Keeping this derived from the config preserves the existing
    # order/reproducibility contract while allowing the standard to grow.
    cycles, remainder = divmod(config.rules.turns, len(actor_order))
    if remainder:
        raise ValueError("rules.turns must be divisible by participant count")
    turn_order = tuple(person.id for person in actor_order) * cycles
    states = _initial_states(config, condition)
    events: list[Event] = []
    dialogue_history: list[DialogueTurn] = []
    turn_logs: list[dict[str, Any]] = []
    # Optional conversation episodes use one monotonic sequence for the whole
    # run. Rule mode never creates an episode and therefore remains unchanged.
    next_utterance_sequence = 1
    reasons: list[dict[str, Any]] = []
    last_target: dict[str, str | None] = {person.id: None for person in participants}
    snapshots = [_snapshot(0, "start", states, pairs)]
    # This is deliberately created inside each single-condition run.  It is
    # immutable and therefore cannot leak updates into another run or side of
    # an A/B comparison.
    initial_affect_state = DirectionalAffectState.initial(tuple(person.id for person in participants))
    affect_state = initial_affect_state
    affect_updater = AffectUpdater()
    affect_transitions: list[dict[str, Any]] = []
    affect_enabled = conversation_provider is None or conversation_provider.affect_enabled
    active_provider = conversation_provider or participant_provider

    for turn, actor in enumerate(turn_order, start=1):
        response_output = None
        candidates = tuple(person for person in participants if person.id != actor)
        context = ParticipantContext(
            turn=turn,
            actor=participant_by_id[actor],
            candidates=candidates,
            pair_states=dict(states),
            events=tuple(events),
            last_target=last_target[actor],
            rules=config.rules,
            # Actor ordering and provider randomness are separate streams.
            rng=random.Random(_stable_provider_seed(seed, turn)),
            # Only the actor's own projection crosses the provider boundary.
            # ``online_known`` remains in pair states as a compatibility
            # boolean; the versioned memory/content is never global there.
            online_memories=(
                condition.online_experience.memory_for(actor)
                if condition.online_experience is not None else ()
            ),
            outgoing_affect=(
                _outgoing_affect(affect_state, actor, candidates)
                if affect_enabled else {candidate.id: 0 for candidate in candidates}
            ),
            dialogue_history=tuple(dialogue_history),
        )
        if conversation_provider is None:
            proposal = _validate_turn(
                participant_provider.propose(context),
                expected_turn=turn,
                actor_id=actor,
                candidate_ids={person.id for person in candidates},
                participant_ids=participant_ids,
            )
            response_audit = _response_audit(participant_provider, turn, proposal.response)
        else:
            initiation = _validate_initiation(
                conversation_provider.initiate(context),
                actor_id=actor,
                candidate_ids={person.id for person in candidates},
            )
            target = participant_by_id[initiation.target_id]
            pair = canonical_pair(actor, target.id)
            pair_events = tuple(
                item for item in events
                if canonical_pair(item.actor, item.target) == pair
            )
            memory = (
                condition.online_experience.memory_for_pair(actor, target.id)
                if condition.online_experience is not None else None
            )
            response_context = ResponseContext(
                responder=target,
                initiator=participant_by_id[actor],
                pair_events=pair_events,
                online_memory=memory,
                affect=(affect_state.value(target.id, actor) if affect_enabled else 0),
                turn=turn,
                dialogue_history=tuple(
                    item for item in dialogue_history
                    if canonical_pair(item.actor, item.target) == pair
                ),
                public_dialogue_history=tuple(dialogue_history[-4:]),
                public_participant_tokens=tuple(
                    token
                    for person in participants
                    for token in (person.id, person.name)
                    if token
                ),
            )
            episode_runner = getattr(conversation_provider, "respond_episode", None)
            if callable(episode_runner):
                raw_response = episode_runner(context, response_context, initiation)
            else:
                raw_response = conversation_provider.respond(response_context, initiation)
            response, reply, response_output = adapt_response(raw_response)
            proposal = ParticipantTurn(
                turn,
                actor,
                initiation.target_id,
                initiation.topic,
                initiation.approach,
                response,
            )
            response_audit = _conversation_response_audit(
                conversation_provider,
                turn,
                response,
                response_output.reply if response_output is not None else reply,
            )
        dialogue_history.append(DialogueTurn(
            turn=proposal.turn,
            actor=proposal.actor,
            target=proposal.target,
            topic=proposal.topic,
            approach=proposal.approach,
            reply=response_audit["reply"],
            reaction=response_audit["reaction"],
        ))
        episode = response_output.episode if response_output is not None else None
        episode_payload, next_utterance_sequence = _episode_payload(
            episode, next_sequence=next_utterance_sequence
        )
        event = Event(
            proposal.turn,
            proposal.actor,
            proposal.target,
            proposal.topic,
            proposal.approach,
            proposal.response,
        )
        events.append(event)
        last_target[actor] = proposal.target
        if affect_enabled:
            affect_state, transition = affect_updater.update(
                affect_state,
                turn,
                receiver=proposal.target,
                initiator=proposal.actor,
                reaction=proposal.response,
            )
            transition_payload = _transition_payload(transition)
            affect_transitions.append(transition_payload)
        else:
            transition_payload = None
        pair = canonical_pair(proposal.actor, proposal.target)
        pair_events = tuple(item for item in events if canonical_pair(item.actor, item.target) == pair)
        update = relationship_evaluator.evaluate(pair_events)
        if not isinstance(update, RelationshipUpdate):
            raise ValueError("relationship evaluator must return RelationshipUpdate")
        if update.pair.pair != pair:
            raise ValueError("relationship evaluator returned an invalid pair")
        previous_state = states[pair]
        expected_actor_ids = tuple(dict.fromkeys(event.actor for event in pair_events))
        expected_positive_count = sum(event.response is ResponseKind.POSITIVE for event in pair_events)
        if update.pair.online_known is not OnlineKnown.NONE:
            raise ValueError("relationship evaluator must not return online_known")
        if not isinstance(update.previous_relationship, OnsiteRelationship) or not isinstance(update.relationship, OnsiteRelationship):
            raise ValueError("relationship evaluator returned an invalid relationship stage")
        if _RELATIONSHIP_RANK[update.relationship] < _RELATIONSHIP_RANK[previous_state.onsite_relationship]:
            raise ValueError("relationship evaluator must not regress onsite relationship")
        if update.relationship is not update.pair.onsite_relationship:
            raise ValueError("relationship evaluator relationship does not match pair state")
        if update.pair.contact_history.value != "会話経験あり":
            raise ValueError("relationship evaluator contact_history is invalid")
        if update.pair.conversation_count != len(pair_events):
            raise ValueError("relationship evaluator conversation_count is invalid")
        if update.pair.positive_count != expected_positive_count:
            raise ValueError("relationship evaluator positive_count is invalid")
        if update.pair.actor_ids != expected_actor_ids:
            raise ValueError("relationship evaluator actor_ids are invalid")
        if update.previous_relationship is not previous_state.onsite_relationship:
            raise ValueError("relationship evaluator previous_relationship is invalid")
        if not isinstance(update.reason, str) or not update.reason:
            raise ValueError("relationship evaluator returned an invalid reason")
        observed = update.pair
        # Only blind onsite fields cross the evaluator boundary.  In
        # particular, online_known always comes from the condition held here.
        states[pair] = PairState(
            *pair,
            online_known=states[pair].online_known,
            contact_history=observed.contact_history,
            onsite_relationship=observed.onsite_relationship,
            conversation_count=observed.conversation_count,
            positive_count=observed.positive_count,
            actor_ids=observed.actor_ids,
        )
        reasons.append(
            {
                "turn": turn,
                "pair": list(pair),
                "previous_relationship": update.previous_relationship.value,
                "relationship": update.relationship.value,
                "contact_history": update.pair.contact_history.value,
                "reason": update.reason,
            }
        )
        turn_log = {
                **to_jsonable(event),
                "reply": response_audit["reply"],
                "reaction": response_audit["reaction"],
                "fallback": response_audit["fallback"],
                # This is the sole per-turn record of the private affect
                # projection used by the decision.  The transition below is
                # retained separately as the response-induced state change.
                "decision_affect": _decision_affect_payload(context),
                "affect_transition": transition_payload,
                "pair": list(pair),
                "evaluation": {
                    "previous_relationship": update.previous_relationship.value,
                    "relationship": update.relationship.value,
                    "contact_history": update.pair.contact_history.value,
                    "reason": update.reason,
                },
                "evaluation_reason": update.reason,
            }
        verbal_evaluation: VerbalEpisodeEvaluation | None = None
        if episode_payload is not None and episode.ended:
            local_evaluation = verbal_episode_evaluator.evaluate(episode)
            if not isinstance(local_evaluation, VerbalEpisodeEvaluation):
                raise ValueError(
                    "verbal episode evaluator must return VerbalEpisodeEvaluation"
                )
            # Evaluators see the episode-local transcript numbers. The stored
            # replay uses run-global numbers, so translate evidence explicitly
            # instead of leaking a misleading 1/2/... reference.
            global_sequences = {
                item.sequence: serialized["sequence"]
                for item, serialized in zip(episode.utterances, episode_payload)
            }
            try:
                evidence = tuple(
                    global_sequences[sequence]
                    for sequence in local_evaluation.evidence_sequences
                )
            except KeyError as exc:
                raise ValueError(
                    "verbal episode evaluator returned unknown evidence sequence"
                ) from exc
            verbal_evaluation = VerbalEpisodeEvaluation(
                relationship_movement=local_evaluation.relationship_movement,
                end_momentum=local_evaluation.end_momentum,
                reason=local_evaluation.reason,
                evidence_sequences=evidence,
            )
        if episode_payload is not None:
            turn_log["episode"] = episode_payload
            turn_log["episode_ended"] = episode.ended
            turn_log["episode_end_reason"] = episode.end_reason
            if verbal_evaluation is not None:
                turn_log["verbal_evaluation"] = to_jsonable(verbal_evaluation)
        turn_logs.append(turn_log)
        snapshots.append(_snapshot(turn, "turn_end", states, pairs))

    snapshots.append(_snapshot(config.rules.turns, "end", states, pairs))
    result: dict[str, Any] = {
        "schema_version": ENGINE_SCHEMA_VERSION,
        "config_version": config.config_version,
        "root_seed": seed,
        "condition": to_jsonable(condition),
        "scenario": {
            "version": config.scenario_version,
            "online_experience": to_jsonable(condition.online_experience),
        },
        "condition_id": condition.id,
        "participants": _legacy_participant_payload(participants),
        "rules": to_jsonable(config.rules),
        "participant_provider": _provider_name(active_provider, "unknown"),
        "relationship_evaluator": _provider_name(relationship_evaluator, "unknown"),
        "actor_order": [person.id for person in actor_order],
        "turn_order": list(turn_order),
        "turns": turn_logs,
        "evaluation_reasons": reasons,
        "snapshots": snapshots,
        "initial_relationships": list(snapshots[0]["pairs"]),
        "final_relationships": list(snapshots[-1]["pairs"]),
        "initial_affect_state": _affect_state_payload(initial_affect_state),
        "final_affect_state": _affect_state_payload(affect_state),
        "affect_transitions": affect_transitions,
    }
    result["metrics"] = calculate_metrics(
        condition_id=condition.id,
        final_pairs=states,
        participants=participants,
        online_known_pairs=condition.online_known_pairs,
        turns=turn_logs,
    )
    # Narrative is an optional presentation boundary. It sees only the actual
    # dialogue DTOs and is invoked after evaluation/metrics; its output never
    # participates in simulation state or scoring.
    if narrative_provider is None and conversation_provider is not None:
        narrative_provider = getattr(conversation_provider, "narrative_provider", None)
    if narrative_provider is not None:
        try:
            generated = narrative_provider.generate(tuple(dialogue_history))
        except Exception:
            generated = LocalNarrativeProvider().generate(tuple(dialogue_history))
            narrative_meta = {"source": "local_fallback", "fallback_reason": "provider_error"}
        else:
            narrative_meta = _narrative_metadata(narrative_provider)
        if isinstance(generated, Mapping):
            result["narrative"] = to_jsonable(generated)
            result["narrative_meta"] = narrative_meta
    # M4 adapters may expose a safe, serializable audit trail.  Keep this
    # optional so the established rule-only result remains byte-for-byte
    # unchanged.
    if conversation_provider is not None:
        result["conversation_provider"] = _provider_name(conversation_provider, "unknown")
        result["conversation_mode"] = conversation_provider.mode
        result["affect_mode"] = "enabled" if affect_enabled else "disabled"
        conversation_audit = conversation_provider.audit_snapshot()
        result["comparison"] = _comparison_summary(
            turn_logs,
            condition=condition,
            mode=conversation_provider.mode,
            audit=conversation_audit,
        )
    audit_snapshot = getattr(active_provider, "audit_snapshot", None)
    if callable(audit_snapshot):
        result["participant_audit"] = to_jsonable(audit_snapshot())
    return result


def run_ab(
    *,
    seed: int,
    participant_provider: ParticipantProvider | None = None,
    conversation_provider_factory: Any | None = None,
    relationship_evaluator: RelationshipEvaluator | None = None,
    participant_provider_factory: Any | None = None,
    relationship_evaluator_factory: Any | None = None,
    verbal_episode_evaluator: VerbalEpisodeEvaluator | None = None,
    verbal_episode_evaluator_factory: Any | None = None,
    narrative_provider_factory: Any | None = None,
    config: PocConfig | None = None,
) -> dict[str, Any]:
    """Run A and B independently with one shared root seed.

    A/B runs must not share mutable provider or evaluator state.  Factories
    are the preferred injection point for stateful implementations.  For
    compatibility with the original API, an explicitly supplied instance is
    treated as a prototype and deep-copied once per condition.
    """

    seed = _validate_seed(seed)
    if config is None:
        config = load_config()
    elif isinstance(config, Mapping):
        config = parse_config(config)

    if participant_provider is not None and participant_provider_factory is not None:
        raise ValueError("pass either participant_provider or participant_provider_factory, not both")
    if conversation_provider_factory is not None and (
        participant_provider is not None or participant_provider_factory is not None
    ):
        raise ValueError(
            "conversation_provider_factory cannot be combined with participant_provider "
            "or participant_provider_factory"
        )
    if relationship_evaluator is not None and relationship_evaluator_factory is not None:
        raise ValueError("pass either relationship_evaluator or relationship_evaluator_factory, not both")
    if verbal_episode_evaluator is not None and verbal_episode_evaluator_factory is not None:
        raise ValueError(
            "pass either verbal_episode_evaluator or verbal_episode_evaluator_factory, not both"
        )
    if verbal_episode_evaluator_factory is not None and not callable(verbal_episode_evaluator_factory):
        raise ValueError("verbal_episode_evaluator_factory must be callable")
    if narrative_provider_factory is not None and not callable(narrative_provider_factory):
        raise ValueError("narrative_provider_factory must be callable")

    if conversation_provider_factory is not None:
        conversation_a, conversation_b = _conversation_factory_pair(
            conversation_provider_factory
        )
        participant_a = participant_b = None
    elif participant_provider_factory is not None:
        participant_a, participant_b = _factory_pair(
            participant_provider_factory,
            kind="participant",
            factory_name="participant_provider_factory",
        )
    elif participant_provider is not None:
        participant_a, participant_b = _clone_prototype(
            participant_provider,
            kind="participant",
            factory_name="participant_provider_factory",
        )
    else:
        participant_a, participant_b = RuleParticipantProvider(), RuleParticipantProvider()

    if relationship_evaluator_factory is not None:
        evaluator_a, evaluator_b = _factory_pair(
            relationship_evaluator_factory,
            kind="relationship",
            factory_name="relationship_evaluator_factory",
        )
    elif relationship_evaluator is not None:
        evaluator_a, evaluator_b = _clone_prototype(
            relationship_evaluator,
            kind="relationship",
            factory_name="relationship_evaluator_factory",
        )
    else:
        evaluator_a, evaluator_b = (
            RuleRelationshipEvaluator(config.rules),
            RuleRelationshipEvaluator(config.rules),
        )

    if verbal_episode_evaluator_factory is not None:
        verbal_a, verbal_b = _factory_pair(
            verbal_episode_evaluator_factory,
            kind="verbal_episode",
            factory_name="verbal_episode_evaluator_factory",
        )
    elif verbal_episode_evaluator is not None:
        verbal_a, verbal_b = _clone_prototype(
            verbal_episode_evaluator,
            kind="verbal_episode",
            factory_name="verbal_episode_evaluator_factory",
        )
    else:
        verbal_a = verbal_b = None

    if narrative_provider_factory is not None:
        try:
            narrative_a = narrative_provider_factory()
            narrative_b = narrative_provider_factory()
        except Exception as exc:
            raise ValueError("narrative_provider_factory failed while creating A/B providers") from exc
        if narrative_a is narrative_b or narrative_a is None or narrative_b is None:
            raise ValueError("narrative_provider_factory must return two independent providers")
        if not callable(getattr(narrative_a, "generate", None)) or not callable(getattr(narrative_b, "generate", None)):
            raise ValueError("narrative providers must implement generate()")
    else:
        narrative_a = narrative_b = None

    # Materialize both sides before condition A starts.  This makes the
    # boundary explicit and prevents construction order from leaking state.
    result_a = run_condition(
        "A", seed=seed,
        participant_provider=participant_a,
        conversation_provider=(conversation_a if conversation_provider_factory is not None else None),
        relationship_evaluator=evaluator_a,
        verbal_episode_evaluator=verbal_a,
        narrative_provider=narrative_a,
        config=config,
    )
    result_b = run_condition(
        "B", seed=seed,
        participant_provider=participant_b,
        conversation_provider=(conversation_b if conversation_provider_factory is not None else None),
        relationship_evaluator=evaluator_b,
        verbal_episode_evaluator=verbal_b,
        narrative_provider=narrative_b,
        config=config,
    )
    controlled_fields = {
        "participants": result_a["participants"],
        "rules": result_a["rules"],
        "root_seed": seed,
        "turn_order": result_a["turn_order"],
        "participant_provider": result_a["participant_provider"],
        "relationship_evaluator": result_a["relationship_evaluator"],
        "scenario_version": config.scenario_version,
    }
    if conversation_provider_factory is not None:
        controlled_fields["conversation_provider"] = result_a["conversation_provider"]
        controlled_fields["conversation_mode"] = result_a["conversation_mode"]
    result = {
        "schema_version": ENGINE_SCHEMA_VERSION,
        "config_version": config.config_version,
        "root_seed": seed,
        "participants": result_a["participants"],
        "conditions": {"A": result_a["condition"], "B": result_b["condition"]},
        "actor_order": result_a["actor_order"],
        "turn_order": result_a["turn_order"],
        "runs": {"A": result_a, "B": result_b},
        "metrics": {"A": result_a["metrics"], "B": result_b["metrics"]},
        "controlled_fields": controlled_fields,
        # The experiment factor is the versioned prior-experience setting.
        # online_known_pairs is retained as its legacy coarse projection.
        "changed_field": "condition.online_experience",
    }
    if conversation_provider_factory is not None:
        result["conversation_mode"] = result_a["conversation_mode"]
        result["affect_mode"] = result_a["affect_mode"]
        result["comparison"] = {
            "A": result_a["comparison"],
            "B": result_b["comparison"],
        }
    if "narrative" in result_a or "narrative" in result_b:
        result["narrative"] = {
            key: run_result["narrative"]
            for key, run_result in (("A", result_a), ("B", result_b))
            if "narrative" in run_result
        }
    if "narrative_meta" in result_a or "narrative_meta" in result_b:
        result["narrative_meta"] = {
            key: run_result["narrative_meta"]
            for key, run_result in (("A", result_a), ("B", result_b))
            if "narrative_meta" in run_result
        }
    if "participant_audit" in result_a or "participant_audit" in result_b:
        result["participant_audit"] = {
            key: run_result["participant_audit"]
            for key, run_result in (("A", result_a), ("B", result_b))
            if "participant_audit" in run_result
        }
    return result


__all__ = ["ENGINE_SCHEMA_VERSION", "run_ab", "run_condition"]
