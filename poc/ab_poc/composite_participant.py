"""Composable participant providers for the M5.1 responsibility split."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from .contracts import (
    Initiation,
    ParticipantContext,
    ResponseOutput,
    ResponseProvider,
    TalkInitiator,
    adapt_response,
)
from .domain import ParticipantTurn, ResponseKind
from .rule_providers import RuleResponseProvider, RuleTalkInitiator


class CompositeParticipantProvider:
    """Compose a talk initiator and response provider into one participant API.

    The optional fallback is deliberately transactional with respect to the
    run-owned RNG.  If a component fails after consuming randomness, the
    fallback starts from the same stream position and therefore cannot consume
    a second response jitter for the same turn.
    """

    name = "composite-participant"
    version = "ab_poc_composite_v1"

    def __init__(
        self,
        initiator: TalkInitiator,
        responder: ResponseProvider,
        *,
        fallback: Any | None = None,
    ) -> None:
        if initiator is None or not callable(getattr(initiator, "initiate", None)):
            raise ValueError("initiator must implement initiate()")
        if responder is None or not callable(getattr(responder, "respond", None)):
            raise ValueError("responder must implement respond()")
        self.initiator = initiator
        self.responder = responder
        # Descriptive aliases make the role boundary discoverable to callers
        # without changing the compact constructor terminology.
        self.talk_initiator = initiator
        self.response_provider = responder
        self.fallback = fallback
        self._audit_turns: list[dict[str, Any]] = []

    @staticmethod
    def _metadata(provider: Any, fallback: str) -> dict[str, str]:
        return {
            "name": str(getattr(provider, "name", provider.__class__.__name__)),
            "version": str(getattr(provider, "version", fallback)),
        }

    def _roles(self) -> dict[str, dict[str, str]]:
        return {
            "initiator": self._metadata(self.initiator, "unknown"),
            "responder": self._metadata(self.responder, "unknown"),
        }

    @staticmethod
    def _component_description(provider: Any) -> dict[str, Any]:
        describe = getattr(provider, "describe", None)
        if not callable(describe):
            return {}
        try:
            value = describe()
        except Exception:
            return {}
        return dict(value) if isinstance(value, dict) else {}

    def _initiator_provenance(self, turn: int) -> dict[str, Any]:
        """Read one initiator audit turn, retaining only safe audit fields."""

        snapshotter = getattr(self.initiator, "audit_snapshot", None)
        try:
            snapshot = snapshotter() if callable(snapshotter) else None
        except Exception:
            snapshot = None
        turns = snapshot.get("turns") if isinstance(snapshot, dict) else None
        current: Any = None
        if isinstance(turns, (tuple, list)):
            for item in turns:
                if isinstance(item, dict) and item.get("turn") == turn:
                    current = item
        if not isinstance(current, dict):
            return {
                "prompt_input": {},
                "attempts": [{"attempt": 1, "status": "fallback"}],
                "outcome": "fallback",
            }
        prompt_input = current.get("prompt_input")
        attempts = current.get("attempts")
        outcome = current.get("outcome")
        if not isinstance(prompt_input, dict):
            prompt_input = {}
        if not isinstance(attempts, (tuple, list)) or not attempts:
            attempts = [{"attempt": 1, "status": "fallback"}]
        if outcome not in {"llm", "hybrid", "fallback"}:
            outcome = "fallback"
        result = {
            "prompt_input": copy.deepcopy(prompt_input),
            "attempts": copy.deepcopy(list(attempts)),
            "outcome": outcome,
        }
        conversation_act = current.get("conversation_act")
        if isinstance(conversation_act, str):
            result["conversation_act"] = conversation_act
        for key in (
            "planner_prompt",
            "utterance_prompt",
            "planner_attempts",
            "utterance_attempts",
            "planner",
            "utterance",
            "selected_output",
            "adopted_plan",
            "final_approach",
            "fallback_stage",
            "fallback_reason",
            "candidate_count",
            "selected_plan_id",
            "token_usage",
            "total_latency",
            "total_latency_ms",
            "decision_source",
            "approach_source",
            "approach_validation_code",
            "history_reference",
            "episode",
            "episode_ended",
            "episode_reaction",
        ):
            if key in current:
                result[key] = copy.deepcopy(current[key])
        return result

    def _record_audit(
        self,
        context: ParticipantContext,
        initiation: Initiation | None,
        turn: ParticipantTurn,
        response_provider: Any,
        *,
        initiator_provenance: dict[str, Any],
        outcome: str,
        response_output: ResponseOutput | None = None,
        fallback_stage: str | None = None,
        fallback_reason: str | None = None,
    ) -> None:
        item: dict[str, Any] = {
            "turn": context.turn,
            "prompt_input": initiator_provenance["prompt_input"],
            "attempts": initiator_provenance["attempts"],
            "outcome": outcome,
            "response_output": {
                "response": turn.response.value,
                "reaction": response_output.reaction if response_output else _reaction_label(turn.response),
                "reply": response_output.reply if response_output else None,
                "provider": self._metadata(response_provider, "unknown"),
            },
            "fallback": {
                "initiator": fallback_stage == "initiator",
                "response": fallback_stage == "response",
                "response_reason": fallback_reason if fallback_stage == "response" else None,
            },
        }
        conversation_act = initiator_provenance.get("conversation_act")
        if isinstance(conversation_act, str):
            # Audit-only metadata; do not add it to ParticipantTurn/Event.
            item["conversation_act"] = conversation_act
        for key in (
            "planner_prompt",
            "utterance_prompt",
            "planner_attempts",
            "utterance_attempts",
            "planner",
            "utterance",
            "selected_output",
            "adopted_plan",
            "final_approach",
            "fallback_stage",
            "fallback_reason",
            "candidate_count",
            "selected_plan_id",
            "token_usage",
            "total_latency",
            "total_latency_ms",
            "decision_source",
            "approach_source",
            "approach_validation_code",
            "history_reference",
            "episode",
            "episode_ended",
            "episode_reaction",
        ):
            if key in initiator_provenance:
                item[key] = copy.deepcopy(initiator_provenance[key])
        # Preserve an adopted LLM decision for both full and hybrid outcomes.
        # The approach_source field distinguishes model wording from the
        # deterministic rule renderer; a whole fallback has no such output.
        if outcome in {"llm", "hybrid"} and initiation is not None:
            initiator_output = {
                "target_id": initiation.target_id,
                "topic": initiation.topic,
                "approach": initiation.approach,
            }
            if isinstance(conversation_act, str):
                initiator_output["act"] = conversation_act
            item["initiator_output"] = copy.deepcopy(initiator_output)
            item["structured_output"] = {
                **initiator_output,
                "response": turn.response.value,
            }
        self._audit_turns.append(item)

    def describe(self) -> dict[str, Any]:
        """Describe the composite and expose its two role providers."""

        source = self._component_description(self.initiator)
        if not source:
            snapshotter = getattr(self.initiator, "audit_snapshot", None)
            try:
                snapshot = snapshotter() if callable(snapshotter) else {}
            except Exception:
                snapshot = {}
            source = dict(snapshot) if isinstance(snapshot, dict) else {}
        result: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "roles": self._roles(),
        }
        for key in ("model", "prompt_version", "generation", "retry_policy", "stages", "candidate_builder", "transport"):
            if key in source:
                result[key] = copy.deepcopy(source[key])
        return result

    def audit_snapshot(self) -> dict[str, Any]:
        """Return a quality-review-compatible composite audit snapshot."""

        source_snapshotter = getattr(self.initiator, "audit_snapshot", None)
        try:
            source = source_snapshotter() if callable(source_snapshotter) else {}
        except Exception:
            source = {}
        source = source if isinstance(source, dict) else {}
        result: dict[str, Any] = {
            "adapter": {"name": self.name, "version": self.version},
            "roles": self._roles(),
            "turns": copy.deepcopy(self._audit_turns),
        }
        for key in ("model", "model_metadata", "prompt_version", "generation", "retry_policy", "stages", "candidate_builder", "transport"):
            if key in source:
                result[key] = copy.deepcopy(source[key])
        summary = source.get("summary")
        if isinstance(summary, dict):
            result["summary"] = copy.deepcopy(summary)
        else:
            result["summary"] = {}
        # Keep summary counters truthful if the responder caused a whole-turn
        # fallback after a successful initiation.
        result["summary"].update(
            {
                "turns": len(self._audit_turns),
                "llm": sum(item["outcome"] == "llm" for item in self._audit_turns),
                "full_llm_wording": sum(item["outcome"] == "llm" for item in self._audit_turns),
                "hybrid": sum(item["outcome"] == "hybrid" for item in self._audit_turns),
                "fallback": sum(item["outcome"] == "fallback" for item in self._audit_turns),
                "whole_fallback": sum(item["outcome"] == "fallback" for item in self._audit_turns),
                "attempts": sum(len(item["attempts"]) for item in self._audit_turns),
            }
        )
        act_counts = {"light_question": 0, "self_disclosure": 0, "invitation": 0, "follow_up": 0}
        for item in self._audit_turns:
            act = item.get("conversation_act")
            if act in act_counts:
                act_counts[act] += 1
        result["summary"]["act_counts"] = act_counts
        return copy.deepcopy(result)

    def propose(self, context: ParticipantContext) -> ParticipantTurn:
        rng_state = context.rng.getstate()
        initiation: Initiation | None = None
        stage = "initiator"
        try:
            initiation = self.initiator.initiate(context)
            if not isinstance(initiation, Initiation):
                raise ValueError("talk initiator must return Initiation")
            if initiation.target_id == context.actor.id:
                raise ValueError("talk initiation cannot target the actor")
            if initiation.target_id not in {candidate.id for candidate in context.candidates}:
                raise ValueError("talk initiation target must be one of the candidates")
            stage = "response"
            raw_response = (
                self.responder.respond_output(context, initiation)
                if callable(getattr(self.responder, "respond_output", None))
                else self.responder.respond(context, initiation)
            )
            response, _, response_output = adapt_response(raw_response)
            result = ParticipantTurn(
                context.turn,
                context.actor.id,
                initiation.target_id,
                initiation.topic,
                initiation.approach,
                response,
            )
            provenance = self._initiator_provenance(context.turn)
            self._record_audit(
                context,
                initiation,
                result,
                self.responder,
                initiator_provenance=provenance,
                outcome=provenance["outcome"],
                response_output=response_output,
            )
            return result
        except Exception:
            if self.fallback is None:
                raise
            context.rng.setstate(rng_state)
            propose = getattr(self.fallback, "propose", None)
            if not callable(propose):
                raise ValueError("fallback must implement propose()")
            result = propose(context)
            provenance = self._initiator_provenance(context.turn)
            fallback_output = getattr(self.fallback, "last_response_output", None)
            self._record_audit(
                context,
                None,
                result,
                self.fallback,
                initiator_provenance=provenance,
                outcome="fallback",
                response_output=(
                    fallback_output if isinstance(fallback_output, ResponseOutput) else None
                ),
                fallback_stage=stage,
                fallback_reason="response_error" if stage == "response" else "initiator_error",
            )
            return result


def _reaction_label(response: ResponseKind) -> str:
    return {
        ResponseKind.POSITIVE: "positive",
        ResponseKind.NEUTRAL: "neutral",
        ResponseKind.MISALIGNED: "misaligned",
    }[response]


class M51SplitParticipantProvider(CompositeParticipantProvider):
    """Rule-backed M5.1 composition with injectable component factories.

    No Ollama import is made here.  An LLM-backed initiator or responder can
    be supplied later either as an instance or through a factory, while the
    default remains fully deterministic and dependency-free.
    """

    name = "m51-split-participant"
    version = "m5_1_split_participant_v1"

    def __init__(
        self,
        initiator: TalkInitiator | None = None,
        responder: ResponseProvider | None = None,
        *,
        initiator_factory: Callable[[], TalkInitiator] | None = None,
        response_provider_factory: Callable[[], ResponseProvider] | None = None,
        fallback: Any | None = None,
    ) -> None:
        if initiator is not None and initiator_factory is not None:
            raise ValueError("pass either initiator or initiator_factory, not both")
        if responder is not None and response_provider_factory is not None:
            raise ValueError("pass either responder or response_provider_factory, not both")
        if initiator_factory is not None:
            if not callable(initiator_factory):
                raise ValueError("initiator_factory must be callable")
            initiator = initiator_factory()
        if response_provider_factory is not None:
            if not callable(response_provider_factory):
                raise ValueError("response_provider_factory must be callable")
            responder = response_provider_factory()
        super().__init__(
            initiator or RuleTalkInitiator(),
            responder or RuleResponseProvider(),
            fallback=fallback,
        )


__all__ = ["CompositeParticipantProvider", "M51SplitParticipantProvider"]
