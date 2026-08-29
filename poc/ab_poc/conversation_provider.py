"""Small split conversation boundary for the M10.6 comparison modes."""

from __future__ import annotations

import copy
from difflib import SequenceMatcher
import unicodedata
from typing import Any

from .contracts import ConversationEpisode, ConversationUtterance, Initiation, ParticipantContext, ResponseContext, ResponseOutput, adapt_response
from .domain import ResponseKind
from .rule_providers import RuleTalkInitiator


_SAFE_FALLBACK_REASONS = {
    "provider_fallback",
    "transport_error",
    "missing_api_key",
    "malformed_response",
    "empty_content",
    "request_too_large",
    "call_budget_exceeded",
    "semantic_invalid",
    "context_invalid",
}


def _is_repeated_utterance(text: str, utterances: list[ConversationUtterance]) -> bool:
    """Detect an exact or near echo without imposing a dialogue template."""

    def normalize(value: str) -> str:
        # NFKC plus removal of whitespace/punctuation catches harmless
        # formatting variation while leaving semantic wording to the model.
        return "".join(
            char for char in unicodedata.normalize("NFKC", value).casefold()
            if not char.isspace() and not unicodedata.category(char).startswith("P")
        )

    candidate = normalize(text)
    if not candidate:
        return True
    return any(
        candidate == normalize(item.text)
        or SequenceMatcher(None, candidate, normalize(item.text)).ratio() >= 0.92
        for item in utterances
    )


class _NeutralResponseFallback:
    name = "neutral-response-fallback"
    version = "m10.6_rule_fallback_v1"

    def respond(self, context: ResponseContext, initiation: Initiation) -> ResponseOutput:
        del context, initiation
        return ResponseOutput("わかりました。", "neutral")


class SplitConversationProvider:
    """Bundle one initiator and responder without changing the old provider API."""

    def __init__(
        self,
        initiator: Any,
        responder: Any,
        *,
        mode: str = "llm_affect",
        initiator_fallback: Any | None = None,
        responder_fallback: Any | None = None,
        narrative_provider: Any | None = None,
    ) -> None:
        if not callable(getattr(initiator, "initiate", None)):
            raise ValueError("initiator must implement initiate()")
        if not callable(getattr(responder, "respond", None)):
            raise ValueError("responder must implement respond()")
        if mode not in {"rule_no_affect", "llm_no_affect", "llm_affect", "deepseek_natural", "deepseek_autonomous"}:
            raise ValueError("conversation mode is invalid")
        self.initiator = initiator
        self.responder = responder
        self.mode = mode
        self.initiator_fallback = initiator_fallback or RuleTalkInitiator()
        self.responder_fallback = responder_fallback or _NeutralResponseFallback()
        # Optional presentation-only layer. The engine never feeds its output
        # back into simulation, evaluation, or metrics.
        self.narrative_provider = narrative_provider
        self._turns: list[dict[str, Any]] = []

    @property
    def affect_enabled(self) -> bool:
        return self.mode in {"llm_affect", "deepseek_natural", "deepseek_autonomous"}

    @property
    def name(self) -> str:
        return "split-conversation-provider"

    @property
    def version(self) -> str:
        return (
            "m17_autonomous_conversation_split_v1"
            if self.mode == "deepseek_autonomous"
            else "m14_natural_conversation_split_v1"
            if self.mode == "deepseek_natural"
            else "m10.6_split_conversation_v1"
        )

    def _turn(self, turn: int) -> dict[str, Any]:
        for item in self._turns:
            if item["turn"] == turn:
                return item
        item = {
            "turn": turn,
            "mode": self.mode,
            "fallback": {"initiator": False, "responder": False},
        }
        self._turns.append(item)
        return item

    @staticmethod
    def _component_fallback(provider: Any, turn: int) -> str | None:
        snapshotter = getattr(provider, "audit_snapshot", None)
        if not callable(snapshotter):
            return None
        try:
            snapshot = snapshotter()
        except Exception:
            return None
        turns = snapshot.get("turns") if isinstance(snapshot, dict) else None
        if not isinstance(turns, (tuple, list)):
            return None
        current = next(
            (item for item in turns if isinstance(item, dict) and item.get("turn") == turn),
            None,
        )
        if not isinstance(current, dict) or current.get("outcome") != "fallback":
            return None
        reason = current.get("fallback_reason")
        return reason if isinstance(reason, str) and reason in _SAFE_FALLBACK_REASONS else "provider_fallback"

    def initiate(self, context: ParticipantContext) -> Initiation:
        item = self._turn(context.turn)
        try:
            result = self.initiator.initiate(context)
            if not isinstance(result, Initiation):
                raise ValueError("talk initiator must return Initiation")
            reason = self._component_fallback(self.initiator, context.turn)
            if reason is not None:
                item["fallback"]["initiator"] = True
                item["initiator_reason"] = reason
            return result
        except Exception:
            result = self.initiator_fallback.initiate(context)
            item["fallback"]["initiator"] = True
            item["initiator_reason"] = "initiator_error"
            return result

    def respond(self, context: ResponseContext, initiation: Initiation) -> ResponseOutput | ResponseKind:
        item = self._turn(context.turn)
        try:
            result = self.responder.respond(context, initiation)
            if not isinstance(result, (ResponseOutput, ResponseKind)):
                raise ValueError("response provider returned an invalid response")
            reason = self._component_fallback(self.responder, context.turn)
            if reason is not None:
                item["fallback"]["responder"] = True
                item["responder_reason"] = reason
            return result
        except Exception:
            item["fallback"]["responder"] = True
            item["responder_reason"] = "response_error"
            return self.responder_fallback.respond(context, initiation)

    def respond_episode(
        self,
        participant_context: ParticipantContext,
        response_context: ResponseContext,
        initiation: Initiation,
    ) -> ResponseOutput | ResponseKind:
        """Optionally orchestrate role-owned follow-ups, capped at four.

        The first two utterances always come from the existing initiate/respond
        APIs. Continuations are requested from the owning provider only, and
        each provider receives the transcript as a DTO rather than another
        participant's persona.
        """

        raw = self.respond(response_context, initiation)
        reaction, reply, output = adapt_response(raw)
        if output is None or output.reply is None or output.continue_conversation is None:
            return raw
        utterances = [
            ConversationUtterance(1, participant_context.actor.id, initiation.approach),
            ConversationUtterance(2, response_context.responder.id, output.reply),
        ]
        should_continue = output.continue_conversation
        ended = not should_continue
        end_reason = "responder_end"
        while should_continue and len(utterances) < 4:
            episode = ConversationEpisode(tuple(utterances), ended=False, reaction=output.reaction)
            if len(utterances) % 2 == 0:
                method = getattr(self.initiator, "continue_episode", None)
                if not callable(method):
                    ended = True
                    end_reason = "provider_unavailable"
                    break
                try:
                    text, should_continue = method(participant_context, initiation, episode)
                except Exception:
                    ended = True
                    end_reason = "provider_unavailable"
                    break
                speaker = participant_context.actor.id
            else:
                method = getattr(self.responder, "continue_episode", None)
                if not callable(method):
                    ended = True
                    end_reason = "provider_unavailable"
                    break
                try:
                    text, should_continue = method(response_context, initiation, episode)
                except Exception:
                    ended = True
                    end_reason = "provider_unavailable"
                    break
                speaker = response_context.responder.id
            if type(text) is not str or not text.strip() or text != text.strip():
                ended = True
                end_reason = "provider_unavailable"
                break
            # A continuation that contributes no new wording is not useful
            # evidence of a live exchange. Drop it at this boundary and end
            # naturally, while preserving the valid transcript so far.
            if _is_repeated_utterance(text, utterances):
                ended = True
                end_reason = "initiator_end" if speaker == participant_context.actor.id else "responder_end"
                break
            utterances.append(ConversationUtterance(len(utterances) + 1, speaker, text))
            ended = not should_continue
            if ended:
                end_reason = "initiator_end" if speaker == participant_context.actor.id else "responder_end"
        if should_continue and len(utterances) >= 4:
            ended = True
            end_reason = "utterance_limit"
        episode = ConversationEpisode(
            tuple(utterances), ended=ended, reaction=output.reaction, end_reason=end_reason
        )
        return ResponseOutput(output.reply, output.reaction, episode=episode, continue_conversation=output.continue_conversation)

    def audit_snapshot(self) -> dict[str, Any]:
        turns = copy.deepcopy(self._turns)
        result = {
            "adapter": {"name": self.name, "version": self.version},
            "mode": self.mode,
            "roles": {
                "initiator": {
                    "name": str(getattr(self.initiator, "name", type(self.initiator).__name__)),
                    "version": str(getattr(self.initiator, "version", "unknown")),
                },
                "responder": {
                    "name": str(getattr(self.responder, "name", type(self.responder).__name__)),
                    "version": str(getattr(self.responder, "version", "unknown")),
                },
            },
            "turns": turns,
            "summary": {
                "turns": len(turns),
                "initiator_fallback": sum(item["fallback"]["initiator"] for item in turns),
                "responder_fallback": sum(item["fallback"]["responder"] for item in turns),
            },
        }
        if self.mode == "deepseek_natural":
            result["prompt_version"] = "m14_natural_conversation_deepseek_v1"
        if self.mode == "deepseek_autonomous":
            result["prompt_version"] = "m19_deepseek_autonomous_episode_v1"
        if self.narrative_provider is not None:
            result["narrative_provider"] = {
                "name": str(getattr(self.narrative_provider, "name", type(self.narrative_provider).__name__)),
                "version": str(getattr(self.narrative_provider, "version", "unknown")),
            }
            describe = getattr(self.narrative_provider, "describe", None)
            if callable(describe):
                try:
                    result["narrative_provider_detail"] = copy.deepcopy(describe())
                except Exception:
                    pass
        return result


__all__ = ["SplitConversationProvider"]
