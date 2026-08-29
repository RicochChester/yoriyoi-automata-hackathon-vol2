"""M11 grounded response selection boundary.

The response wording is produced locally from a small, rule-built candidate
set.  DeepSeek is allowed to select a candidate, but never to write wording
or a reaction.  This keeps the cloud boundary useful for emotional/interaction
strategy while making grounding and reply/reaction coherence deterministic.
"""

from __future__ import annotations

import copy
import json
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .contracts import Initiation, ResponseContext, ResponseOutput
from .deepseek_adapter import DeepSeekAdapterError, DeepSeekConfig, DeepSeekJsonClient


PROMPT_VERSION = "m11_grounded_response_candidates_deepseek_v1"
MAX_CANDIDATES = 5
MAX_OUTPUT_TOKENS = 16
MAX_TEMPERATURE = 0.4
_OUTPUT_KEYS = {"candidate_id"}
_FALLBACK_REPLY = "わかりました。"


class CandidateResponseOutputError(ValueError):
    """Stable semantic validation category for a candidate decision."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class RuleResponseCandidate:
    """A complete safe response whose reaction is fixed before model choice."""

    candidate_id: str
    strategy: str
    reply: str
    reaction: str
    evidence: tuple[str, ...] = ()

    @property
    def safe_reply(self) -> str:
        """Explicit name used at the cloud boundary and in audits/tests."""

        return self.reply

    @property
    def predetermined_reaction(self) -> str:
        return self.reaction

    def projection(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "strategy": self.strategy,
            "safe_reply": self.reply,
            "reaction": self.reaction,
            "evidence": list(self.evidence),
        }


def _elapsed_ms(started: float) -> int:
    return max(0, int(round((time.monotonic() - started) * 1000)))


def _redact(value: Any, context: ResponseContext) -> Any:
    """Keep people pseudonymous even if a configured summary contains them."""

    tokens = (
        (context.responder.name, "返答者"),
        (context.responder.id, "返答者"),
        (context.initiator.name, "話しかけた相手"),
        (context.initiator.id, "話しかけた相手"),
    )
    if isinstance(value, Mapping):
        return {key: _redact(item, context) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_redact(item, context) for item in value]
    if isinstance(value, str):
        result = value
        for token, replacement in sorted(tokens, key=lambda item: len(item[0]), reverse=True):
            result = result.replace(token, replacement)
        return result
    return copy.deepcopy(value)


class RuleResponseCandidateBuilder:
    """Build grounded replies from the narrow response context."""

    name = "rule-response-candidate-builder"
    version = "m11_grounded_response_candidates_v1"

    def build(self, context: ResponseContext, initiation: Initiation) -> tuple[RuleResponseCandidate, ...]:
        if initiation.target_id != context.responder.id:
            raise ValueError("initiation target must match response context responder")
        topic = initiation.topic
        history_topics = tuple(
            dict.fromkeys(event.topic for event in context.pair_events if event.topic)
        )
        has_history_topic = topic in history_topics
        memory_topic = (
            context.online_memory.topic
            if context.online_memory is not None and context.online_memory.topic
            else None
        )
        has_memory_topic = memory_topic == topic
        exact_interest = topic in context.responder.interests
        # The fixed reactions are part of the candidates.  Affect influences
        # which emotional options exist, but never changes a selected reply
        # after the model decision.
        candidates: list[RuleResponseCandidate] = [
            RuleResponseCandidate(
                "c_acknowledge",
                "acknowledge",
                f"そうなんですね。{topic}について、もう少し聞かせてもらえますか？",
                "positive" if context.affect >= 2 else "neutral",
                ("initiation_topic",),
            ),
            RuleResponseCandidate(
                "c_ask_more",
                "ask_more",
                f"{topic}について、気になることをもう少し教えてもらえますか？",
                "positive" if context.affect >= 1 else "neutral",
                ("initiation_topic",),
            ),
        ]
        if exact_interest:
            candidates.append(RuleResponseCandidate(
                "c_interest_bridge",
                "configured_interest_bridge",
                f"私の関心にある{topic}について、もう少し聞かせてもらえますか？",
                "positive" if context.affect >= 0 else "neutral",
                ("configured_responder_interest",),
            ))
        if has_history_topic or has_memory_topic:
            evidence = []
            if has_history_topic:
                evidence.append("actual_pair_history")
            if has_memory_topic:
                evidence.append("actual_online_memory")
            candidates.append(RuleResponseCandidate(
                "c_history_bridge",
                "history_bridge",
                f"前に話した{topic}について、続きがあれば聞かせてもらえますか？",
                "positive" if context.affect >= 0 else "neutral",
                tuple(evidence),
            ))
        # A cautious option is always available so a negative affect/guarded
        # stance can lead to a less committal, potentially misaligned reply.
        candidates.append(RuleResponseCandidate(
            "c_cautious_clarify",
            "cautious_clarify",
            f"{topic}の意味を、もう少し具体的に教えてもらえますか？",
            "misaligned" if context.affect < 0 or context.responder.stance.value == "慎重" else "neutral",
            ("initiation_topic", "affect_or_stance"),
        ))
        return tuple(candidates[:MAX_CANDIDATES])


class DeepSeekGroundedResponseProvider:
    """Let DeepSeek choose among local safe candidate responses."""

    name = "deepseek-grounded-response-provider"
    version = PROMPT_VERSION

    def __init__(
        self,
        config: DeepSeekConfig | None = None,
        *,
        candidate_builder: RuleResponseCandidateBuilder | None = None,
        client: DeepSeekJsonClient | None = None,
        transport: Any | None = None,
    ) -> None:
        if client is not None and (config is not None or transport is not None):
            raise ValueError("client cannot be combined with config or transport")
        self.client = client or DeepSeekJsonClient(config, transport=transport)
        self.config = self.client.config
        self.candidate_builder = candidate_builder or RuleResponseCandidateBuilder()
        self.last_response_output: ResponseOutput | None = None
        self._turns: list[dict[str, Any]] = []

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "model": self.config.model,
            "prompt_version": PROMPT_VERSION,
            "generation": {
                "temperature": min(self.config.temperature, MAX_TEMPERATURE),
                "max_tokens": min(self.config.max_tokens, MAX_OUTPUT_TOKENS),
                "think": False,
            },
            "candidate_builder": {
                "name": self.candidate_builder.name,
                "version": self.candidate_builder.version,
                "max_candidates": MAX_CANDIDATES,
            },
            "retry_policy": {
                "normal_calls_per_turn": 1,
                "max_attempts": 1,
                "retryable": [],
                "semantic_retry": False,
                "max_calls_per_run": self.config.max_calls,
            },
            "transport": self.client.describe(),
        }

    @staticmethod
    def _body(
        context: ResponseContext,
        initiation: Initiation,
        candidates: tuple[RuleResponseCandidate, ...],
        config: DeepSeekConfig,
    ) -> dict[str, Any]:
        history_topics = list(dict.fromkeys(event.topic for event in context.pair_events if event.topic))
        memory = None
        if context.online_memory is not None:
            memory = {
                "kind": context.online_memory.kind,
                "topic": context.online_memory.topic,
                "exchange_count": context.online_memory.exchange_count,
                "summary": context.online_memory.summary,
            }
        projection = _redact({
            "turn": context.turn,
            "responder": {
                "interests": list(context.responder.interests),
                "stance": context.responder.stance.value,
                "affect_to_initiator": context.affect,
            },
            "initiator": {"stance": context.initiator.stance.value},
            "initiation": {"topic": initiation.topic},
            "pair_history": {
                "count": len(context.pair_events),
                "topics": history_topics,
            },
            "online_memory": memory,
            "response_candidates": [item.projection() for item in candidates],
        }, context)
        return {
            "model": config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "候補のcandidate_idを1つだけ選び、JSON {\"candidate_id\":\"...\"} のみ返します。"
                        "候補のsafe_replyとreactionは変更・再生成しません。"
                        "候補から選ぶ際は、responderの現在のaffect、stance、実際のpair_history、"
                        "online_memoryを感情・対話戦略の根拠として考慮します。"
                        "affectが低いときや慎重なstanceでは、必ずpositiveを選ぶ必要はありません。"
                        "configured_interest_bridgeは設定済みの関心、history_bridgeは入力にある実際の履歴だけを根拠にします。"
                        "入力にない安定した関心・習慣・経験・観察・共有感情を推測または追加しません。"
                    ),
                },
                {"role": "user", "content": json.dumps(projection, ensure_ascii=False, separators=(",", ":"))},
            ],
            "stream": False,
            "response_format": {"type": "json_object"},
            "temperature": min(config.temperature, MAX_TEMPERATURE),
            "max_tokens": min(config.max_tokens, MAX_OUTPUT_TOKENS),
            "thinking": {"type": "disabled"},
        }

    @staticmethod
    def _validate_decision(value: Any, candidates: tuple[RuleResponseCandidate, ...]) -> RuleResponseCandidate:
        if not isinstance(value, Mapping) or set(value) != _OUTPUT_KEYS:
            raise CandidateResponseOutputError("missing_or_extra_keys")
        candidate_id = value.get("candidate_id")
        if type(candidate_id) is not str:
            raise CandidateResponseOutputError("invalid_candidate_id")
        selected = next((item for item in candidates if item.candidate_id == candidate_id), None)
        if selected is None:
            raise CandidateResponseOutputError("unknown_candidate_id")
        return selected

    def respond(self, context: ResponseContext, initiation: Initiation) -> ResponseOutput:
        started = time.monotonic()
        candidates: tuple[RuleResponseCandidate, ...] = ()
        before_usage: Mapping[str, Any] = {}
        attempts: list[dict[str, Any]] = []
        selected: RuleResponseCandidate | None = None
        fallback_reason: str | None = None
        try:
            candidates = self.candidate_builder.build(context, initiation)
            if not candidates:
                raise ValueError("no_candidates")
            before_usage = self.client.describe().get("usage", {})
            value = self.client.request_json(self._body(context, initiation, candidates, self.config))
            selected = self._validate_decision(value, candidates)
            attempts.append({"attempt": 1, "status": "success", "elapsed_ms": _elapsed_ms(started)})
            output = ResponseOutput(selected.reply, selected.reaction)
            outcome = "deepseek"
        except DeepSeekAdapterError as exc:
            fallback_reason = exc.category
            attempts.append({"attempt": 1, "status": "fallback", "error_category": exc.category, "elapsed_ms": _elapsed_ms(started)})
            output = ResponseOutput(_FALLBACK_REPLY, "neutral")
            outcome = "fallback"
        except CandidateResponseOutputError as exc:
            fallback_reason = "semantic_invalid"
            attempts.append({"attempt": 1, "status": "fallback", "error_category": "semantic_invalid", "validation_code": exc.code, "elapsed_ms": _elapsed_ms(started)})
            output = ResponseOutput(_FALLBACK_REPLY, "neutral")
            outcome = "fallback"
        except Exception:
            fallback_reason = "context_invalid"
            attempts.append({"attempt": 1, "status": "fallback", "error_category": "context_invalid", "elapsed_ms": _elapsed_ms(started)})
            output = ResponseOutput(_FALLBACK_REPLY, "neutral")
            outcome = "fallback"

        after_usage = self.client.describe().get("usage", {})
        usage = {
            key: max(0, int(after_usage.get(key, 0)) - int(before_usage.get(key, 0)))
            for key in ("prompt_tokens", "completion_tokens", "cache_hit_tokens", "cache_miss_tokens")
        }
        self.last_response_output = output
        self._turns.append({
            "turn": context.turn,
            "candidate_count": len(candidates),
            "selected_candidate_id": selected.candidate_id if selected else None,
            "selected_strategy": selected.strategy if selected else None,
            "outcome": outcome,
            "fallback_reason": fallback_reason,
            "attempts": attempts,
            "reaction": output.reaction,
            "token_usage": usage,
            "total_latency_ms": _elapsed_ms(started),
        })
        return output

    def audit_snapshot(self) -> dict[str, Any]:
        outcomes = Counter(item["outcome"] for item in self._turns)
        reasons = Counter(item["fallback_reason"] for item in self._turns if item["fallback_reason"] is not None)
        return {
            "adapter": {"name": self.name, "version": self.version},
            "model": self.config.model,
            "prompt_version": PROMPT_VERSION,
            "generation": self.describe()["generation"],
            "candidate_builder": self.describe()["candidate_builder"],
            "retry_policy": self.describe()["retry_policy"],
            "transport": self.client.describe(),
            "turns": copy.deepcopy(self._turns),
            "summary": {
                "turns": len(self._turns),
                "deepseek": outcomes.get("deepseek", 0),
                "fallback": outcomes.get("fallback", 0),
                "fallback_reasons": dict(sorted(reasons.items())),
            },
        }


__all__ = [
    "CandidateResponseOutputError",
    "DeepSeekGroundedResponseProvider",
    "MAX_CANDIDATES",
    "MAX_OUTPUT_TOKENS",
    "MAX_TEMPERATURE",
    "PROMPT_VERSION",
    "RuleResponseCandidate",
    "RuleResponseCandidateBuilder",
]
