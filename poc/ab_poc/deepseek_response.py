"""Fake-transport-ready DeepSeek response boundary for M10.5.

The provider accepts a narrow ``ResponseContext`` and returns only a short
reply plus a three-valued reaction.  The transport client owns timeout,
request-size, and call-budget enforcement; this adapter owns semantic
validation and a deterministic neutral fallback.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from collections.abc import Mapping
from typing import Any

from .contracts import Initiation, ResponseContext, ResponseOutput
from .deepseek_adapter import (
    DeepSeekAdapterError,
    DeepSeekConfig,
    DeepSeekJsonClient,
)


PROMPT_VERSION = "m10_5_deepseek_response_v2"
MAX_REPLY_LENGTH = 160
_OUTPUT_KEYS = {"reply", "reaction"}
_REACTIONS = {"positive", "neutral", "misaligned"}
_FALLBACK_REPLY = "わかりました。"


class ResponseOutputError(ValueError):
    """Stable semantic validation category for model response JSON."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _redact(value: Any, context: ResponseContext) -> Any:
    """Remove participant names and internal IDs from nested payload data."""

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


def _response_projection(context: ResponseContext, initiation: Initiation) -> dict[str, Any]:
    if initiation.target_id != context.responder.id:
        raise ValueError("initiation target must match response context responder")
    history = []
    for event in context.pair_events:
        history.append({
            "turn": event.turn,
            "speaker_role": "responder" if event.actor == context.responder.id else "initiator",
            "topic": event.topic,
            "approach": event.approach,
            "response": event.response.value,
        })
    memory = None
    if context.online_memory is not None:
        memory = {
            "kind": context.online_memory.kind,
            "topic": context.online_memory.topic,
            "exchange_count": context.online_memory.exchange_count,
            "summary": context.online_memory.summary,
        }
    return _redact(
        {
            "turn": context.turn,
            "responder": {
                # Only the configured fictional interest list is a stable
                # responder profile fact exposed to the response model.
                "interests": list(context.responder.interests),
                "stance": context.responder.stance.value,
                "affect_to_initiator": context.affect,
            },
            "initiator": {"stance": context.initiator.stance.value},
            "initiation": {
                "topic": initiation.topic,
                "approach": initiation.approach,
            },
            "pair_history": history,
            "online_memory": memory,
        },
        context,
    )


def _validate_output(value: Any) -> ResponseOutput:
    if not isinstance(value, Mapping) or set(value) != _OUTPUT_KEYS:
        raise ResponseOutputError("missing_or_extra_keys")
    reply = value.get("reply")
    reaction = value.get("reaction")
    if type(reply) is not str or not reply.strip() or reply != reply.strip():
        raise ResponseOutputError("invalid_reply")
    if len(reply) > MAX_REPLY_LENGTH:
        raise ResponseOutputError("reply_too_long")
    if type(reaction) is not str or reaction not in _REACTIONS:
        raise ResponseOutputError("invalid_reaction")
    return ResponseOutput(reply, reaction)


class DeepSeekResponseProvider:
    """Generate one strictly validated response through a bounded client."""

    name = "deepseek-response-provider"
    version = PROMPT_VERSION

    def __init__(
        self,
        config: DeepSeekConfig | None = None,
        *,
        client: DeepSeekJsonClient | None = None,
        transport: Any | None = None,
    ) -> None:
        if client is not None and (config is not None or transport is not None):
            raise ValueError("client cannot be combined with config or transport")
        self.client = client or DeepSeekJsonClient(config, transport=transport)
        self.config = self.client.config
        self.last_response_output: ResponseOutput | None = None
        self._turns: list[dict[str, Any]] = []

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "model": self.config.model,
            "prompt_version": PROMPT_VERSION,
            "generation": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "think": False,
            },
            "reply_max_length": MAX_REPLY_LENGTH,
            "transport": self.client.describe(),
        }

    @staticmethod
    def _body(context: ResponseContext, initiation: Initiation, config: DeepSeekConfig) -> dict[str, Any]:
        return {
            "model": config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "JSON objectを厳密に返します。キーはreplyとreactionだけです。"
                        "replyは短い自然な返答、reactionはpositive、neutral、misalignedのいずれかです。"
                        "responder.interestsは設定された架空プロフィールの関心だけです。"
                        "responderとinitiatorのstanceは口調とreactionの調整だけに使い、"
                        "関心・習慣・経験・観察などの事実源にはしません。"
                        "入力にない安定した関心・習慣・経験・観察を自分のものとして発明しません。"
                        "話題が設定された関心の外でも、その話題について会話できますが、"
                        "それを自分の恒常的な関心として主張しません。"
                        "入力にない事実・記憶・相手の感情も作りません。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        _response_projection(context, initiation),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            "stream": False,
            "response_format": {"type": "json_object"},
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "thinking": {"type": "disabled"},
        }

    def respond(self, context: ResponseContext, initiation: Initiation) -> ResponseOutput:
        try:
            body = self._body(context, initiation, self.config)
            value = self.client.request_json(body)
            output = _validate_output(value)
        except DeepSeekAdapterError as exc:
            return self._fallback(context, exc.category)
        except ResponseOutputError as exc:
            return self._fallback(context, "semantic_invalid", validation_code=exc.code)
        except Exception:
            return self._fallback(context, "context_invalid")
        self.last_response_output = output
        self._turns.append({
            "turn": context.turn,
            "outcome": "deepseek",
            "attempts": [{"attempt": 1, "status": "success"}],
            "fallback_reason": None,
            "reaction": output.reaction,
        })
        return output

    def _fallback(
        self,
        context: ResponseContext,
        reason: str,
        *,
        validation_code: str | None = None,
    ) -> ResponseOutput:
        output = ResponseOutput(_FALLBACK_REPLY, "neutral")
        self.last_response_output = output
        attempt: dict[str, Any] = {
            "attempt": 1,
            "status": "fallback",
            "error_category": reason,
        }
        if validation_code is not None:
            attempt["validation_code"] = validation_code
        self._turns.append({
            "turn": context.turn,
            "outcome": "fallback",
            "attempts": [attempt],
            "fallback_reason": reason,
            "reaction": output.reaction,
        })
        return output

    def audit_snapshot(self) -> dict[str, Any]:
        outcomes = Counter(item["outcome"] for item in self._turns)
        reasons = Counter(
            item["fallback_reason"]
            for item in self._turns
            if item["fallback_reason"] is not None
        )
        return {
            "adapter": {"name": self.name, "version": self.version},
            "prompt_version": PROMPT_VERSION,
            "transport": self.client.describe(),
            "turns": copy.deepcopy(self._turns),
            "summary": {
                "turns": len(self._turns),
                "deepseek": outcomes.get("deepseek", 0),
                "fallback": outcomes.get("fallback", 0),
                "fallback_reasons": dict(sorted(reasons.items())),
            },
        }


NATURAL_PROMPT_VERSION = "m14_natural_response_deepseek_v1"


class DeepSeekNaturalResponseProvider(DeepSeekResponseProvider):
    """M14 response provider with participant-only persona and dialogue input."""

    name = "deepseek-natural-response-provider"
    version = NATURAL_PROMPT_VERSION

    @staticmethod
    def _body(context: ResponseContext, initiation: Initiation, config: DeepSeekConfig) -> dict[str, Any]:
        body = DeepSeekResponseProvider._body(context, initiation, config)
        projection = _response_projection(context, initiation)
        projection["responder"]["persona"] = context.responder.persona
        projection["dialogue_history"] = [
            {
                "turn": item.turn,
                "speaker_role": "responder" if item.actor == context.responder.id else "initiator",
                "topic": item.topic,
                "approach": item.approach,
                "approach_speaker_role": "actor",
                "reply": item.reply,
                "reply_speaker_role": "target",
                "reaction": item.reaction,
            }
            for item in context.dialogue_history[-2:]
        ]
        projection = _redact(projection, context)
        body["messages"][0]["content"] += (
            "personaは話し方の傾向です。全項目を毎回言わず、文言をそのまま自己紹介せず、実際の直近発言・履歴を優先します。"
            "personaのpurposeとinterestsは経歴・経験の事実ではありません。"
            "入力にない過去・現在の具体的出来事、職業経験、習慣、実績を本人の事実として作りません。"
            "新しいアイデアは「〜なら面白そう」「〜してみたい」「〜はどうでしょう」のような希望・仮定・提案として表現します。"
            "実在する地域イベントや場所が起きたと断定しません。実際の履歴にある内容だけを引き継ぎます。"
        )
        body["messages"][1]["content"] = json.dumps(
            projection, ensure_ascii=False, separators=(",", ":")
        )
        return body

    def describe(self) -> dict[str, Any]:
        result = super().describe()
        result.update({"name": self.name, "version": self.version, "prompt_version": NATURAL_PROMPT_VERSION})
        return result

    def audit_snapshot(self) -> dict[str, Any]:
        result = super().audit_snapshot()
        result["adapter"] = {"name": self.name, "version": self.version}
        result["prompt_version"] = NATURAL_PROMPT_VERSION
        return result


__all__ = ["DeepSeekResponseProvider", "DeepSeekNaturalResponseProvider", "MAX_REPLY_LENGTH", "PROMPT_VERSION", "NATURAL_PROMPT_VERSION", "ResponseOutputError"]
