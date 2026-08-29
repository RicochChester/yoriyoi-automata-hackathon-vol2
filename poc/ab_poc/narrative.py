"""Optional, evaluation-independent narrative of one encounter.

Only the actual dialogue log crosses this boundary.  Narrative output is
presentation metadata and is never read by the simulation or metrics code.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import DialogueTurn
from .deepseek_adapter import (
    DeepSeekAdapterError,
    DeepSeekCallBudgetExceeded,
    DeepSeekConfig,
    DeepSeekJsonClient,
)


NARRATIVE_PROMPT_VERSION = "m15_encounter_narrative_deepseek_v1"
MAX_TITLE_LENGTH = 80
MAX_SUMMARY_LENGTH = 300
MAX_MOMENT_LENGTH = 160
MAX_MOMENTS = 5
MAX_ENDING_LENGTH = 200
_OUTPUT_KEYS = {"title", "summary", "moments", "ending"}
_SAFE_ADAPTER_REASONS = {
    "empty_content",
    "malformed_response",
    "missing_api_key",
    "request_too_large",
    "transport_error",
}


class NarrativeOutputError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _fallback_reason(error: Exception) -> str:
    if isinstance(error, NarrativeOutputError):
        return error.code
    if isinstance(error, DeepSeekCallBudgetExceeded):
        return "call_budget"
    if isinstance(error, DeepSeekAdapterError):
        code = error.args[0] if error.args else None
        if isinstance(code, str) and code in _SAFE_ADAPTER_REASONS:
            return code
    return "provider_error"


def _log_payload(log: Sequence[DialogueTurn]) -> list[dict[str, Any]]:
    return [
        {
            "turn": item.turn,
            "actor": item.actor,
            "target": item.target,
            "topic": item.topic,
            "approach": item.approach,
            "reply": item.reply,
            "reaction": item.reaction,
        }
        for item in log
        if isinstance(item, DialogueTurn)
    ]


def _validate(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _OUTPUT_KEYS:
        raise NarrativeOutputError("missing_or_extra_keys")
    result: dict[str, Any] = {}
    for key, limit in (("title", MAX_TITLE_LENGTH), ("summary", MAX_SUMMARY_LENGTH), ("ending", MAX_ENDING_LENGTH)):
        text = value.get(key)
        if type(text) is not str or not text.strip() or text != text.strip():
            raise NarrativeOutputError(f"invalid_{key}")
        if len(text) > limit:
            raise NarrativeOutputError(f"{key}_too_long")
        result[key] = text
    moments = value.get("moments")
    if not isinstance(moments, list) or not moments or len(moments) > MAX_MOMENTS:
        raise NarrativeOutputError("invalid_moments")
    clean_moments = []
    for moment in moments:
        if type(moment) is str:
            if not moment.strip() or moment != moment.strip() or len(moment) > MAX_MOMENT_LENGTH:
                raise NarrativeOutputError("invalid_moment")
            clean_moments.append(moment)
            continue
        if isinstance(moment, Mapping):
            # Some compatible models return a turn object despite the schema
            # instruction. Normalize only fields that are actual dialogue-log
            # fields; actor/target/reaction/evaluation and unknown keys never
            # enter the narrative text.
            parts: list[str] = []
            turn = moment.get("turn")
            if isinstance(turn, int) and not isinstance(turn, bool):
                parts.append(f"{turn}回目")
            topic = moment.get("topic")
            approach = moment.get("approach")
            reply = moment.get("reply")
            if isinstance(topic, str) and topic.strip():
                parts.append(topic.strip())
            if isinstance(approach, str) and approach.strip():
                parts.append(approach.strip())
            if isinstance(reply, str) and reply.strip():
                parts.append(reply.strip())
            if not parts:
                raise NarrativeOutputError("invalid_moment")
            normalized = "：".join((parts[0], " / ".join(parts[1:]))) if len(parts) > 1 else parts[0]
            normalized = normalized.strip()[:MAX_MOMENT_LENGTH].rstrip()
            if not normalized:
                raise NarrativeOutputError("invalid_moment")
            clean_moments.append(normalized)
            continue
        raise NarrativeOutputError("invalid_moment")
    result["moments"] = clean_moments
    return result


class LocalNarrativeProvider:
    """Deterministic narrative fallback based solely on actual dialogue."""

    name = "local-encounter-narrative"
    version = "m15_local_narrative_v1"

    def narrative_metadata(self) -> dict[str, str]:
        """Return presentation-only provenance without exposing raw errors."""

        return {"source": "local"}

    def generate(self, dialogue: Sequence[DialogueTurn]) -> dict[str, Any]:
        log = tuple(dialogue)
        topics = [item.topic for item in log if item.topic]
        unique_topics = list(dict.fromkeys(topics))
        first_topic = unique_topics[0] if unique_topics else "出会い"
        moments = [
            f"{item.topic}について、{item.approach}"
            for item in log[:MAX_MOMENTS]
        ] or ["短い会話が始まりました。"]
        return {
            "title": f"{first_topic}から始まる出会い",
            "summary": f"{len(log)}回の会話で、{', '.join(unique_topics[:3]) or '言葉'}をめぐるやり取りが生まれました。",
            "moments": moments,
            "ending": "会話の余韻が残りました。",
        }


class DeepSeekNarrativeProvider:
    """Generate narrative presentation fields with at most one cloud call."""

    name = "deepseek-encounter-narrative"
    version = NARRATIVE_PROMPT_VERSION

    def __init__(self, config: DeepSeekConfig | None = None, *, client: DeepSeekJsonClient | None = None, transport: Any | None = None) -> None:
        if client is not None and (config is not None or transport is not None):
            raise ValueError("client cannot be combined with config or transport")
        self.client = client or DeepSeekJsonClient(config, transport=transport)
        self.config = self.client.config
        self._calls = 0
        self._fallbacks = 0
        self._generated: dict[str, Any] | None = None
        self._last_source: str | None = None
        self._last_fallback_reason: str | None = None

    @staticmethod
    def _body(dialogue: Sequence[DialogueTurn], config: DeepSeekConfig) -> dict[str, Any]:
        return {
            "model": config.model,
            "messages": [
                {"role": "system", "content": (
                    "実際の会話ログだけをもとに、出会いのナラティブをJSONで作ります。"
                    "キーはtitle, summary, moments, endingだけです。momentsは1〜5個の短い文字列だけの配列で、objectは禁止です。"
                    "例: {\"moments\":[\"技術について話し、次の一歩を考えました。\"]}。"
                    "ログにない人物関係・記憶・出来事・評価を追加せず、会話の話題、実際の発話、reply、reactionの範囲で短くまとめます。"
                )},
                {"role": "user", "content": json.dumps({"dialogue": _log_payload(dialogue)}, ensure_ascii=False, separators=(",", ":"))},
            ],
            "stream": False,
            "response_format": {"type": "json_object"},
            "temperature": config.temperature,
            # Narrative JSON has four fields and may need the full adapter
            # allowance even when conversational turns use a smaller budget.
            "max_tokens": 128,
            "thinking": {"type": "disabled"},
        }

    def generate(self, dialogue: Sequence[DialogueTurn]) -> dict[str, Any]:
        # A run invokes this boundary once per condition. Cache the first
        # result as a defensive call-budget guard if a consumer retries.
        if self._generated is not None:
            return copy.deepcopy(self._generated)
        try:
            if not isinstance(dialogue, Sequence):
                raise NarrativeOutputError("invalid_dialogue")
            self._calls += 1
            value = self.client.request_json(self._body(dialogue, self.config))
            self._generated = _validate(value)
            self._last_source = "deepseek"
            self._last_fallback_reason = None
            return copy.deepcopy(self._generated)
        except Exception as exc:
            self._fallbacks += 1
            self._generated = LocalNarrativeProvider().generate(dialogue)
            self._last_source = "local_fallback"
            # Keep only a stable, non-sensitive classification. Never retain
            # exception text because adapters may include request details.
            self._last_fallback_reason = _fallback_reason(exc)
            return copy.deepcopy(self._generated)

    def narrative_metadata(self) -> dict[str, str]:
        """Return the source of the cached narrative, never raw API details."""

        source = self._last_source or "deepseek"
        metadata = {"source": source}
        if source == "local_fallback" and self._last_fallback_reason:
            metadata["fallback_reason"] = self._last_fallback_reason
        return metadata

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "prompt_version": NARRATIVE_PROMPT_VERSION, "transport": self.client.describe(), "call_count": self._calls, "fallback_count": self._fallbacks}


__all__ = [
    "DeepSeekNarrativeProvider",
    "LocalNarrativeProvider",
    "NarrativeOutputError",
    "NARRATIVE_PROMPT_VERSION",
]
