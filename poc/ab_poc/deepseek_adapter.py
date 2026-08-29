"""DeepSeek utterance adapter for the M6.11 two-stage boundary.

The planner remains local (Ollama).  This module only implements the
``UtteranceGenerator`` protocol and sends a minimal, pseudonymous projection
to DeepSeek's OpenAI-compatible Chat Completions endpoint.  It deliberately
does not retain request/response envelopes, API keys, or exception text.
"""

from __future__ import annotations

import copy
import json
import math
import os
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from urllib.parse import urlparse
from typing import Any

from .contracts import ParticipantContext
from .two_stage_initiator import ConversationPlan, _utterance_projection


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_PROMPT_VERSION = "m8_deepseek_utterance_v1"
DEFAULT_MAX_CALLS = 24  # Upper bound for a 12-turn run including continuations.
DEFAULT_MAX_REQUEST_BYTES = 16_384


@dataclass(frozen=True)
class DeepSeekConfig:
    """Validated runtime settings for the DeepSeek utterance adapter.

    ``api_key`` is intentionally not included in ``describe`` or any audit
    object.  With the default constructor it is read from the environment at
    adapter construction time; tests may inject a fake key without making a
    network call.
    """

    api_key: str | None = field(default=None, repr=False)
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    model: str = DEFAULT_DEEPSEEK_MODEL
    timeout: float = 10.0
    temperature: float = 0.2
    max_tokens: int = 64
    max_calls: int = DEFAULT_MAX_CALLS
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES

    def __post_init__(self) -> None:
        key = self.api_key
        if key is not None and (not isinstance(key, str) or not key.strip()):
            object.__setattr__(self, "api_key", None)
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("base_url must be a non-empty URL")
        parsed = urlparse(self.base_url)
        if parsed.scheme != "https" or parsed.hostname != "api.deepseek.com" or parsed.port is not None:
            raise ValueError("base_url must be the official DeepSeek URL")
        if parsed.query or parsed.fragment or parsed.path not in ("", "/"):
            raise ValueError("base_url must be a URL root")
        if parsed.username or parsed.password:
            raise ValueError("base_url must not contain credentials")
        if self.model != DEFAULT_DEEPSEEK_MODEL:
            raise ValueError("model must be deepseek-v4-flash")
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)):
            raise ValueError("timeout must be a positive finite number")
        if not math.isfinite(float(self.timeout)) or not 0 < float(self.timeout) <= 120:
            raise ValueError("timeout must be finite and in the range (0, 120]")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise ValueError("temperature must be a finite non-negative number")
        if not math.isfinite(float(self.temperature)) or not 0 <= float(self.temperature) <= 2:
            raise ValueError("temperature must be finite and in the range 0..2")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or not 1 <= self.max_tokens <= 128:
            raise ValueError("max_tokens must be an integer in the range 1..128")
        if isinstance(self.max_calls, bool) or not isinstance(self.max_calls, int) or not 1 <= self.max_calls <= 24:
            raise ValueError("max_calls must be an integer in the range 1..24")
        if isinstance(self.max_request_bytes, bool) or not isinstance(self.max_request_bytes, int) or not 1024 <= self.max_request_bytes <= 131_072:
            raise ValueError("max_request_bytes must be an integer in the range 1024..131072")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "DeepSeekConfig":
        env = os.environ if environ is None else environ

        def value(name: str, default: str, legacy_name: str | None = None) -> str:
            if name in env:
                return env[name]
            if legacy_name and legacy_name in env:
                return env[legacy_name]
            return default

        try:
            # The namespaced variable is canonical.  The short spelling is a
            # compatibility fallback only and never gets written anywhere.
            api_key = env.get("YORIYOI_DEEPSEEK_API_KEY") or env.get("DEEPSEEK_API_KEY")
            return cls(
                api_key=api_key,
                base_url=value("YORIYOI_DEEPSEEK_BASE_URL", cls.base_url, "DEEPSEEK_BASE_URL"),
                model=value("YORIYOI_DEEPSEEK_MODEL", cls.model, "DEEPSEEK_MODEL"),
                timeout=float(value("YORIYOI_DEEPSEEK_TIMEOUT", str(cls.timeout), "DEEPSEEK_TIMEOUT")),
                temperature=float(value("YORIYOI_DEEPSEEK_TEMPERATURE", str(cls.temperature), "DEEPSEEK_TEMPERATURE")),
                max_tokens=int(value("YORIYOI_DEEPSEEK_MAX_TOKENS", str(cls.max_tokens), "DEEPSEEK_MAX_TOKENS")),
                max_calls=int(value("YORIYOI_DEEPSEEK_MAX_CALLS", str(cls.max_calls), "DEEPSEEK_MAX_CALLS")),
                max_request_bytes=int(value("YORIYOI_DEEPSEEK_MAX_REQUEST_BYTES", str(cls.max_request_bytes), "DEEPSEEK_MAX_REQUEST_BYTES")),
            )
        except (TypeError, ValueError) as exc:
            # Do not expose the offending value; callers only need a stable
            # configuration failure category.
            raise ValueError("invalid YORIYOI_DEEPSEEK_* configuration") from exc


def load_deepseek_config(environ: Mapping[str, str] | None = None) -> DeepSeekConfig:
    return DeepSeekConfig.from_env(environ)


def redact_known_people(value: Any, context: ParticipantContext, target_id: str) -> Any:
    """Recursively replace every known participant name/ID in JSON-like data."""

    tokens: list[tuple[str, str]] = []
    for person in (context.actor, *context.candidates):
        replacement = "相手" if person.id == target_id else "参加者"
        for token in (person.name, person.id):
            if token:
                tokens.append((token, replacement))

    def redact(child: Any) -> Any:
        if isinstance(child, Mapping):
            return {key: redact(item) for key, item in child.items()}
        if isinstance(child, list):
            return [redact(item) for item in child]
        if isinstance(child, tuple):
            return [redact(item) for item in child]
        if isinstance(child, str):
            redacted = child
            for token, replacement in sorted(tokens, key=lambda item: len(item[0]), reverse=True):
                redacted = redacted.replace(token, replacement)
            return redacted
        return copy.deepcopy(child)

    return redact(value)


def _pseudonymous_projection(
    context: ParticipantContext,
    plan: ConversationPlan,
    projection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the smallest utterance input, excluding real/persona names and IDs."""

    source = dict(projection or _utterance_projection(context, plan))

    actor = source.get("actor") if isinstance(source.get("actor"), Mapping) else {}
    target = source.get("selected_target") if isinstance(source.get("selected_target"), Mapping) else {}
    grounding = source.get("topic_grounding") if isinstance(source.get("topic_grounding"), Mapping) else {}
    latest = source.get("latest_pair_history") if isinstance(source.get("latest_pair_history"), Mapping) else None
    latest_safe = None
    if latest is not None:
        latest_safe = {
            key: copy.deepcopy(latest[key])
            for key in ("turn", "topic", "approach", "act")
            if key in latest
        }
    recent = source.get("recent_approaches")
    recent_safe = []
    if isinstance(recent, (list, tuple)):
        for item in recent[-2:]:
            if isinstance(item, Mapping):
                recent_safe.append({
                    key: copy.deepcopy(item[key])
                    for key in ("turn", "topic", "approach")
                    if key in item
                })
    safe_projection = {
        "turn": source.get("turn"),
        "actor": {"stance": copy.deepcopy(actor.get("stance"))},
        "selected_plan": {"topic": plan.topic, "act": plan.act},
        "selected_target": {
            "interests": copy.deepcopy(target.get("interests", [])),
            "stance": copy.deepcopy(target.get("stance")),
        },
        "topic_grounding": {
            "topic": plan.topic,
            "covered_topics": copy.deepcopy(grounding.get("covered_topics", [])),
            "topic_options": copy.deepcopy(grounding.get("topic_options", [])),
        },
        "latest_pair_history": latest_safe,
        "recent_approaches": recent_safe,
        "act_definitions": copy.deepcopy(source.get("act_definitions", {})),
    }

    # Redact every nested string, not just conversation wording. This keeps
    # accidental names/IDs in custom stance, interest, or grounding data from
    # crossing the cloud boundary.
    return redact_known_people(safe_projection, context, plan.target_id)


class DeepSeekAdapterError(RuntimeError):
    """Stable transport/provider error with no provider content in its text."""

    def __init__(self, category: str = "provider_error") -> None:
        self.category = category
        super().__init__(category)


class DeepSeekCallBudgetExceeded(DeepSeekAdapterError):
    def __init__(self) -> None:
        super().__init__("call_budget_exceeded")


def _safe_usage(usage: Any) -> dict[str, int]:
    """Keep only numeric token counters from a provider usage object."""

    usage = usage if isinstance(usage, Mapping) else {}

    def number(*names: str) -> int:
        for name in names:
            value = usage.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if math.isfinite(float(value)) and value >= 0:
                return int(value)
        return 0

    return {
        "prompt_tokens": number("prompt_tokens", "input_tokens"),
        "completion_tokens": number("completion_tokens", "output_tokens"),
        "cache_hit_tokens": number("prompt_cache_hit_tokens", "cache_hit_tokens"),
        "cache_miss_tokens": number("prompt_cache_miss_tokens", "cache_miss_tokens"),
    }


def _decode_chat_response_with_usage(raw: bytes | str) -> tuple[dict[str, Any], dict[str, int]]:
    """Decode only the structured message content; discard the envelope."""

    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        envelope = json.loads(raw)
        choices = envelope.get("choices")
        message = choices[0].get("message") if isinstance(choices, list) and choices else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise DeepSeekAdapterError("empty_content")
        parsed = json.loads(content.strip())
        if not isinstance(parsed, dict):
            raise DeepSeekAdapterError("malformed_response")
        return parsed, _safe_usage(envelope.get("usage"))
    except DeepSeekAdapterError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError, IndexError, TypeError, KeyError):
        raise DeepSeekAdapterError("malformed_response")


def _decode_chat_response(raw: bytes | str) -> dict[str, Any]:
    """Compatibility helper returning only structured message content."""

    parsed, _ = _decode_chat_response_with_usage(raw)
    return parsed


class DeepSeekUtteranceGenerator:
    """M8 cloud utterance generator; planner and fallback remain local/rule-based."""

    name = "deepseek-utterance-generator"
    version = DEEPSEEK_PROMPT_VERSION

    def __init__(
        self,
        config: DeepSeekConfig | None = None,
        *,
        transport: Callable[..., Any] | None = None,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        if transport is not None and opener is not None:
            raise ValueError("pass either transport or opener, not both")
        self.config = config or load_deepseek_config()
        self._transport = transport or opener
        self._calls = 0
        self._successes = 0
        self._errors = 0
        self._usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 0,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(model={self.config.model!r}, "
            f"calls={self._calls}, successes={self._successes}, errors={self._errors})"
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "model": self.config.model,
            "backend": "deepseek-chat-completions",
            "thinking": False,
            "max_tokens": self.config.max_tokens,
            "max_calls": self.config.max_calls,
            "max_request_bytes": self.config.max_request_bytes,
            "api_key_configured": bool(self.config.api_key),
            "call_count": self._calls,
            "success_count": self._successes,
            "error_count": self._errors,
            "usage": copy.deepcopy(self._usage),
        }

    def generate(
        self,
        context: ParticipantContext,
        plan: ConversationPlan,
        *,
        projection: Mapping[str, Any] | None = None,
        attempt: int = 1,
    ) -> dict[str, Any]:
        body = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "JSONで出力します。例: {\"approach\":\"...\"}。固定されたselected_planに従い、approachキーだけを持つJSON objectを返します。"
                        "target/topic/actや説明は出力せず、短い自然な日本語の一文にします。"
                        "入力にない事実・記憶・観察は作りません。"
                    ),
                },
                {"role": "user", "content": json.dumps(
                    _pseudonymous_projection(context, plan, projection),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )},
            ],
            "stream": False,
            "response_format": {"type": "json_object"},
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "thinking": {"type": "disabled"},
        }
        return self.request_json(body)

    def request_json(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """Send one bounded JSON-mode request without retaining its envelope."""

        try:
            if self._calls >= self.config.max_calls:
                raise DeepSeekCallBudgetExceeded()
            if not self.config.api_key:
                raise DeepSeekAdapterError("missing_api_key")
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            if len(payload) > self.config.max_request_bytes:
                raise DeepSeekAdapterError("request_too_large")
            request = urllib.request.Request(
                self.config.base_url.rstrip("/") + "/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + self.config.api_key,
                },
                method="POST",
            )
            opener = self._transport or urllib.request.urlopen
            self._calls += 1
            try:
                response = opener(request, timeout=self.config.timeout)
                try:
                    raw = response.read()
                finally:
                    close = getattr(response, "close", None)
                    if callable(close):
                        close()
                result, usage = _decode_chat_response_with_usage(raw)
            except DeepSeekAdapterError:
                raise
            except Exception:
                # Never propagate provider exception text, which may contain
                # request IDs, payload fragments, or secrets from a fake
                # transport/error page.
                raise DeepSeekAdapterError("transport_error") from None
            self._successes += 1
            for key, value in usage.items():
                self._usage[key] += value
            return result
        except Exception:
            self._errors += 1
            raise


class DeepSeekJsonClient(DeepSeekUtteranceGenerator):
    """Generic safe JSON client used by DeepSeek-only experiment adapters."""

    name = "deepseek-json-client"
    version = "m8_2_deepseek_json_client_v1"


__all__ = [
    "DEFAULT_DEEPSEEK_BASE_URL",
    "DEFAULT_DEEPSEEK_MODEL",
    "DEEPSEEK_PROMPT_VERSION",
    "DEFAULT_MAX_REQUEST_BYTES",
    "DeepSeekAdapterError",
    "DeepSeekCallBudgetExceeded",
    "DeepSeekConfig",
    "DeepSeekJsonClient",
    "DeepSeekUtteranceGenerator",
    "load_deepseek_config",
    "redact_known_people",
]
