"""Ollama-backed ParticipantProvider used by the M4 connection trial.

Only the participant's next action crosses the adapter boundary.  The
relationship evaluator and all counters remain ordinary Python code.  The
adapter is deliberately defensive: every model response is schema- and
semantics-checked, and a failed turn uses the deterministic rule provider.
"""

from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .contracts import ParticipantContext
from .domain import Event, OnlineKnown, Participant, ParticipantTurn, ResponseKind
from .llm_config import OllamaConfig, load_llm_config
from .rule_providers import RuleParticipantProvider


PROMPT_VERSION = "m5_participant_projection_v6"
_RESPONSE_VALUES = tuple(item.value for item in ResponseKind)
_OUTPUT_KEYS = {"target_id", "topic", "approach", "response"}
_PAST_HISTORY_MARKERS = (
    "前回",
    "以前",
    "この前",
    "先ほど",
    "さっき",
    "前に",
    "話していた",
    "お話した",
    "話しました",
)
_INTEREST_CLAIM_MARKERS = (
    "興味を持っていました",
    "興味を持っていた",
    "興味がありました",
    "興味があった",
    "関心を持っていました",
    "関心を持っていた",
    "関心がありました",
    "関心があった",
    "興味を示していました",
    "関心を示していました",
    "好きでした",
    "好きだった",
)


def _pair_events(context: ParticipantContext, target_id: str) -> tuple[Event, ...]:
    return tuple(
        event
        for event in context.events
        if {event.actor, event.target} == {context.actor.id, target_id}
    )


def _topic_options(context: ParticipantContext, candidate: Participant) -> tuple[str, ...]:
    online_topics = tuple(
        memory.topic
        for memory in context.online_memories
        if memory.applies_to(candidate.id) and memory.topic
    )
    shared = tuple(item for item in context.actor.interests if item in candidate.interests)
    history = _pair_events(context, candidate.id)
    prior_topics = tuple(dict.fromkeys(event.topic for event in history if event.topic))
    # Keep the order meaningful for the prompt while exposing only topics
    # grounded in this actor-target pair.  Actor-only interests are excluded:
    # the target's interests (or an actual shared interest) are the evidence
    # available to a participant for choosing a topic.
    options = (
        *online_topics,
        *shared,
        *prior_topics,
        *candidate.interests,
        context.rules.common_topic,
    )
    return tuple(dict.fromkeys(options))


def project_participant_context(context: ParticipantContext) -> dict[str, Any]:
    """Return the intentionally narrow, JSON-safe prompt projection.

    Notice that this is built candidate-by-candidate.  There is no condition
    identifier, metrics, evaluator object, or global list of known pairs in
    the resulting payload.
    """

    actor = context.actor
    candidates: list[dict[str, Any]] = []
    for candidate in context.candidates:
        pair = tuple(sorted((actor.id, candidate.id)))
        state = context.pair_states.get(pair) or context.pair_states.get((pair[1], pair[0]))
        history = _pair_events(context, candidate.id)
        candidate_payload = {
                "id": candidate.id,
                "name": candidate.name,
                "interests": list(candidate.interests),
                "stance": candidate.stance.value,
                "direct_known": bool(state is not None and state.online_known is OnlineKnown.DIRECT),
                "onsite_history": [
                    {
                        "turn": event.turn,
                        "actor": event.actor,
                        "target": event.target,
                        "topic": event.topic,
                        "response": event.response.value,
                    }
                    for event in history
                ],
                "topic_options": list(_topic_options(context, candidate)),
            }
        memory = next(
            (item for item in context.online_memories if item.applies_to(candidate.id)),
            None,
        )
        if memory is not None:
            candidate_payload["online_prior_experience"] = context.private_online_projection[
                next(index for index, item in enumerate(context.online_memories) if item is memory)
            ]
        candidates.append(candidate_payload)
    return {
        "turn": context.turn,
        "actor": {
            "id": actor.id,
            "name": actor.name,
            "interests": list(actor.interests),
            "stance": actor.stance.value,
        },
        "candidates": candidates,
        "last_target": context.last_target,
        "common_topic": context.rules.common_topic,
    }


def _json_schema(context: ParticipantContext) -> dict[str, Any]:
    candidate_ids = [person.id for person in context.candidates]
    topic_options = list(
        dict.fromkeys(
            topic
            for candidate in context.candidates
            for topic in _topic_options(context, candidate)
        )
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["target_id", "topic", "approach", "response"],
        "properties": {
            "target_id": {"type": "string", "enum": candidate_ids},
            "topic": {"type": "string", "enum": topic_options},
            "approach": {"type": "string", "maxLength": 96},
            "response": {"type": "string", "enum": list(_RESPONSE_VALUES)},
        },
    }


def _copy_seed(rng: random.Random, attempt: int = 1) -> int:
    copied = random.Random()
    copied.setstate(rng.getstate())
    seed = 0
    for _ in range(max(1, attempt)):
        seed = copied.randrange(0, 2**31)
    return seed


def _starts_with_self_address(approach: str, actor: Participant) -> bool:
    """Detect only an obvious self-address at the beginning of a sentence.

    A participant may legitimately introduce themselves (``あかねです``), or
    mention their own name later in a sentence.  This deliberately narrow
    check catches the conversational address pattern without rejecting those
    cases.
    """

    text = approach.lstrip()
    for identifier in (actor.name, actor.id):
        if not identifier or not text.startswith(identifier):
            continue
        remainder = text[len(identifier):]
        if remainder.startswith(("さん", "ちゃん", "くん", "様", "さま", "殿")):
            return True
        if remainder and remainder[0] in "、,，:：.!！?？；;・~〜～-—":
            return True
    return False


def _unsupported_history_claim(approach: str, context: ParticipantContext, target_id: str) -> bool:
    """Reject assertions about an onsite past that this pair has not logged."""

    if _pair_events(context, target_id):
        return False
    return any(marker in approach for marker in _PAST_HISTORY_MARKERS)


def _unsupported_interest_claim(approach: str, context: ParticipantContext, target: Participant) -> bool:
    """Reject claims that the target previously showed an interest.

    A target-interest question such as ``展示に興味ありますか？`` remains
    valid.  A profile interest is an explicit piece of evidence, so a claim
    about a target profile interest is allowed; otherwise the claim must be
    grounded in a topic from a positive onsite event for this pair.  A positive
    event on an unrelated topic does not authorize every interest claim.
    """

    target_refs = (target.name, target.id, "あなた", "そちら")
    if not any(reference and reference in approach for reference in target_refs):
        return False
    if not any(marker in approach for marker in _INTEREST_CLAIM_MARKERS):
        return False
    positive_topics = {
        event.topic
        for event in _pair_events(context, target.id)
        if event.response is ResponseKind.POSITIVE and event.topic
    }
    evidence_terms = (*target.interests, *positive_topics)
    return not any(term in approach for term in evidence_terms)


def _error_category(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError) or isinstance(exc, socket_timeout_types()):
        return "timeout"
    if isinstance(exc, urllib.error.HTTPError):
        return "http_error"
    if isinstance(exc, urllib.error.URLError):
        return "connection_error"
    if isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError)):
        return "malformed_json"
    if isinstance(exc, KeyError):
        return "malformed_response"
    if isinstance(exc, ValueError):
        return "semantic_invalid"
    return "provider_error"


def socket_timeout_types() -> tuple[type[BaseException], ...]:
    # Avoid importing the socket module solely for isinstance on platforms
    # where the timeout exception is already a TimeoutError subclass.
    import socket

    return (socket.timeout,)


class OllamaParticipantProvider:
    name = "ollama-participant"
    version = "m4_ollama_participant_v1"

    def __init__(
        self,
        config: OllamaConfig | None = None,
        *,
        fallback: Any | None = None,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config or load_llm_config()
        self.fallback = fallback or RuleParticipantProvider()
        self._opener = opener
        self._turns: list[dict[str, Any]] = []
        self._model_metadata: dict[str, Any] | None = None

    def describe(self) -> dict[str, Any]:
        """Return serializable adapter metadata without contacting Ollama."""

        return {
            "name": self.name,
            "version": self.version,
            "model": self.config.model,
            "prompt_version": PROMPT_VERSION,
            "generation": {
                "temperature": self.config.temperature,
                "num_predict": self.config.num_predict,
            },
            "retry_policy": {"max_attempts": self.config.max_attempts},
        }

    def propose(self, context: ParticipantContext) -> ParticipantTurn:
        projection = project_participant_context(context)
        attempts: list[dict[str, Any]] = []
        structured: dict[str, Any] | None = None
        outcome = "fallback"
        for attempt_no in range(1, self.config.max_attempts + 1):
            started = time.monotonic()
            try:
                payload = self._request(context, projection, attempt_no)
                structured = self._validate_output(payload, context)
                attempts.append(
                    {"attempt": attempt_no, "status": "success", "elapsed_ms": _elapsed_ms(started)}
                )
                outcome = "llm"
                break
            except Exception as exc:  # provider errors are isolated to this turn
                attempts.append(
                    {
                        "attempt": attempt_no,
                        "status": "error",
                        "elapsed_ms": _elapsed_ms(started),
                        "error_category": _error_category(exc),
                    }
                )

        if structured is not None:
            result = ParticipantTurn(
                context.turn,
                context.actor.id,
                structured["target_id"],
                structured["topic"],
                structured["approach"],
                structured["response"],
            )
        else:
            # Rule fallback is intentionally called with the original context;
            # the copied seed above means a successful LLM request never moves
            # the run-owned RNG stream.
            result = self.fallback.propose(context)

        audit: dict[str, Any] = {
            "turn": context.turn,
            "prompt_input": projection,
            "attempts": attempts,
            "outcome": outcome,
        }
        if structured is not None:
            audit["structured_output"] = {
                **structured,
                "response": structured["response"].value
                if isinstance(structured["response"], ResponseKind)
                else structured["response"],
            }
        self._turns.append(audit)
        return result

    def _request(self, context: ParticipantContext, projection: Mapping[str, Any], attempt: int) -> dict[str, Any]:
        user_content = json.dumps(projection, ensure_ascii=False, separators=(",", ":"))
        system = (
            "あなたは共有空間にいる参加者の次の行動を一つだけ決めます。"
            "target_idは候補のidから選び、topicはその候補のtopic_optionsから選びます。"
            "actorと候補双方のinterestsとstance、direct_known、onsite_historyを考慮してください。"
            "direct_knownは事前のオンライン会話による面識で、話し方を少し自然にくだけさせる参考情報です。"
            "direct_knownからオンライン会話の内容は分かりません。online_prior_experienceがある場合だけ、その入力に明記されたtopicと回数を使えます。"
            "具体的な過去の話題を作らず、onsite_historyにある話題だけを過去の現地話題として扱ってください。"
            "topic_optionsは共有関心、当該ペアの現地履歴、許可されたonline_prior_experience、targetの関心、共通話題で構成されています。"
            "actorだけの関心をtargetも共有しているとは考えず、共有していない関心を共通の話題として主張しないでください。"
            "onsite_historyが空なら、前回・以前・この前・先ほど・さっきなどの過去の現地会話を断定せず、"
            "targetが以前に興味を示したとも断定しないでください。targetの関心を尋ねる質問は構いません。"
            "direct_knownだけを理由に相手を機械的に選ばず、関心や履歴も合わせて判断してください。"
            "approachではactorが選んだtargetに話しかけます。actor自身の名前や敬称で呼びかけず、名前を呼ぶならtargetの名前を使ってください。"
            "ただしactorが自分を紹介する「あかねです」のような文は構いません。"
            "approachは相手に話しかける短い自然な文、responseは選んだ候補がこの直後に示しそうな反応で、"
            "前向き・ふつう・噛み合わないのいずれかです。"
            "target_id/topic/approach/response以外のキーを含めず、JSONオブジェクトだけを返してください。"
        )
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "format": _json_schema(context),
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.num_predict,
                "seed": _copy_seed(context.rng, attempt),
            },
        }
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/api/chat",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        opener = self._opener or urllib.request.urlopen
        response = opener(request, timeout=self.config.timeout)
        try:
            raw = response.read()
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        envelope = json.loads(raw)
        content = envelope["message"]["content"]
        if isinstance(content, Mapping):
            return dict(content)
        if not isinstance(content, str):
            raise ValueError("message content is not a JSON string")
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("structured response must be an object")
        return parsed

    def _validate_output(self, value: Mapping[str, Any], context: ParticipantContext) -> dict[str, Any]:
        if set(value) != _OUTPUT_KEYS:
            raise ValueError("structured output must contain exactly target_id/topic/approach/response")
        target_id = value["target_id"]
        if not isinstance(target_id, str) or target_id not in {person.id for person in context.candidates}:
            raise ValueError("target_id is not a candidate")
        target = next(person for person in context.candidates if person.id == target_id)
        topic = value["topic"]
        options = _topic_options(context, target)
        if not isinstance(topic, str) or topic not in options:
            raise ValueError("topic is not an allowed candidate topic")
        approach = value["approach"]
        if not isinstance(approach, str) or not approach.strip() or len(approach) > 96:
            raise ValueError("approach is empty or too long")
        if _starts_with_self_address(approach, context.actor):
            raise ValueError("approach addresses the actor instead of the target")
        if _unsupported_history_claim(approach, context, target_id):
            raise ValueError("approach asserts an onsite history that is not recorded")
        if _unsupported_interest_claim(approach, context, target):
            raise ValueError("approach asserts an unsupported target interest")
        compact = re.sub(r"[\s\W_]+", "", approach, flags=re.UNICODE).casefold()
        forbidden_names = {context.actor.id.casefold(), context.actor.name.casefold(), target.id.casefold(), target.name.casefold()}
        if compact in {re.sub(r"[\s\W_]+", "", item, flags=re.UNICODE).casefold() for item in forbidden_names if item}:
            raise ValueError("approach is only an identifier/name")
        response_raw = value["response"]
        if isinstance(response_raw, ResponseKind):
            response = response_raw
        elif isinstance(response_raw, str):
            try:
                response = ResponseKind(response_raw)
            except ValueError as exc:
                raise ValueError("response is not a valid response enum") from exc
        else:
            raise ValueError("response is not a valid response enum")
        return {
            "target_id": target_id,
            "topic": topic,
            "approach": approach,
            "response": response,
        }

    def audit_snapshot(self) -> dict[str, Any]:
        success = sum(item["outcome"] == "llm" for item in self._turns)
        fallback = len(self._turns) - success
        attempts = sum(len(item["attempts"]) for item in self._turns)
        return {
            "adapter": {"name": self.name, "version": self.version},
            "model": self.config.model,
            "model_metadata": self._model_metadata,
            "prompt_version": PROMPT_VERSION,
            "generation": {
                "temperature": self.config.temperature,
                "num_predict": self.config.num_predict,
            },
            "retry_policy": {"max_attempts": self.config.max_attempts},
            "turns": list(self._turns),
            "summary": {
                "turns": len(self._turns),
                "llm": success,
                "fallback": fallback,
                "attempts": attempts,
            },
        }

    def health_metadata(self) -> dict[str, Any]:
        """Best-effort model metadata probe; errors are safe categories only."""

        request = urllib.request.Request(self.config.base_url.rstrip("/") + "/api/tags", method="GET")
        try:
            opener = self._opener or urllib.request.urlopen
            response = opener(request, timeout=self.config.timeout)
            try:
                raw = response.read()
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
            payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
            models = payload.get("models", []) if isinstance(payload, dict) else []
            entries = [item for item in models if isinstance(item, dict) and isinstance(item.get("name"), str)]
            names = [item["name"] for item in entries]
            selected = next((item for item in entries if item["name"] == self.config.model), None)
            if selected is not None:
                # Keep only the standard Ollama tags fields.  This is useful
                # provenance without copying arbitrary server response data.
                self._model_metadata = {
                    key: selected[key]
                    for key in ("name", "digest", "size", "modified_at", "details")
                    if key in selected
                }
            return {
                "ok": True,
                "model": self.config.model,
                "available": selected is not None,
                "models": names,
                "model_metadata": self._model_metadata,
            }
        except Exception as exc:
            return {"ok": False, "model": self.config.model, "error_category": _error_category(exc)}

    # Explicit aliases keep the optional helper easy to discover without
    # changing the core ParticipantProvider contract.
    model_metadata = health_metadata
    check_health = health_metadata


def _elapsed_ms(started: float) -> int:
    return max(0, int(round((time.monotonic() - started) * 1000)))


__all__ = ["OllamaParticipantProvider", "PROMPT_VERSION", "project_participant_context"]
