"""Ollama-backed talk initiator for the M6.10 connection trial.

The initiator chooses an addressee, a grounded topic, a conversation act, and
a short opening.  The act is an internal/audit label only: the public
``Initiation`` contract remains target/topic/approach, and reactions remain
the responsibility of the simulation layer.
"""

from __future__ import annotations

import json
import random
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

from .contracts import Initiation, ParticipantContext, TalkInitiator

from .domain import Event, OnlineKnown, Participant, ResponseKind
from .llm_config import OllamaConfig, load_llm_config
from .rule_providers import RuleTalkInitiator


PROMPT_VERSION = "m6_10_initiator_v1"
MAX_APPROACH_LENGTH = 64
_ACTS = ("light_question", "self_disclosure", "invitation", "follow_up")
_ACT_SET = frozenset(_ACTS)
_OUTPUT_KEYS = {"target_id", "topic", "act", "approach"}
_UNSAFE_APPROACH_CHARACTERS = frozenset("`{}[]｛｝［］")
_SENTENCE_ENDINGS = frozenset("。！？!?")
# A small, deliberately conservative set of endings which normally introduce
# the rest of a Japanese clause.  Plain-form verbs, nouns, and questions are
# valid short openings, so this is not a general Japanese grammar checker.
_TRUNCATED_APPROACH_SUFFIXES = (
    "気が",
    "気",
    "けれども",
    "けれど",
    "けど",
    "のですが",
    "ですが",
    "ので",
    "のに",
    "ながら",
    "つつ",
    "ため",
    "せいで",
    "ことが",
    "ことを",
    "ことは",
    "のが",
    "のを",
    "のは",
)
_PAST_HISTORY_MARKERS = (
    "前回", "以前", "この前", "先ほど", "さっき", "前に", "話していた", "お話した", "話しました",
)
_INTEREST_CLAIM_MARKERS = (
    "興味を持っていました", "興味を持っていた", "興味がありました", "興味があった",
    "関心を持っていました", "関心を持っていた", "関心がありました", "関心があった",
    "興味を示していました", "関心を示していました", "好きでした", "好きだった",
    "好きですよね", "興味があるのね", "興味あるのね", "関心があるのね",
    "興味があるよね", "興味あるよね", "関心があるよね",
    "興味があるんですね", "興味あるんですね", "関心があるんですね",
)
_OBSERVATION_CLAIM_MARKERS = (
    "話しているのを聞きました",
    "話していたのを聞きました",
    "話しているのを聞いた",
    "話していたのを聞いた",
    "話しているのを耳にしました",
    "話していたのを耳にしました",
    "話しているのを耳にした",
    "話していたのを耳にした",
    "話しているのを聞いていました",
    "話していたのを聞いていました",
    "話の内容を聞きました",
    "話の内容を聞いた",
    "話を聞きました",
    "話を聞いた",
    "話を耳にしました",
    "話を耳にした",
)
_NAMED_OBSERVATION_VERBS = (
    "見ていました",
    "見ていた",
    "見ました",
    "見た",
    "話していました",
    "話していた",
    "読んでいました",
    "読んでいた",
)


class OutputValidationError(ValueError):
    """Stable, non-content validation reason for a model output.

    The code is intentionally the only diagnostic retained by the audit
    layer.  It lets us analyze fallback causes without storing the generated
    text or an exception message that could contain user/model content.
    """

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _pair_events(context: ParticipantContext, target_id: str) -> tuple[Event, ...]:
    return tuple(
        event
        for event in context.events
        if {event.actor, event.target} == {context.actor.id, target_id}
    )


def _topic_options(context: ParticipantContext, candidate: Participant) -> tuple[str, ...]:
    """Return only grounded topics, in the M5.1 evidence order."""

    online_topics = tuple(
        item.topic
        for item in context.online_memories
        if item.applies_to(candidate.id) and item.topic
    )
    shared = tuple(item for item in context.actor.interests if item in candidate.interests)
    prior_topics = tuple(dict.fromkeys(event.topic for event in _pair_events(context, candidate.id) if event.topic))
    options = (*online_topics, *shared, *prior_topics, *candidate.interests, context.rules.common_topic)
    return tuple(dict.fromkeys(options))


_TOPIC_SOURCE_ORDER = (
    "online_prior_experience",
    "shared_interest",
    "candidate_interest",
    "prior_pair_topic",
    "common_topic",
)


def _topic_sources(context: ParticipantContext, candidate: Participant) -> dict[str, list[str]]:
    """Explain each topic option without changing the option order."""

    pair_topics = {
        event.topic
        for event in _pair_events(context, candidate.id)
        if event.topic
    }
    shared = set(context.actor.interests).intersection(candidate.interests)
    candidate_interests = set(candidate.interests)
    options = _topic_options(context, candidate)
    sources: dict[str, list[str]] = {}
    for topic in options:
        available = {
            "online_prior_experience": any(
                memory.applies_to(candidate.id) and memory.topic == topic
                for memory in context.online_memories
            ),
            "shared_interest": topic in shared,
            "candidate_interest": topic in candidate_interests,
            "prior_pair_topic": topic in pair_topics,
            "common_topic": topic == context.rules.common_topic,
        }
        sources[topic] = [name for name in _TOPIC_SOURCE_ORDER if available[name]]
    return sources


def _participant_reference(person: Participant, role: str) -> dict[str, str]:
    return {"id": person.id, "name": person.name, "role": role}


def _eligible_acts(context: ParticipantContext, target_id: str) -> tuple[str, ...]:
    """Return acts the model may choose for one target.

    All acts are available for a candidate with no pair history except
    ``follow_up``.  Keeping this rule in the projection and validator makes
    the model's subjective choice bounded by facts held by the simulation.
    """

    acts = list(_ACTS)
    if not _pair_events(context, target_id):
        acts.remove("follow_up")
    return tuple(acts)


def _act_counts(act_history: Mapping[int, str] | None) -> dict[str, int]:
    counts = {act: 0 for act in _ACTS}
    if act_history:
        for act in act_history.values():
            if act in _ACT_SET:
                counts[act] += 1
    return counts


def project_initiator_context(
    context: ParticipantContext,
    act_history: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Build the narrow, JSON-safe context allowed to reach the model.

    Only the latest pair exchange is copied into the prompt.  Raw Event
    reaction values are not copied; the latest opening itself is enough to
    tell the model what it must not repeat.
    """

    actor = context.actor
    previous_act = None
    if act_history:
        prior_turns = [turn for turn in act_history if turn < context.turn]
        if prior_turns:
            previous_act = act_history[max(prior_turns)]
    candidates: list[dict[str, Any]] = []
    for candidate in context.candidates:
        pair = tuple(sorted((actor.id, candidate.id)))
        state = context.pair_states.get(pair) or context.pair_states.get((pair[1], pair[0]))
        all_history = _pair_events(context, candidate.id)
        # Keep the prompt small and focus the model on the latest exchange.
        # topic_options still uses the complete history so a previously
        # covered topic remains grounded as an option.
        latest = all_history[-1] if all_history else None
        topic_options = _topic_options(context, candidate)
        candidate_projection: dict[str, Any] = {
                "id": candidate.id,
                "name": candidate.name,
                "interests": list(candidate.interests),
                "stance": candidate.stance.value,
                "direct_known": bool(state is not None and state.online_known is OnlineKnown.DIRECT),
                "covered_topics": list(dict.fromkeys(event.topic for event in all_history if event.topic)),
                "latest_history": (
                    None
                    if latest is None
                    else {
                        "speaker": (
                            _participant_reference(actor, "self")
                            if latest.actor == actor.id
                            else _participant_reference(candidate, "candidate")
                        ),
                        "addressee": (
                            _participant_reference(candidate, "candidate")
                            if latest.target == candidate.id
                            else _participant_reference(actor, "self")
                        ),
                        "topic": latest.topic,
                        "approach": latest.approach,
                    }
                ),
                "topic_options": list(topic_options),
                "topic_sources": _topic_sources(context, candidate),
                "eligible_acts": list(_eligible_acts(context, candidate.id)),
            }
        memory = next(
            (item for item in context.online_memories if item.applies_to(candidate.id)),
            None,
        )
        if memory is not None:
            candidate_projection["online_prior_experience"] = context.private_online_projection[
                next(index for index, item in enumerate(context.online_memories) if item is memory)
            ]
        if latest is not None and act_history is not None:
            latest_act = act_history.get(latest.turn)
            if latest_act in _ACT_SET:
                candidate_projection["latest_history"]["act"] = latest_act
        candidates.append(candidate_projection)
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
        "previous_act": previous_act,
        "act_counts": _act_counts(act_history),
    }


def _json_schema(context: ParticipantContext) -> dict[str, Any]:
    candidate_ids = [person.id for person in context.candidates]
    topics = list(
        dict.fromkeys(
            topic
            for candidate in context.candidates
            for topic in _topic_options(context, candidate)
        )
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["target_id", "topic", "act", "approach"],
        "properties": {
            "target_id": {"type": "string", "enum": candidate_ids},
            "topic": {"type": "string", "enum": topics},
            "act": {"type": "string", "enum": list(_ACTS)},
            "approach": {"type": "string", "minLength": 1, "maxLength": MAX_APPROACH_LENGTH},
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


def _normalise_approach(approach: str) -> str:
    """Normalise only enough to catch exact repeated openings."""

    return " ".join(unicodedata.normalize("NFKC", approach).split()).casefold()


def _duplicate_approach(approach: str, context: ParticipantContext) -> bool:
    normalised = _normalise_approach(approach)
    return any(
        normalised == _normalise_approach(event.approach)
        for event in context.events
    )


_HONORIFICS = ("さん", "ちゃん", "くん", "様", "さま", "殿")


def _normalise_template(
    approach: str,
    context: ParticipantContext,
    topic: str,
) -> str:
    """Normalise exact name/topic slot substitutions, without fuzzy matching."""

    text = _normalise_approach(approach)
    replacements: list[tuple[str, str]] = []
    for person in (context.actor, *context.candidates):
        for honorific in _HONORIFICS:
            replacements.append((_normalise_approach(person.name + honorific), "{person}"))
        replacements.append((_normalise_approach(person.name), "{person}"))
    replacements.append((_normalise_approach(topic), "{topic}"))
    for source, placeholder in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        if source:
            text = text.replace(source, placeholder)
    return text


def _duplicate_template(approach: str, context: ParticipantContext, topic: str) -> bool:
    current = _normalise_template(approach, context, topic)
    return any(
        current == _normalise_template(event.approach, context, event.topic)
        for event in context.events
    )


def _unsupported_history_claim(approach: str, context: ParticipantContext, target_id: str) -> bool:
    if _pair_events(context, target_id):
        return False
    return any(marker in approach for marker in _PAST_HISTORY_MARKERS)


def _unsupported_observation_claim(
    approach: str,
    context: ParticipantContext,
    target: Participant,
) -> bool:
    """Reject explicit observations when the POC has no observation input."""

    for marker in _OBSERVATION_CLAIM_MARKERS:
        start = approach.find(marker)
        while start >= 0:
            suffix = approach[start + len(marker):]
            if not suffix.startswith(("か", "？", "?")):
                return True
            start = approach.find(marker, start + 1)
    for honorific in ("", *_HONORIFICS):
        name = re.escape(target.name + honorific)
        verbs = "|".join(re.escape(verb) for verb in _NAMED_OBSERVATION_VERBS)
        if re.search(rf"{name}\s*が[^。！？!?]{{0,20}}(?:{verbs})(?![か？?])", approach):
            return True
    return False


def _unsupported_interest_claim(
    approach: str,
    context: ParticipantContext,
    target: Participant,
    topic: str,
) -> bool:
    # Some model outputs omit the addressee while still using an explicit
    # interest-assertion ending.  Treat that conservative marker as a claim
    # about the selected candidate; ordinary questions and self-statements do
    # not use these markers and therefore remain allowed.
    if not any(marker in approach for marker in _INTEREST_CLAIM_MARKERS):
        return False
    if topic in target.interests:
        return False
    positive_topic_evidence = {
        event.topic
        for event in _pair_events(context, target.id)
        if (
            event.target == target.id
            and event.response is ResponseKind.POSITIVE
            and event.topic == topic
        )
    }
    return topic not in positive_topic_evidence


def _has_unsafe_approach_characters(approach: str) -> bool:
    if any(character in _UNSAFE_APPROACH_CHARACTERS for character in approach):
        return True
    return any(unicodedata.category(character).startswith("C") for character in approach)


def _contains_ascii_participant_id(approach: str, context: ParticipantContext) -> bool:
    for participant in (context.actor, *context.candidates):
        identifier = participant.id
        if not identifier or not identifier.isascii():
            continue
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])"
        if re.search(pattern, approach, flags=re.IGNORECASE):
            return True
    return False


def _looks_truncated_approach(approach: str) -> bool:
    """Reject an obvious generation cutoff without judging Japanese fluency.

    The local model has occasionally emitted the beginning of a clause (for
    example, ``一緒にやってみたい気``) and then stopped.  A sentence ending
    explicitly marked with 。！？ is trusted, while a response that reaches
    the schema's hard length boundary without such a marker is treated as a
    likely cutoff.  This intentionally leaves ordinary plain-form and
    noun-ending openings untouched.
    """

    text = approach.rstrip()
    if text.endswith(_TRUNCATED_APPROACH_SUFFIXES):
        return True
    if len(approach) >= MAX_APPROACH_LENGTH and text[-1] not in _SENTENCE_ENDINGS:
        return True
    return False


def socket_timeout_types() -> tuple[type[BaseException], ...]:
    import socket

    return (socket.timeout,)


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


def _fallback_reason(attempts: list[dict[str, Any]]) -> str:
    """Classify exhausted attempts without changing their detailed audit."""

    categories = {
        str(attempt.get("error_category"))
        for attempt in attempts
        if isinstance(attempt.get("error_category"), str) and attempt.get("error_category")
    }
    if not categories:
        return "unknown"
    if len(categories) > 1:
        return "mixed"
    category = next(iter(categories))
    if category == "semantic_invalid":
        return "validation"
    if category in {"malformed_json", "malformed_response"}:
        return "parse"
    if category in {"timeout", "http_error", "connection_error", "provider_error"}:
        return "transport"
    return "unknown"


class OllamaTalkInitiator:
    name = "ollama-talk-initiator"
    version = "m6_10_ollama_initiator_v1"

    def __init__(
        self,
        config: OllamaConfig | None = None,
        *,
        fallback: Any | None = None,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config or load_llm_config()
        self.fallback = fallback or RuleTalkInitiator()
        if not callable(getattr(self.fallback, "initiate", None)):
            raise ValueError("fallback must implement initiate()")
        self._opener = opener
        self._turns: list[dict[str, Any]] = []
        # This is deliberately private metadata.  It never crosses the
        # Initiation/Event/ParticipantTurn contracts.
        self._act_history: dict[int, str] = {}
        self._model_metadata: dict[str, Any] | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "model": self.config.model,
            "prompt_version": PROMPT_VERSION,
            "generation": {
                "temperature": self.config.temperature,
                "num_predict": self.config.num_predict,
                "think": self.config.think,
            },
            "retry_policy": {"max_attempts": self.config.max_attempts},
        }

    def initiate(self, context: ParticipantContext) -> Initiation:
        projection = project_initiator_context(context, self._act_history)
        attempts: list[dict[str, Any]] = []
        structured: dict[str, Any] | None = None
        outcome = "fallback"
        for attempt_no in range(1, self.config.max_attempts + 1):
            started = time.monotonic()
            try:
                payload = self._request(context, projection, attempt_no)
                structured = self._validate_output(payload, context)
                attempts.append({"attempt": attempt_no, "status": "success", "elapsed_ms": _elapsed_ms(started)})
                outcome = "llm"
                break
            except Exception as exc:
                attempt_audit: dict[str, Any] = {
                    "attempt": attempt_no,
                    "status": "error",
                    "elapsed_ms": _elapsed_ms(started),
                    "error_category": _error_category(exc),
                }
                if attempt_audit["error_category"] == "semantic_invalid" and isinstance(exc, OutputValidationError):
                    attempt_audit["validation_code"] = exc.code
                attempts.append(attempt_audit)

        if structured is not None:
            act = structured["act"]
            result = Initiation(structured["target_id"], structured["topic"], structured["approach"])
        else:
            # The fallback chooses only target/topic.  Its wording is
            # replaced with a deterministic act-specific opening so a failed
            # model turn cannot leak a stale or arbitrary partial output.
            initiate = getattr(self.fallback, "initiate", None)
            if not callable(initiate):
                # Constructor validation normally makes this unreachable;
                # retain a clear error if a caller mutates the dependency.
                raise ValueError("fallback must implement initiate()")
            fallback_result = initiate(context)
            if not isinstance(fallback_result, Initiation):
                raise ValueError("fallback initiate() must return Initiation")
            target_id, topic = fallback_result.target_id, fallback_result.topic
            has_history = bool(_pair_events(context, target_id))
            act = "follow_up" if has_history else "light_question"
            if has_history:
                approach = f"{topic}の続きを、もう少し聞かせてもらえますか？"
            else:
                approach = f"{topic}について、少し聞かせてもらえますか？"
            result = Initiation(target_id, topic, approach)

        # A turn has exactly one adopted act.  Assignment also makes repeated
        # direct calls for the same context turn idempotent in the audit
        # history rather than counting an unsuccessful attempt or duplicate.
        self._act_history[context.turn] = act

        audit: dict[str, Any] = {
            "turn": context.turn,
            "prompt_input": projection,
            "attempts": attempts,
            "outcome": outcome,
            "conversation_act": act,
        }
        if structured is not None:
            audit["initiator_output"] = dict(structured)
        else:
            audit["fallback_reason"] = _fallback_reason(attempts)
        self._turns.append(audit)
        return result

    # ParticipantProvider-style callers can use the same adapter while the
    # M5.1 contract is being integrated.
    propose = initiate

    def _request(self, context: ParticipantContext, projection: Mapping[str, Any], attempt: int) -> dict[str, Any]:
        user_content = json.dumps(projection, ensure_ascii=False, separators=(",", ":"))
        system = (
            "共有空間の参加者として、次の話しかけを1つ決めます。JSONはtarget_id/topic/act/approachの4キーだけにし、相手の返答は作りません。"
            "target_idは候補から、topicは選んだ相手のtopic_optionsから1つ選びます。"
            "まずactを1つ選び、その後でactの発話目的と形式に一致するapproachを作ります。act名だけを分散させず、実際の発話行為を一致させます。"
            "light_questionは相手から情報・希望・考えを得る短い質問で、共同提案や自分の感想だけにはしません。"
            "self_disclosureは入力にある自分の関心、またはこの場での現在の希望・感想を述べる発話で、相手への質問・依頼を含めません。"
            "invitationはactorとtargetがこの場で一緒に行う小さな行動提案で、共同行動が分かり、相手の希望・知識を尋ねるだけにはしません。"
            "follow_upはlatest_historyの具体的内容を受けて掘り下げる質問または受け止めで、履歴と無関係な新規質問にはしません。"
            "approachは文の前半に選択topicを1回だけ含め、45字程度までの完結した自然な1文にします。"
            "latest_historyはspeakerからaddresseeへの過去発話です。過去発話中の属性をaddresseeの事実にしません。"
            "latest_historyがない相手にもeligible_actsの範囲で選べますが、経験・好み・感情を相手の事実として断定しません。"
            "self_disclosureでも入力にない自分の過去の経験・行動・観察を作りません。"
            "latest_historyがある相手には、直前の問いと言い回しを繰り返さず、一段進めます。過去approachをそのまま再利用しません。"
            "相手名・話題だけを入れ替えた同じ質問の型を使わず、入力にない会話を聞いた・見たと作りません。"
            "相手の興味を断定するのはtopic_sourcesにcandidate_interestがある場合だけです。prior_pair_topicは興味の根拠ではありません。"
            "履歴にない事実・過去会話・関係は作りません。選んだ1人に1話題だけ話しかけ、説明・メタ発言・ASCIIのidは入れません。"
            "direct_knownはオンライン上の面識だけで、会話内容や好みの根拠にはなりません。online_prior_experienceがある場合だけ、そこに明記されたtopicと回数を使えます。固定の完成文や例文は使わず、相手とtopicに合わせて作ります。"
            "actorのstanceは言い方のニュアンスに反映します（積極的なら少し積極的、慎重なら控えめ、ふつうなら自然体）。"
            "previous_actとact_countsは偏りを避ける参考であり、特定actを強制しません。"
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
            "think": self.config.think,
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
            raise ValueError("structured initiation must be an object")
        return parsed

    def _validate_output(self, value: Mapping[str, Any], context: ParticipantContext) -> dict[str, Any]:
        if set(value) != _OUTPUT_KEYS:
            raise OutputValidationError("missing_or_extra_keys")
        target_id = value["target_id"]
        if not isinstance(target_id, str) or target_id not in {person.id for person in context.candidates}:
            raise OutputValidationError("invalid_target")
        target = next(person for person in context.candidates if person.id == target_id)
        topic = value["topic"]
        if not isinstance(topic, str) or topic not in _topic_options(context, target):
            raise OutputValidationError("invalid_topic")
        act = value["act"]
        if not isinstance(act, str) or act not in _ACT_SET:
            raise OutputValidationError("invalid_act")
        if act not in _eligible_acts(context, target_id):
            raise OutputValidationError("ineligible_act")
        approach = value["approach"]
        if not isinstance(approach, str) or not approach.strip() or len(approach) > MAX_APPROACH_LENGTH:
            raise OutputValidationError("empty_or_too_long")
        if topic not in approach:
            raise OutputValidationError("topic_not_mentioned")
        if _has_unsafe_approach_characters(approach):
            raise OutputValidationError("unsafe_chars")
        if _duplicate_approach(approach, context):
            raise OutputValidationError("duplicate_approach")
        if _duplicate_template(approach, context, topic):
            raise OutputValidationError("duplicate_template")
        if _contains_ascii_participant_id(approach, context):
            raise OutputValidationError("ascii_id")
        if sum(character in _SENTENCE_ENDINGS for character in approach) > 2:
            raise OutputValidationError("sentence_count")
        if _looks_truncated_approach(approach):
            raise OutputValidationError("truncated")
        if _starts_with_self_address(approach, context.actor):
            raise OutputValidationError("self_address")
        if _unsupported_history_claim(approach, context, target_id):
            raise OutputValidationError("unsupported_history")
        if _unsupported_observation_claim(approach, context, target):
            raise OutputValidationError("unsupported_observation")
        if _unsupported_interest_claim(approach, context, target, topic):
            raise OutputValidationError("unsupported_interest")
        compact = re.sub(r"[\s\W_]+", "", approach, flags=re.UNICODE).casefold()
        names = {context.actor.id.casefold(), context.actor.name.casefold(), target.id.casefold(), target.name.casefold()}
        if compact in {re.sub(r"[\s\W_]+", "", item, flags=re.UNICODE).casefold() for item in names if item}:
            raise OutputValidationError("only_name")
        return {"target_id": target_id, "topic": topic, "act": act, "approach": approach}

    def audit_snapshot(self) -> dict[str, Any]:
        success = sum(item["outcome"] == "llm" for item in self._turns)
        fallback_reasons = Counter(
            str(item.get("fallback_reason", "unknown"))
            for item in self._turns
            if item.get("outcome") == "fallback"
        )
        fallback_validation_codes = Counter(
            str(attempt["validation_code"])
            for item in self._turns
            if item.get("outcome") == "fallback"
            for attempt in item.get("attempts", ())
            if isinstance(attempt, dict) and isinstance(attempt.get("validation_code"), str)
        )
        return {
            "adapter": {"name": self.name, "version": self.version},
            "model": self.config.model,
            "model_metadata": self._model_metadata,
            "prompt_version": PROMPT_VERSION,
            "generation": {
                "temperature": self.config.temperature,
                "num_predict": self.config.num_predict,
                "think": self.config.think,
            },
            "retry_policy": {"max_attempts": self.config.max_attempts},
            "turns": list(self._turns),
            "summary": {
                "turns": len(self._turns),
                "llm": success,
                "fallback": len(self._turns) - success,
                "attempts": sum(len(item["attempts"]) for item in self._turns),
                "act_counts": _act_counts(self._act_history),
                "fallback_reasons": dict(sorted(fallback_reasons.items())),
                "fallback_validation_codes": dict(sorted(fallback_validation_codes.items())),
            },
        }

    def health_metadata(self) -> dict[str, Any]:
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

    model_metadata = health_metadata
    check_health = health_metadata


def _elapsed_ms(started: float) -> int:
    return max(0, int(round((time.monotonic() - started) * 1000)))


OllamaInitiator = OllamaTalkInitiator


__all__ = [
    "Initiation",
    "MAX_APPROACH_LENGTH",
    "OllamaInitiator",
    "OllamaTalkInitiator",
    "OutputValidationError",
    "PROMPT_VERSION",
    "RuleTalkInitiator",
    "TalkInitiator",
    "project_initiator_context",
]
