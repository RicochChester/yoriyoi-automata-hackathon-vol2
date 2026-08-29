"""Two-stage Ollama talk initiator for the M6.11 trial.

The planner chooses a grounded conversation plan.  A separate utterance
generator renders only that fixed plan into one opening sentence.  The public
TalkInitiator contract remains unchanged: conversation acts stay in audit
metadata and never enter ``Initiation``, ``Event``, or ``ParticipantTurn``.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import random
import time
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .contracts import Initiation, ParticipantContext, TalkInitiator
from .llm_config import OllamaConfig, load_llm_config
from .rule_providers import RuleTalkInitiator
from . import ollama_initiator as _single_stage


PROMPT_VERSION = "m6_11_two_stage_v1"
MAX_STAGE_ATTEMPTS = 2
_PLAN_KEYS = {"target_id", "topic", "act"}
_UTTERANCE_KEYS = {"approach"}

_ACT_DEFINITIONS = {
    "light_question": "相手から情報・希望・考えを得る短い質問",
    "self_disclosure": "入力にある自分の関心または現在の希望・感想を述べる（質問・依頼なし）",
    "invitation": "actorとtargetがこの場で一緒に行う小さな行動提案",
    "follow_up": "latest_historyの具体内容を受けた掘り下げ質問または受け止め",
}


def _elapsed_ms(started: float) -> int:
    return max(0, int(round((time.monotonic() - started) * 1000)))


def _stage_seed(rng: random.Random, stage: str, attempt: int) -> int:
    """Derive an Ollama seed without consuming the run-owned RNG.

    The stage and attempt are explicit inputs so a planner retry cannot share
    a seed with an utterance retry, while the original stream remains intact.
    """

    token = repr(rng.getstate()).encode("utf-8") + f"\0{stage}\0{attempt}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big")


def _without_approach(history: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(history, Mapping):
        return None
    return {
        key: copy.deepcopy(history[key])
        for key in ("speaker", "addressee", "topic", "act")
        if key in history
    }


def _planner_projection(
    context: ParticipantContext,
    act_history: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Expose planning facts without passing any prior wording."""

    full = _single_stage.project_initiator_context(context, act_history)
    candidates = []
    for candidate in full["candidates"]:
        candidates.append(
            {
                key: copy.deepcopy(candidate[key])
                for key in (
                    "id", "name", "interests", "stance", "direct_known",
                    "covered_topics", "topic_options", "topic_sources", "eligible_acts",
                    "online_prior_experience",
                )
                if key in candidate
            }
            | {"role": "candidate", "latest_history": _without_approach(candidate.get("latest_history"))}
        )
    actor = full["actor"]
    return {
        "turn": full["turn"],
        "actor": {**copy.deepcopy(actor), "role": "self"},
        "candidates": candidates,
        "last_target": full["last_target"],
        "common_topic": full["common_topic"],
        "previous_act": full["previous_act"],
        "act_counts": copy.deepcopy(full["act_counts"]),
        "act_definitions": copy.deepcopy(_ACT_DEFINITIONS),
    }


def _utterance_projection(
    context: ParticipantContext,
    plan: ConversationPlan,
    act_history: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Expose only the selected target's grounding and minimal recent wording."""

    full = _single_stage.project_initiator_context(context, act_history)
    target = next(item for item in full["candidates"] if item["id"] == plan.target_id)
    pair_events = _single_stage._pair_events(context, plan.target_id)
    latest_pair = pair_events[-1] if pair_events else None
    latest_pair_history = None
    if latest_pair is not None:
        latest_pair_history = {
            "turn": latest_pair.turn,
            "speaker": latest_pair.actor,
            "addressee": latest_pair.target,
            "topic": latest_pair.topic,
            "approach": latest_pair.approach,
        }
        prior_act = (act_history or {}).get(latest_pair.turn)
        if prior_act in _single_stage._ACT_SET:
            latest_pair_history["act"] = prior_act
    recent_approaches = [
        {"turn": event.turn, "topic": event.topic, "approach": event.approach}
        for event in context.events[-2:]
    ]
    return {
        "turn": full["turn"],
        "actor": {**copy.deepcopy(full["actor"]), "role": "self"},
        "selected_plan": plan.to_dict(),
        "selected_target": {
            key: copy.deepcopy(target[key])
            for key in ("id", "name", "interests", "stance", "topic_sources", "online_prior_experience")
            if key in target
        },
        "topic_grounding": {
            "topic": plan.topic,
            "covered_topics": copy.deepcopy(target.get("covered_topics", [])),
            "topic_options": copy.deepcopy(target.get("topic_options", [])),
        },
        "latest_pair_history": latest_pair_history,
        "recent_approaches": recent_approaches,
        "act_definitions": copy.deepcopy(_ACT_DEFINITIONS),
    }


@dataclass(frozen=True)
class ConversationPlan:
    """Model-independent plan exchanged between the two LLM stages."""

    target_id: str
    topic: str
    act: str

    def __post_init__(self) -> None:
        for name in ("target_id", "topic", "act"):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"conversation plan {name} must be a non-empty trimmed string")
        if self.act not in _single_stage._ACT_SET:
            raise ValueError("conversation plan act is not supported")

    def to_dict(self) -> dict[str, str]:
        return {"target_id": self.target_id, "topic": self.topic, "act": self.act}


@runtime_checkable
class Planner(Protocol):
    def plan(
        self,
        context: ParticipantContext,
        *,
        projection: Mapping[str, Any],
        attempt: int,
    ) -> ConversationPlan | Mapping[str, Any]:
        """Choose one grounded plan without generating wording."""


@runtime_checkable
class UtteranceGenerator(Protocol):
    def generate(
        self,
        context: ParticipantContext,
        plan: ConversationPlan,
        *,
        projection: Mapping[str, Any],
        attempt: int,
    ) -> str | Mapping[str, Any]:
        """Render only the fixed plan as one approach."""


class StageValidationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _planner_schema(context: ParticipantContext) -> dict[str, Any]:
    topics = list(
        dict.fromkeys(
            topic
            for candidate in context.candidates
            for topic in _single_stage._topic_options(context, candidate)
        )
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["target_id", "topic", "act"],
        "properties": {
            "target_id": {"type": "string", "enum": [item.id for item in context.candidates]},
            "topic": {"type": "string", "enum": topics},
            "act": {"type": "string", "enum": list(_single_stage._ACTS)},
        },
    }


def _utterance_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["approach"],
        "properties": {
            "approach": {"type": "string", "minLength": 1, "maxLength": _single_stage.MAX_APPROACH_LENGTH},
        },
    }


def _decode_response(raw: bytes | str) -> dict[str, Any]:
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
        import re

        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("structured stage output must be an object")
    return parsed


class _OllamaStage:
    def __init__(self, config: OllamaConfig, *, opener: Callable[..., Any] | None = None) -> None:
        self.config = config
        self._opener = opener

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
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
        return _decode_response(raw)


class OllamaConversationPlanner(_OllamaStage):
    name = "ollama-conversation-planner"
    version = "m6_11_planner_v1"

    def plan(
        self,
        context: ParticipantContext,
        *,
        projection: Mapping[str, Any] | None = None,
        attempt: int = 1,
    ) -> dict[str, Any]:
        projection = projection or _planner_projection(context)
        system = (
            "共有空間での次の会話計画を1つ決めます。JSONはtarget_id/topic/actの3キーだけにします。"
            "target_idは候補から、topicは選んだ候補のtopic_optionsから、actはその候補のeligible_actsから選びます。"
            "light_questionは相手から情報・希望・考えを得る短い質問、self_disclosureは入力にある自分の関心またはこの場での現在の希望・感想、"
            "invitationはactorとtargetがこの場で一緒に行う小さな行動提案、follow_upはlatest_historyの具体的内容を受けた掘り下げです。"
            "approachや文章は生成しません。入力にない事実・記憶・観察は作りません。"
        )
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(projection, ensure_ascii=False, separators=(",", ":"))},
            ],
            "stream": False,
            "format": _planner_schema(context),
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.num_predict,
                "seed": _stage_seed(context.rng, "planner", attempt),
            },
            "think": self.config.think,
        }
        return self._post(body)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "model": self.config.model}


class OllamaUtteranceGenerator(_OllamaStage):
    name = "ollama-utterance-generator"
    version = "m6_11_utterance_v1"

    def generate(
        self,
        context: ParticipantContext,
        plan: ConversationPlan,
        *,
        projection: Mapping[str, Any] | None = None,
        attempt: int = 1,
    ) -> dict[str, Any]:
        base = dict(projection or _utterance_projection(context, plan))
        base["selected_plan"] = plan.to_dict()
        system = (
            "固定されたselected_planだけを自然な一文にします。JSONはapproachの1キーだけにします。"
            "target_id/topic/actを変更・追加せず、計画のactの目的と形式に一致する短い発話を返します。"
            "light_questionは相手から情報・希望・考えを得る質問、self_disclosureは自分の現在の関心・希望・感想だけ、"
            "invitationはactorとtargetの共同行動提案、follow_upはlatest_historyの具体内容を受けた質問または受け止めです。"
            "入力にない事実・記憶・観察、説明やメタ発言は作りません。"
        )
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(base, ensure_ascii=False, separators=(",", ":"))},
            ],
            "stream": False,
            "format": _utterance_schema(),
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.num_predict,
                "seed": _stage_seed(context.rng, "utterance", attempt),
            },
            "think": self.config.think,
        }
        return self._post(body)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "model": self.config.model}


def _validate_plan(value: ConversationPlan | Mapping[str, Any], context: ParticipantContext) -> ConversationPlan:
    if isinstance(value, ConversationPlan):
        raw = value.to_dict()
    elif isinstance(value, Mapping):
        if set(value) != _PLAN_KEYS:
            raise StageValidationError("missing_or_extra_keys")
        raw = dict(value)
    else:
        raise StageValidationError("malformed_response")
    target_id = raw.get("target_id")
    if not isinstance(target_id, str) or target_id not in {item.id for item in context.candidates}:
        raise StageValidationError("invalid_target")
    target = next(item for item in context.candidates if item.id == target_id)
    topic = raw.get("topic")
    if not isinstance(topic, str) or topic not in _single_stage._topic_options(context, target):
        raise StageValidationError("invalid_topic")
    act = raw.get("act")
    if not isinstance(act, str) or act not in _single_stage._ACT_SET:
        raise StageValidationError("invalid_act")
    if act not in _single_stage._eligible_acts(context, target_id):
        raise StageValidationError("ineligible_act")
    try:
        return ConversationPlan(target_id, topic, act)
    except ValueError as exc:
        raise StageValidationError("invalid_plan") from exc


def _validate_utterance(value: str | Mapping[str, Any], context: ParticipantContext, plan: ConversationPlan) -> str:
    if isinstance(value, str):
        raw = {"approach": value}
    elif isinstance(value, Mapping):
        if set(value) != _UTTERANCE_KEYS:
            raise StageValidationError("missing_or_extra_keys")
        raw = dict(value)
    else:
        raise StageValidationError("malformed_response")
    approach = raw.get("approach")
    if not isinstance(approach, str):
        raise StageValidationError("empty_or_too_long")
    try:
        validated = _single_stage.OllamaTalkInitiator._validate_output(
            object.__new__(_single_stage.OllamaTalkInitiator),
            {**plan.to_dict(), "approach": approach},
            context,
        )
    except _single_stage.OutputValidationError as exc:
        raise StageValidationError(exc.code) from exc
    except Exception as exc:
        raise StageValidationError("semantic_invalid") from exc
    return validated["approach"]


def _fallback_initiation(context: ParticipantContext) -> Initiation:
    """Delegate the complete failed turn to the established rule baseline."""

    return RuleTalkInitiator().initiate(context)


class TwoStageTalkInitiator:
    name = "ollama-two-stage-talk-initiator"
    version = "m6_11_two_stage_initiator_v1"

    def __init__(
        self,
        config: OllamaConfig | None = None,
        *,
        planner: Planner | None = None,
        utterance_generator: UtteranceGenerator | None = None,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config or load_llm_config()
        self.planner = planner or OllamaConversationPlanner(self.config, opener=opener)
        self.utterance_generator = utterance_generator or OllamaUtteranceGenerator(self.config, opener=opener)
        self._act_history: dict[int, str] = {}
        self._turns: list[dict[str, Any]] = []

    def describe(self) -> dict[str, Any]:
        stages = self._stage_descriptions()
        result: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "prompt_version": PROMPT_VERSION,
            "generation": {
                "temperature": self.config.temperature,
                "num_predict": self.config.num_predict,
                "think": self.config.think,
            },
            "stages": stages,
            "retry_policy": {"max_attempts_per_stage": MAX_STAGE_ATTEMPTS},
        }
        model = _common_stage_model(stages)
        if model is not None:
            result["model"] = model
        return result

    def _stage_descriptions(self) -> dict[str, dict[str, Any]]:
        return {
            "planner": _stage_metadata(self.planner),
            "utterance": _stage_metadata(self.utterance_generator),
        }

    def initiate(self, context: ParticipantContext) -> Initiation:
        started = time.monotonic()
        projection = _planner_projection(context, self._act_history)
        planner_started = time.monotonic()
        planner_attempts: list[dict[str, Any]] = []
        utterance_attempts: list[dict[str, Any]] = []
        selected: ConversationPlan | None = None
        utterance: str | None = None
        fallback_stage: str | None = None

        for attempt in range(1, min(MAX_STAGE_ATTEMPTS, self.config.max_attempts) + 1):
            attempt_started = time.monotonic()
            try:
                selected = _validate_plan(
                    _invoke_planner(self.planner, context, projection, attempt),
                    context,
                )
                planner_attempts.append({"stage": "planner", "attempt": attempt, "status": "success", "elapsed_ms": _elapsed_ms(attempt_started)})
                break
            except Exception as exc:
                item = {
                    "stage": "planner", "attempt": attempt, "status": "error",
                    "elapsed_ms": _elapsed_ms(attempt_started),
                    "error_category": _single_stage._error_category(exc),
                }
                if isinstance(exc, StageValidationError):
                    item["validation_code"] = exc.code
                planner_attempts.append(item)
        planner_latency_ms = _elapsed_ms(planner_started)
        if selected is None:
            fallback_stage = "planner"
        else:
            utterance_projection = _utterance_projection(context, selected, self._act_history)
            utterance_started = time.monotonic()
            for attempt in range(1, min(MAX_STAGE_ATTEMPTS, self.config.max_attempts) + 1):
                attempt_started = time.monotonic()
                try:
                    utterance = _validate_utterance(
                        _invoke_utterance(
                            self.utterance_generator,
                            context,
                            selected,
                            utterance_projection,
                            attempt,
                        ),
                        context,
                        selected,
                    )
                    utterance_attempts.append({"stage": "utterance", "attempt": attempt, "status": "success", "elapsed_ms": _elapsed_ms(attempt_started)})
                    break
                except Exception as exc:
                    item = {
                        "stage": "utterance", "attempt": attempt, "status": "error",
                        "elapsed_ms": _elapsed_ms(attempt_started),
                        "error_category": _single_stage._error_category(exc),
                    }
                    if isinstance(exc, StageValidationError):
                        item["validation_code"] = exc.code
                    utterance_attempts.append(item)
            if utterance is None:
                fallback_stage = "utterance"

        utterance_latency_ms = (
            _elapsed_ms(utterance_started)
            if selected is not None
            else 0
        )

        if selected is not None and utterance is not None:
            result = Initiation(selected.target_id, selected.topic, utterance)
            adopted_plan = selected.to_dict()
            outcome = "llm"
        else:
            result = _fallback_initiation(context)
            adopted_plan = None
            outcome = "fallback"
        if outcome == "llm":
            self._act_history[context.turn] = selected.act

        all_attempts = planner_attempts + utterance_attempts
        selected_output = selected.to_dict() if selected is not None else None
        total_latency_ms = max(
            _elapsed_ms(started),
            planner_latency_ms + utterance_latency_ms,
        )
        planner_stage = {
            "prompt_input": copy.deepcopy(projection),
            "attempts": copy.deepcopy(planner_attempts),
            "selected_output": copy.deepcopy(selected_output),
            "latency_ms": planner_latency_ms,
        }
        utterance_stage = {
            "prompt_input": copy.deepcopy(utterance_projection) if selected else None,
            "attempts": copy.deepcopy(utterance_attempts),
            "selected_output": {"approach": result.approach} if outcome == "llm" else None,
            "latency_ms": utterance_latency_ms,
        }
        audit: dict[str, Any] = {
            "turn": context.turn,
            "prompt_input": copy.deepcopy(projection),
            "planner_prompt": copy.deepcopy(projection),
            "utterance_prompt": copy.deepcopy(utterance_projection) if selected else None,
            "planner_attempts": copy.deepcopy(planner_attempts),
            "utterance_attempts": copy.deepcopy(utterance_attempts),
            "planner": planner_stage,
            "utterance": utterance_stage,
            "attempts": copy.deepcopy(all_attempts),
            "selected_output": selected_output,
            "adopted_plan": adopted_plan,
            "final_approach": result.approach,
            "outcome": outcome,
            "fallback_stage": fallback_stage,
            "total_latency": total_latency_ms,
            "total_latency_ms": total_latency_ms,
        }
        if outcome == "llm":
            audit["conversation_act"] = selected.act
            audit["structured_output"] = {**selected_output, "approach": result.approach}
        self._turns.append(audit)
        return result

    propose = initiate

    def audit_snapshot(self) -> dict[str, Any]:
        llm = sum(item["outcome"] == "llm" for item in self._turns)
        stages = self._stage_descriptions()
        result: dict[str, Any] = {
            "adapter": {"name": self.name, "version": self.version},
            "prompt_version": PROMPT_VERSION,
            "generation": {
                "temperature": self.config.temperature,
                "num_predict": self.config.num_predict,
                "think": self.config.think,
            },
            "retry_policy": {"max_attempts_per_stage": MAX_STAGE_ATTEMPTS},
            "turns": copy.deepcopy(self._turns),
            "stages": copy.deepcopy(stages),
            "summary": {
                "turns": len(self._turns),
                "llm": llm,
                "fallback": len(self._turns) - llm,
                "attempts": sum(len(item["attempts"]) for item in self._turns),
                "act_counts": _single_stage._act_counts(self._act_history),
                "fallback_reasons": dict(Counter("stage_failure" for item in self._turns if item["outcome"] == "fallback")),
            },
        }
        model = _common_stage_model(stages)
        if model is not None:
            result["model"] = model
        return result


def _stage_metadata(stage: Any) -> dict[str, Any]:
    describe = getattr(stage, "describe", None)
    metadata: dict[str, Any] = {}
    if callable(describe):
        try:
            value = describe()
            if isinstance(value, Mapping):
                metadata = copy.deepcopy(dict(value))
        except Exception:
            metadata = {}
    metadata.setdefault("name", str(getattr(stage, "name", stage.__class__.__name__)))
    metadata.setdefault("version", str(getattr(stage, "version", "unknown")))
    return metadata


def _common_stage_model(stages: Mapping[str, Mapping[str, Any]]) -> str | None:
    models = [stage.get("model") for stage in stages.values()]
    if len(models) != 2 or any(not isinstance(model, str) or not model for model in models):
        return None
    if models[0] != models[1]:
        return None
    return models[0]


def _invoke_planner(stage: Any, context: ParticipantContext, projection: Mapping[str, Any], attempt: int) -> Any:
    method = stage.plan
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    names = {parameter.name for parameter in parameters}
    accepts_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    if accepts_kwargs or {"projection", "attempt"}.intersection(names):
        kwargs = {
            key: value
            for key, value in (("projection", projection), ("attempt", attempt))
            if accepts_kwargs or key in names
        }
        return method(context, **kwargs)
    return method(context)


def _invoke_utterance(
    stage: Any,
    context: ParticipantContext,
    plan: ConversationPlan,
    projection: Mapping[str, Any],
    attempt: int,
) -> Any:
    method = stage.generate
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    names = {parameter.name for parameter in parameters}
    accepts_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    if accepts_kwargs or {"projection", "attempt"}.intersection(names):
        kwargs = {
            key: value
            for key, value in (("projection", projection), ("attempt", attempt))
            if accepts_kwargs or key in names
        }
        return method(context, plan, **kwargs)
    return method(context, plan)


__all__ = [
    "ConversationPlan",
    "Planner",
    "UtteranceGenerator",
    "OllamaConversationPlanner",
    "OllamaUtteranceGenerator",
    "TwoStageTalkInitiator",
    "PROMPT_VERSION",
]
