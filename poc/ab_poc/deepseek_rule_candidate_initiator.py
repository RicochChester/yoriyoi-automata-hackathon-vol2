"""M8.2a rule-candidate + DeepSeek selection/utterance initiator.

Rules construct a bounded set of grounded plans without any local LLM.
DeepSeek receives only pseudonymous plan candidates and returns a plan ID,
conversation act, and one utterance.  Target IDs remain in an in-memory map.
"""

from __future__ import annotations

import copy
import json
import random
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from . import ollama_initiator as _validation
from .contracts import Initiation, ParticipantContext
from .deepseek_adapter import (
    DeepSeekAdapterError,
    DeepSeekConfig,
    DeepSeekJsonClient,
    redact_known_people,
)
from .domain import OnlineKnown, canonical_pair
from .rule_providers import RuleTalkInitiator


PROMPT_VERSION = "m11_n0_grounded_rule_candidates_deepseek_v1"
NATURAL_PROMPT_VERSION = "m14_natural_conversation_deepseek_v1"
MAX_PLAN_CANDIDATES = 3
RECENT_TOPIC_WINDOW = 3
MAX_OUTPUT_TOKENS = 64
_OUTPUT_KEYS = {"plan_id", "act", "approach"}
_RETRYABLE_PROVIDER_ERRORS = {"transport_error", "empty_content", "malformed_response"}
_ACT_ORDER = ("light_question", "self_disclosure", "invitation", "follow_up")


def _elapsed_ms(started: float) -> int:
    return max(0, int(round((time.monotonic() - started) * 1000)))


@dataclass(frozen=True)
class RulePlanCandidate:
    plan_id: str
    target_id: str
    topic: str
    eligible_acts: tuple[str, ...]
    projection: Mapping[str, Any]


class RulePlanCandidateBuilder:
    """Build at most one grounded representative plan for each target."""

    name = "rule-plan-candidate-builder"
    version = "m9_1_rule_plan_candidates_v1"

    def __init__(self, *, include_persona: bool = False) -> None:
        # Legacy candidate projections are kept byte-for-byte compatible;
        # the M14 initiator opts into the additional prompt card.
        self.include_persona = include_persona
        if include_persona:
            self.version = "m14_persona_rule_plan_candidates_v1"

    @staticmethod
    def _latest_pair_exchange(context: ParticipantContext, target_id: str) -> dict[str, Any] | None:
        pair = canonical_pair(context.actor.id, target_id)
        events = [
            event for event in context.events
            if canonical_pair(event.actor, event.target) == pair
        ]
        if not events:
            return None
        latest = events[-1]
        return {
            "turn": latest.turn,
            "speaker_role": "actor" if latest.actor == context.actor.id else "target",
            "topic": latest.topic,
            "approach": latest.approach,
            "response": latest.response.value,
        }

    @staticmethod
    def _online_familiarity(context: ParticipantContext, target_id: str) -> str:
        state = context.pair_states.get(canonical_pair(context.actor.id, target_id))
        return (
            "direct_online_known"
            if state is not None and state.online_known is OnlineKnown.DIRECT
            else "no_direct_online_tie"
        )

    @staticmethod
    def _select_topic(
        context: ParticipantContext,
        target_id: str,
        topics: tuple[str, ...] | list[str],
    ) -> str:
        pair = canonical_pair(context.actor.id, target_id)
        pair_topics = {
            event.topic
            for event in context.events
            if canonical_pair(event.actor, event.target) == pair
        }
        recent_topics = {event.topic for event in context.events[-RECENT_TOPIC_WINDOW:]}
        # Stable tiers preserve the grounded option order while preferring a
        # topic unused by both this pair and the recent conversation overall.
        for predicate in (
            lambda topic: topic not in pair_topics and topic not in recent_topics,
            lambda topic: topic not in pair_topics,
            lambda topic: topic not in recent_topics,
            lambda topic: True,
        ):
            selected = next((topic for topic in topics if predicate(topic)), None)
            if selected is not None:
                return selected
        raise ValueError("grounded topic options must not be empty")

    def build(self, context: ParticipantContext) -> tuple[RulePlanCandidate, ...]:
        result: list[RulePlanCandidate] = []
        targets = list(context.candidates)
        local_rng = random.Random()
        local_rng.setstate(context.rng.getstate())
        local_rng.shuffle(targets)
        for index, target in enumerate(targets[:MAX_PLAN_CANDIDATES], start=1):
            topics = _validation._topic_options(context, target)
            if not topics:
                continue
            plan_id = f"p{index}"
            topic = self._select_topic(context, target.id, topics)
            eligible_acts = _validation._eligible_acts(context, target.id)
            latest_exchange = self._latest_pair_exchange(context, target.id)
            # A follow-up is only coherent when the current actor spoke last:
            # the target can now be invited to expand on that opening.  If the
            # target spoke last, another follow-up from the actor would invert
            # the direction encoded by this act.
            if (
                "follow_up" in eligible_acts
                and (
                    latest_exchange is None
                    or latest_exchange["speaker_role"] != "actor"
                    or latest_exchange["topic"] != topic
                )
            ):
                eligible_acts = tuple(act for act in eligible_acts if act != "follow_up")
            projection = {
                "plan_id": plan_id,
                "topic": topic,
                "eligible_acts": list(eligible_acts),
                "actor_stance": context.actor.stance.value,
                "target_stance": target.stance.value,
                "actor_affect": context.outgoing_affect.get(target.id, 0),
                "online_familiarity": self._online_familiarity(context, target.id),
                "latest_pair_exchange": latest_exchange,
            }
            if self.include_persona:
                projection["actor_persona"] = context.actor.persona
                pair = canonical_pair(context.actor.id, target.id)
                pair_history = [
                    item for item in context.dialogue_history
                    if canonical_pair(item.actor, item.target) == pair
                ]
                recent_history = list(context.dialogue_history[-2:])
                for item in pair_history[-1:]:
                    if item not in recent_history:
                        recent_history.append(item)
                projection["recent_dialogue_history"] = [
                    {
                        "turn": item.turn,
                        "actor": item.actor,
                        "target": item.target,
                        "topic": item.topic,
                        "approach": item.approach,
                        "approach_speaker_role": "actor",
                        "reply": item.reply,
                        "reply_speaker_role": "target",
                        "reaction": item.reaction,
                    }
                    for item in recent_history
                ]
            memory = next(
                (item for item in context.online_memories if item.applies_to(target.id)),
                None,
            )
            if memory is not None:
                projection["online_prior_experience"] = context.private_online_projection[
                    next(index for index, item in enumerate(context.online_memories) if item is memory)
                ]
            projection = redact_known_people(projection, context, target.id)
            result.append(RulePlanCandidate(
                plan_id=plan_id,
                target_id=target.id,
                topic=topic,
                eligible_acts=eligible_acts,
                projection=projection,
            ))
        if not result:
            raise ValueError("rule plan candidate builder produced no candidates")
        return tuple(result)


class CandidateOutputError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ApproachOutputError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def render_rule_approach(topic: str, act: str, variant: int = 0) -> str:
    """Render a short deterministic utterance for one already-selected act."""

    templates = {
        "light_question": (
            "{topic}について、どう思いますか？",
            "{topic}で、気になることはありますか？",
            "{topic}について、少し聞かせてもらえますか？",
        ),
        "self_disclosure": (
            "私は{topic}について話してみたいです。",
            "この場では{topic}について話してみたいです。",
            "私は{topic}について話したいです。",
        ),
        "invitation": (
            "{topic}について、よかったら一緒に話しませんか？",
            "{topic}のことを、少し一緒に話してみませんか？",
            "よければ、{topic}について一緒に考えませんか？",
        ),
        "follow_up": (
            "{topic}について、もう少し聞かせてもらえますか？",
            "{topic}の続きも、よければ聞かせてください。",
            "{topic}について、もう少し詳しく教えてもらえますか？",
        ),
    }
    if not isinstance(variant, int) or isinstance(variant, bool) or variant < 0:
        raise ValueError("variant must be a non-negative integer")
    try:
        choices = templates[act]
        return choices[variant % len(choices)].format(topic=topic)
    except KeyError as exc:
        raise ValueError("unsupported conversation act") from exc


def is_meta_speech_description(approach: Any) -> bool:
    """Detect a small set of action-description endings, without regex."""

    if not isinstance(approach, str):
        return False
    normalized = approach.strip().rstrip("。.!！?？").strip()
    return normalized.endswith((
        "と尋ねます",
        "と質問します",
        "と話します",
        "と提案します",
        "と声をかけます",
    ))


def _usage_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, int]:
    return {
        key: max(0, int(after.get(key, 0)) - int(before.get(key, 0)))
        for key in ("prompt_tokens", "completion_tokens", "cache_hit_tokens", "cache_miss_tokens")
    }


def preferred_acts(
    candidates: tuple[RulePlanCandidate, ...],
    act_history: Mapping[int, str],
    turn: int,
) -> tuple[str, ...]:
    """Rank, but never force, up to two currently eligible acts."""

    eligible = {
        act
        for candidate in candidates
        for act in candidate.eligible_acts
    }
    counts = _validation._act_counts(act_history)
    prior_turns = [prior_turn for prior_turn in act_history if prior_turn < turn]
    previous_act = act_history[max(prior_turns)] if prior_turns else None
    ranked = sorted(
        (act for act in _ACT_ORDER if act in eligible),
        key=lambda act: (
            act == previous_act,
            counts.get(act, 0),
            _ACT_ORDER.index(act),
        ),
    )
    return tuple(ranked[:2])


class DeepSeekRuleCandidateInitiator:
    """Select one rule-built plan and render it with a single cloud model."""

    name = "deepseek-rule-candidate-initiator"
    version = "m11_n0_deepseek_rule_candidate_v1"

    def __init__(
        self,
        config: DeepSeekConfig | None = None,
        *,
        candidate_builder: RulePlanCandidateBuilder | None = None,
        client: DeepSeekJsonClient | None = None,
        transport: Any | None = None,
        allow_natural_wording: bool = False,
    ) -> None:
        if client is not None and (config is not None or transport is not None):
            raise ValueError("client cannot be combined with config or transport")
        self.client = client or DeepSeekJsonClient(config, transport=transport)
        self.config = self.client.config
        self.candidate_builder = candidate_builder or RulePlanCandidateBuilder()
        self._turns: list[dict[str, Any]] = []
        self._act_history: dict[int, str] = {}
        self._renderer_counts: Counter[tuple[str, ...]] = Counter()
        self.allow_natural_wording = bool(allow_natural_wording)
        self.prompt_version = NATURAL_PROMPT_VERSION if self.allow_natural_wording else PROMPT_VERSION

    def _render_safe_approach(self, topic: str, act: str) -> str:
        key = (topic, act)
        variant = self._renderer_counts[key]
        self._renderer_counts[key] += 1
        return render_rule_approach(topic, act, variant)

    def _render_light_question(
        self,
        candidate: RulePlanCandidate,
    ) -> tuple[str, dict[str, Any]]:
        latest = candidate.projection.get("latest_pair_exchange")
        if not isinstance(latest, Mapping):
            mode = "first_contact"
            previous_topic = None
            history_turn = None
        else:
            previous_topic = latest.get("topic") if isinstance(latest.get("topic"), str) else None
            history_turn = latest.get("turn") if isinstance(latest.get("turn"), int) else None
            mode = "continue" if previous_topic == candidate.topic else "bridge"
        templates = {
            "first_contact": (
                "{topic}について、どう思いますか？",
                "{topic}で、気になることはありますか？",
                "{topic}について、少し聞かせてもらえますか？",
            ),
            "continue": (
                "さっき{topic}の話が出ましたが、続きを聞かせてもらえますか？",
                "先ほど{topic}の話が出ましたが、もう少し聞かせてもらえますか？",
                "さっき{topic}の話が出ましたが、続きも教えてもらえますか？",
            ),
            "bridge": (
                "さっき{previous_topic}の話が出ましたが、{topic}についてはどう思いますか？",
                "先ほどの{previous_topic}の話からつなげて、{topic}について聞かせてもらえますか？",
                "{previous_topic}の話に続けて、{topic}で気になることはありますか？",
            ),
        }
        key = ("light_question", mode, previous_topic or "", candidate.topic)
        variant = self._renderer_counts[key]
        self._renderer_counts[key] += 1
        approach = templates[mode][variant % len(templates[mode])].format(
            previous_topic=previous_topic,
            topic=candidate.topic,
        )
        return approach, {
            "mode": mode,
            "turn": history_turn,
            "previous_topic": previous_topic,
            "current_topic": candidate.topic,
        }

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "model": self.config.model,
            "prompt_version": self.prompt_version,
            "generation": {
                "temperature": self.config.temperature,
                "max_tokens": min(self.config.max_tokens, MAX_OUTPUT_TOKENS),
                "think": False,
            },
            "retry_policy": {
                "normal_calls_per_turn": 1,
                "max_attempts": 2,
                "retryable": sorted(_RETRYABLE_PROVIDER_ERRORS),
                "semantic_retry": False,
                "max_calls_per_run": self.config.max_calls,
            },
            "candidate_builder": {
                "name": self.candidate_builder.name,
                "version": self.candidate_builder.version,
                "max_candidates": MAX_PLAN_CANDIDATES,
            },
            "transport": self.client.describe(),
        }

    @staticmethod
    def _body(
        candidates: tuple[RulePlanCandidate, ...],
        config: DeepSeekConfig,
        conversation_balance: Mapping[str, Any],
    ) -> dict[str, Any]:
        projection = {
            "conversation_balance": copy.deepcopy(dict(conversation_balance)),
            "plan_candidates": [copy.deepcopy(dict(item.projection)) for item in candidates],
        }
        system = (
            "共有空間の安全な候補から1つを選び、そのtopicを自然な日本語の発話に必ず文字列のまま含めます。"
            "actは選んだ候補のeligible_actsから1つだけ選び、その意味と形式に一致させます。"
            "actor_affectは-2から+2の範囲の本人の現在の主観的な親しみであり、客観的関係や共有記憶ではありません。"
            "相手とactを選ぶ際の1要因として考慮しますが、選択を強制しません。値そのものを発話に出しません。"
            "+2はその相手へ話を続けやすい状態、0は中立を示します。"
            "latest_pair_exchange.speaker_role=targetは直前に話したのが相手であり、自分が尋ねた・話したとは表現しません。"
            "speaker_role=actorは直前に自分が話したことを表します。"
            "正しさを優先し、自然なら直前と異なるactや使用回数の少ないactも検討しますが、強制ではありません。"
            "会話として適切ならpreferred_actsを優先し、light_questionの連続を避けます。"
            "意味に合わない場合は、その候補の他のeligible_actsを選べます。"
            "light_question、self_disclosure、invitationの最終表現は安全なルールで置換されます。"
            "self_disclosureは入力にある本人の関心、またはこの会話中の現在の希望だけを表し、候補topicだけから恒常的な関心・習慣・経験を主張しません。"
            "approachは実際に相手へ話すセリフそのものにし、〜と尋ねます・質問します・話します・提案します・声をかけます等の行動説明や解説は禁止です。"
            "入力にない事実・記憶・観察は作りません。JSON以外を返しません。"
            "厳密な形式例: {\"plan_id\":\"p1\",\"act\":\"...\",\"approach\":\"...\"}"
        )
        return {
            "model": config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(projection, ensure_ascii=False, separators=(",", ":"))},
            ],
            "stream": False,
            "response_format": {"type": "json_object"},
            "temperature": config.temperature,
            "max_tokens": min(config.max_tokens, MAX_OUTPUT_TOKENS),
            "thinking": {"type": "disabled"},
        }

    @staticmethod
    def _validate_decision(
        value: Any,
        candidates: tuple[RulePlanCandidate, ...],
    ) -> tuple[RulePlanCandidate, str, Any]:
        if not isinstance(value, Mapping) or set(value) != _OUTPUT_KEYS:
            raise CandidateOutputError("missing_or_extra_keys")
        plan_id = value.get("plan_id")
        candidate = next((item for item in candidates if item.plan_id == plan_id), None)
        if candidate is None:
            raise CandidateOutputError("invalid_plan_id")
        act = value.get("act")
        if not isinstance(act, str) or act not in candidate.eligible_acts:
            raise CandidateOutputError("ineligible_act")
        return candidate, act, value.get("approach")

    @staticmethod
    def _validate_approach(
        approach: Any,
        candidate: RulePlanCandidate,
        act: str,
        context: ParticipantContext,
    ) -> str:
        try:
            validated = _validation.OllamaTalkInitiator._validate_output(
                object.__new__(_validation.OllamaTalkInitiator),
                {
                    "target_id": candidate.target_id,
                    "topic": candidate.topic,
                    "act": act,
                    "approach": approach,
                },
                context,
            )
        except _validation.OutputValidationError as exc:
            raise ApproachOutputError(exc.code) from exc
        return validated["approach"]

    def initiate(self, context: ParticipantContext) -> Initiation:
        started = time.monotonic()
        candidates = self.candidate_builder.build(context)
        prior_turns = [turn for turn in self._act_history if turn < context.turn]
        previous_act = self._act_history[max(prior_turns)] if prior_turns else None
        conversation_balance = {
            "previous_act": previous_act,
            "act_counts": _validation._act_counts(self._act_history),
            "preferred_acts": list(preferred_acts(candidates, self._act_history, context.turn)),
        }
        body = self._body(candidates, self.config, conversation_balance)
        before_usage = copy.deepcopy(self.client.describe()["usage"])
        attempts: list[dict[str, Any]] = []
        selected: RulePlanCandidate | None = None
        selected_act: str | None = None
        approach: str | None = None
        fallback_reason: str | None = None
        approach_validation_code: str | None = None
        history_reference: dict[str, Any] | None = None
        outcome: str | None = None

        for attempt in (1, 2):
            attempt_started = time.monotonic()
            try:
                value = self.client.request_json(body)
            except DeepSeekAdapterError as exc:
                attempts.append({
                    "attempt": attempt,
                    "status": "error",
                    "error_category": exc.category,
                    "elapsed_ms": _elapsed_ms(attempt_started),
                })
                fallback_reason = exc.category
                if attempt == 1 and exc.category in _RETRYABLE_PROVIDER_ERRORS:
                    continue
                break
            try:
                selected, selected_act, raw_approach = self._validate_decision(value, candidates)
            except CandidateOutputError as exc:
                attempts.append({
                    "attempt": attempt,
                    "status": "error",
                    "error_category": "semantic_invalid",
                    "validation_code": exc.code,
                    "elapsed_ms": _elapsed_ms(attempt_started),
                })
                fallback_reason = "semantic_invalid"
                # Semantic failures are never retried with the same prompt.
                break
            if not self.allow_natural_wording and selected_act == "light_question":
                approach, history_reference = self._render_light_question(selected)
                approach_validation_code = "light_question_history_policy"
                fallback_reason = "light_question_history_policy"
                outcome = "hybrid"
                attempts.append({
                    "attempt": attempt,
                    "status": "hybrid",
                    "error_category": "light_question_history_policy",
                    "validation_code": "light_question_history_policy",
                    "elapsed_ms": _elapsed_ms(attempt_started),
                })
                # Raw model wording is never retained for safe questions.
                break
            if not self.allow_natural_wording and selected_act == "self_disclosure":
                approach = self._render_safe_approach(selected.topic, selected_act)
                approach_validation_code = "self_disclosure_safety_policy"
                fallback_reason = "self_disclosure_safety_policy"
                outcome = "hybrid"
                attempts.append({
                    "attempt": attempt,
                    "status": "hybrid",
                    "error_category": "self_disclosure_safety_policy",
                    "validation_code": "self_disclosure_safety_policy",
                    "elapsed_ms": _elapsed_ms(attempt_started),
                })
                # Self-disclosure wording is never trusted or retained, even
                # when it would otherwise pass semantic validation.
                break
            if not self.allow_natural_wording and selected_act == "invitation":
                approach = self._render_safe_approach(selected.topic, selected_act)
                approach_validation_code = "invitation_safety_policy"
                fallback_reason = "invitation_safety_policy"
                outcome = "hybrid"
                attempts.append({
                    "attempt": attempt,
                    "status": "hybrid",
                    "error_category": "invitation_safety_policy",
                    "validation_code": "invitation_safety_policy",
                    "elapsed_ms": _elapsed_ms(attempt_started),
                })
                # Invitation wording is never trusted or retained; only the
                # model's bounded plan and act decision is adopted.
                break
            if is_meta_speech_description(raw_approach):
                approach = self._render_safe_approach(selected.topic, selected_act)
                approach_validation_code = "meta_speech_description"
                fallback_reason = "meta_speech_description"
                outcome = "hybrid"
                attempts.append({
                    "attempt": attempt,
                    "status": "hybrid",
                    "error_category": "meta_speech_description",
                    "validation_code": "meta_speech_description",
                    "elapsed_ms": _elapsed_ms(attempt_started),
                })
                # The rejected action description is deliberately not kept.
                break
            try:
                approach = self._validate_approach(raw_approach, selected, selected_act, context)
            except ApproachOutputError as exc:
                approach = self._render_safe_approach(selected.topic, selected_act)
                approach_validation_code = exc.code
                fallback_reason = "approach_semantic_invalid"
                outcome = "hybrid"
                attempts.append({
                    "attempt": attempt,
                    "status": "hybrid",
                    "error_category": "approach_semantic_invalid",
                    "validation_code": exc.code,
                    "elapsed_ms": _elapsed_ms(attempt_started),
                })
                # The rejected model wording is deliberately not retained.
                break
            attempts.append({
                "attempt": attempt,
                "status": "success",
                "elapsed_ms": _elapsed_ms(attempt_started),
            })
            fallback_reason = None
            outcome = "llm"
            break

        after_usage = self.client.describe()["usage"]
        usage = _usage_delta(before_usage, after_usage)
        if selected is not None and selected_act is not None and approach is not None:
            result = Initiation(selected.target_id, selected.topic, approach)
            self._act_history[context.turn] = selected_act
            outcome = outcome or "llm"
            selected_plan_id = selected.plan_id
        else:
            result = RuleTalkInitiator().initiate(context)
            outcome = "fallback"
            selected_plan_id = None

        total_latency_ms = _elapsed_ms(started)
        audit: dict[str, Any] = {
            "turn": context.turn,
            "prompt_input": {"candidate_count": len(candidates), "prompt_version": self.prompt_version},
            "candidate_count": len(candidates),
            "attempts": attempts,
            "selected_plan_id": selected_plan_id,
            "selected_output": (
                {"plan_id": selected_plan_id, "topic": selected.topic, "act": selected_act}
                if outcome in {"llm", "hybrid"} and selected is not None
                else None
            ),
            "adopted_plan": (
                {"plan_id": selected_plan_id, "topic": selected.topic, "act": selected_act}
                if outcome in {"llm", "hybrid"} and selected is not None
                else None
            ),
            "final_approach": result.approach,
            "outcome": outcome,
            "decision_source": "llm" if outcome in {"llm", "hybrid"} else "rule_fallback",
            "approach_source": (
                "llm" if outcome == "llm" else "rule_renderer" if outcome == "hybrid" else "rule_fallback"
            ),
            "approach_validation_code": approach_validation_code,
            "history_reference": history_reference,
            "fallback_stage": (
                None if outcome == "llm" else "approach_renderer" if outcome == "hybrid" else "selection_utterance"
            ),
            "fallback_reason": fallback_reason,
            "token_usage": usage,
            "total_latency": total_latency_ms,
            "total_latency_ms": total_latency_ms,
        }
        if outcome in {"llm", "hybrid"}:
            audit["conversation_act"] = selected_act
        self._turns.append(audit)
        return result

    propose = initiate

    def audit_snapshot(self) -> dict[str, Any]:
        llm = sum(item["outcome"] == "llm" for item in self._turns)
        hybrid = sum(item["outcome"] == "hybrid" for item in self._turns)
        fallback = sum(item["outcome"] == "fallback" for item in self._turns)
        fallback_reasons = Counter(
            item.get("fallback_reason") or "unknown"
            for item in self._turns
            if item["outcome"] == "fallback"
        )
        return {
            "adapter": {"name": self.name, "version": self.version},
            "model": self.config.model,
            "prompt_version": self.prompt_version,
            "generation": {
                "temperature": self.config.temperature,
                "max_tokens": min(self.config.max_tokens, MAX_OUTPUT_TOKENS),
                "think": False,
            },
            "retry_policy": self.describe()["retry_policy"],
            "candidate_builder": self.describe()["candidate_builder"],
            "transport": self.client.describe(),
            "turns": copy.deepcopy(self._turns),
            "summary": {
                "turns": len(self._turns),
                "llm": llm,
                "full_llm_wording": llm,
                "hybrid": hybrid,
                "fallback": fallback,
                "whole_fallback": fallback,
                "attempts": sum(len(item["attempts"]) for item in self._turns),
                "act_counts": _validation._act_counts(self._act_history),
                "fallback_reasons": dict(sorted(fallback_reasons.items())),
            },
        }


class DeepSeekNaturalConversationInitiator(DeepSeekRuleCandidateInitiator):
    """Opt-in wording variant; target/topic grounding stays rule-owned."""

    name = "deepseek-natural-conversation-initiator"
    version = "m14_natural_conversation_initiator_v1"

    @staticmethod
    def _body(
        candidates: tuple[RulePlanCandidate, ...],
        config: DeepSeekConfig,
        conversation_balance: Mapping[str, Any],
    ) -> dict[str, Any]:
        body = DeepSeekRuleCandidateInitiator._body(candidates, config, conversation_balance)
        body["messages"][0]["content"] = body["messages"][0]["content"].replace(
            "light_question、self_disclosure、invitationの最終表現は安全なルールで置換されます。",
            "4種類のactの最終表現は検証に通ったモデル文を採用します。",
        )
        body["messages"][0]["content"] += (
            "personaは話し方の傾向です。全項目を毎回言わず、文言をそのまま自己紹介せず、実際の直近発言・履歴を優先します。"
            "personaのpurposeとinterestsは経歴・経験の事実ではありません。"
            "入力にない過去・現在の具体的出来事、職業経験、習慣、実績を本人の事実として作りません。"
            "新しいアイデアは「〜なら面白そう」「〜してみたい」「〜はどうでしょう」のような希望・仮定・提案として表現します。"
            "実在する地域イベントや場所が起きたと断定しません。実際の履歴にある内容だけを引き継ぎます。"
        )
        return body

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Force this behavior for the named pilot so callers cannot
        # accidentally construct a natural provider with legacy semantics.
        kwargs["allow_natural_wording"] = True
        if "candidate_builder" not in kwargs:
            kwargs["candidate_builder"] = RulePlanCandidateBuilder(include_persona=True)
        super().__init__(*args, **kwargs)


__all__ = [
    "ApproachOutputError",
    "CandidateOutputError",
    "DeepSeekRuleCandidateInitiator",
    "DeepSeekNaturalConversationInitiator",
    "MAX_PLAN_CANDIDATES",
    "MAX_OUTPUT_TOKENS",
    "PROMPT_VERSION",
    "NATURAL_PROMPT_VERSION",
    "RECENT_TOPIC_WINDOW",
    "RulePlanCandidate",
    "RulePlanCandidateBuilder",
    "is_meta_speech_description",
    "preferred_acts",
    "render_rule_approach",
]
