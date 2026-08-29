"""M19 opt-in autonomous conversation providers.

The autonomous boundary gives DeepSeek the conversational choice (target,
topic, act, and wording).  The engine still owns world integrity: the target
must be one of the real candidates and the output must satisfy the small
JSON/length contract.  No evaluator or relationship state is sent to the
model.
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Mapping
from typing import Any

from .contracts import (
    ConversationEpisode,
    Initiation,
    ParticipantContext,
    ResponseContext,
    ResponseOutput,
)
from .deepseek_adapter import DeepSeekAdapterError, DeepSeekConfig, DeepSeekJsonClient, redact_known_people
from .deepseek_response import DeepSeekResponseProvider, MAX_REPLY_LENGTH, _redact as _response_redact
from .deepseek_rule_candidate_initiator import DeepSeekRuleCandidateInitiator
from .domain import OnlineKnown
from .rule_providers import RuleTalkInitiator


PROMPT_VERSION = "m19_deepseek_autonomous_episode_v1"
MAX_TOPIC_LENGTH = 80
MAX_APPROACH_LENGTH = 160
MAX_ACT_LENGTH = 40
MAX_OUTPUT_TOKENS = 96
_OUTPUT_KEYS = {"target_id", "topic", "act", "approach"}
FACT_BOUNDARY_INSTRUCTION = (
    "事実境界: persona/interests/stanceは話し方・関心・価値観の手がかりです。"
    "日常会話では設定に沿った小さな補完、好み、身近な経験を自然に表現して構いません。"
    "入力に含まれる実際の会話ログとonline memoryは引き継げます。"
    "入力と明確に矛盾すること、世界観を壊す突拍子もない経歴・事件・実績、"
    "他人の非公開属性の断定は避けます。創作は『〜なら』『〜してみたい』『〜はどうでしょう』など希望・仮定・提案として表現できます。"
)
RHYTHM_INSTRUCTION = (
    "会話は自然な区切りを優先し、十分に答えた、または新しい問い・提案がなければ引き延ばしません。"
    "毎回最大の発話数まで続ける必要はありません。"
)


class AutonomousOutputError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _safe_public_context(context: ParticipantContext) -> dict[str, Any]:
    """Expose only actor-owned memory and public candidate cards."""

    candidates = []
    for index, person in enumerate(context.candidates, start=1):
        state = context.pair_states.get((min(context.actor.id, person.id), max(context.actor.id, person.id)))
        candidates.append({
            # Candidate keys are stable within this request and keep real
            # participant IDs out of the cloud prompt.
            "target_id": f"c{index}",
            "interests": list(person.interests),
            "stance": person.stance.value,
            "online_known": (
                state.online_known.value if state is not None else OnlineKnown.NONE.value
            ),
            "affect": context.outgoing_affect.get(person.id, 0),
        })
    actor_history = [
        item for item in context.dialogue_history
        if item.actor == context.actor.id or item.target == context.actor.id
    ]
    history = [
        {
            "turn": item.turn,
            "actor": item.actor,
            "target": item.target,
            "topic": item.topic,
            "approach": item.approach,
            "reply": item.reply,
            "reaction": item.reaction,
        }
        for item in actor_history[-4:]
    ]
    return redact_known_people({
        "turn": context.turn,
        "actor": {
            "id": context.actor.id,
            "name": context.actor.name,
            "persona": context.actor.persona,
            "interests": list(context.actor.interests),
            "stance": context.actor.stance.value,
        },
        "candidates": candidates,
        "recent_dialogue": history,
        "online_memories": list(context.private_online_projection),
        "last_target": context.last_target,
    }, context, context.actor.id)


def _validate_output(value: Any, context: ParticipantContext) -> tuple[str, str, str, str]:
    if not isinstance(value, Mapping) or set(value) != _OUTPUT_KEYS:
        raise AutonomousOutputError("missing_or_extra_keys")
    target_id, topic, act, approach = (value.get(key) for key in ("target_id", "topic", "act", "approach"))
    candidate_ids = {item.id for item in context.candidates}
    if type(target_id) is not str:
        raise AutonomousOutputError("invalid_target")
    if target_id.startswith("c") and target_id[1:].isdigit() and 1 <= int(target_id[1:]) <= len(context.candidates):
        target_id = context.candidates[int(target_id[1:]) - 1].id
    if target_id not in candidate_ids:
        raise AutonomousOutputError("invalid_target")
    for field, value_, limit in (("topic", topic, MAX_TOPIC_LENGTH), ("act", act, MAX_ACT_LENGTH), ("approach", approach, MAX_APPROACH_LENGTH)):
        if type(value_) is not str or not value_.strip() or value_ != value_.strip():
            raise AutonomousOutputError(f"invalid_{field}")
        if len(value_) > limit:
            raise AutonomousOutputError(f"{field}_too_long")
    return target_id, topic, act, approach


def _validate_continuation(value: Any) -> tuple[str, bool]:
    """Validate one role-owned follow-up and its continue/end decision."""

    if not isinstance(value, Mapping) or set(value) != {"text", "continue"}:
        raise AutonomousOutputError("invalid_continuation_keys")
    text, should_continue = value["text"], value["continue"]
    if type(text) is not str or not text.strip() or text != text.strip():
        raise AutonomousOutputError("invalid_continuation_text")
    if len(text) > MAX_APPROACH_LENGTH:
        raise AutonomousOutputError("continuation_text_too_long")
    if type(should_continue) is not bool:
        raise AutonomousOutputError("invalid_continuation_decision")
    return text, should_continue


class DeepSeekAutonomousConversationInitiator:
    """Let DeepSeek choose a real candidate, free topic, act, and wording."""

    name = "deepseek-autonomous-conversation-initiator"
    version = PROMPT_VERSION

    def __init__(self, config: DeepSeekConfig | None = None, *, client: DeepSeekJsonClient | None = None, transport: Any | None = None) -> None:
        if client is not None and (config is not None or transport is not None):
            raise ValueError("client cannot be combined with config or transport")
        self.client = client or DeepSeekJsonClient(config, transport=transport)
        self.config = self.client.config
        self._turns: list[dict[str, Any]] = []

    @staticmethod
    def _body(context: ParticipantContext, config: DeepSeekConfig) -> dict[str, Any]:
        return {
            "model": config.model,
            "messages": [
                {"role": "system", "content": (
                    "JSON objectを厳密に返します。キーはtarget_id, topic, act, approachだけです。"
                    "target_idはcandidatesにある実在のidから1つ選びます。topicは短い自由な話題、actは会話行為です。"
                    "approachは相手に実際に話す短い一文で、選んだtopicに自然に関係させます。"
                    "dialogue historyに繰り返し現れる話題は、同じ文面を再利用せず新しい角度・質問・具体例へ進めます。"
                    "候補にない対象を作らず、入力済みのprior relationship、dialogue history、online memoryを参照します。"
                    "personaから自然に生じる感想、好きなもの、『やってみたい』という希望、仮定、提案、創作的なアイデアは自由に表現できます。"
                    "personaは話し方・好み・価値観の手がかりであり、説明やJSON以外は返しません。"
                    + FACT_BOUNDARY_INSTRUCTION
                    + RHYTHM_INSTRUCTION
                )},
                {"role": "user", "content": json.dumps(_safe_public_context(context), ensure_ascii=False, separators=(",", ":"))},
            ],
            "stream": False,
            "response_format": {"type": "json_object"},
            "temperature": config.temperature,
            "max_tokens": min(config.max_tokens, MAX_OUTPUT_TOKENS),
            "thinking": {"type": "disabled"},
        }

    def initiate(self, context: ParticipantContext) -> Initiation:
        started = time.monotonic()
        try:
            value = self.client.request_json(self._body(context, self.config))
            target_id, topic, act, approach = _validate_output(value, context)
            outcome, reason = "deepseek", None
            result = Initiation(target_id, topic, approach)
        except DeepSeekAdapterError as exc:
            outcome, reason, act = "fallback", exc.category, None
            result = RuleTalkInitiator().initiate(context)
        except AutonomousOutputError:
            outcome, reason, act = "fallback", "semantic_invalid", None
            result = RuleTalkInitiator().initiate(context)
        except Exception:
            outcome, reason, act = "fallback", "context_invalid", None
            result = RuleTalkInitiator().initiate(context)
        audit_item = {
            "turn": context.turn,
            "outcome": outcome,
            "fallback_reason": reason,
            "selected_target": result.target_id,
            "topic": result.topic,
            "act": act,
            "approach": result.approach,
            "latency_ms": max(0, int(round((time.monotonic() - started) * 1000))),
        }
        self._turns.append(audit_item)
        return result

    propose = initiate

    def continue_episode(
        self,
        context: ParticipantContext,
        initiation: Initiation,
        episode: ConversationEpisode,
    ) -> tuple[str, bool]:
        """Generate only the actor's next utterance, or signal end."""

        target = next((item for item in context.candidates if item.id == initiation.target_id), None)
        if target is None:
            raise AutonomousOutputError("continuation_target_missing")
        pair = (min(context.actor.id, target.id), max(context.actor.id, target.id))
        state = context.pair_states.get(pair)
        target_memory = next(
            (memory for memory in context.online_memories if memory.counterpart_id == target.id),
            None,
        )
        projection = redact_known_people({
            "actor": {
                "persona": context.actor.persona,
                "interests": list(context.actor.interests),
                "stance": context.actor.stance.value,
                "outgoing_affect_to_target": context.outgoing_affect.get(target.id, 0),
            },
            "selected_target": {
                "interests": list(target.interests),
                "stance": target.stance.value,
                "online_known": state.online_known.value if state is not None else OnlineKnown.NONE.value,
                "online_memory": (
                    {
                        "kind": target_memory.kind,
                        "topic": target_memory.topic,
                        "exchange_count": target_memory.exchange_count,
                        "summary": target_memory.summary,
                    }
                    if target_memory is not None else None
                ),
            },
            "current_initiation": {
                "topic": initiation.topic,
                "opening": initiation.approach,
            },
            "remaining_utterances": 4 - len(episode.utterances),
            "episode": [
                {
                    "sequence": item.sequence,
                    "speaker_role": "actor" if item.speaker == context.actor.id else "target",
                    "text": item.text,
                }
                for item in episode.utterances
            ],
        }, context, target.id)
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": (
                    "JSON objectを厳密に返します。キーはtextとcontinueだけです。"
                    "textはactor本人の短い次の一発話です。continueはこの会話を続けるかどうかです。"
                    "直前の発話をそのままコピーしたり言い換えたりせず、新しい反応・具体的詳細・質問を1つ加えます。"
                    "自然な間や話題のまとまりではcontinue=falseにします。会話は常に続ける必要はありません。"
                    "remaining_utterancesが1なら、未回答の質問を残さない短い締めにしてcontinue=falseにします。"
                    "targetの発話やtargetのpersonaを代筆せず、事実境界の指示に従います。"
                    + FACT_BOUNDARY_INSTRUCTION
                    + RHYTHM_INSTRUCTION
                )},
                {"role": "user", "content": json.dumps(projection, ensure_ascii=False, separators=(",", ":"))},
            ],
            "stream": False,
            "response_format": {"type": "json_object"},
            "temperature": self.config.temperature,
            "max_tokens": min(self.config.max_tokens, MAX_OUTPUT_TOKENS),
            "thinking": {"type": "disabled"},
        }
        try:
            return _validate_continuation(self.client.request_json(body))
        except Exception:
            # The orchestrator ends on provider failure and keeps the valid
            # opening/reply; it must not invent a fallback utterance here.
            raise

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "model": self.config.model, "prompt_version": PROMPT_VERSION, "transport": self.client.describe()}

    def audit_snapshot(self) -> dict[str, Any]:
        return {"adapter": {"name": self.name, "version": self.version}, "prompt_version": PROMPT_VERSION, "transport": self.client.describe(), "turns": copy.deepcopy(self._turns)}


class DeepSeekAutonomousResponseProvider(DeepSeekResponseProvider):
    """Natural responder with the M15 autonomy prompt and private persona."""

    name = "deepseek-autonomous-response-provider"
    version = PROMPT_VERSION

    @staticmethod
    def _body(context: ResponseContext, initiation: Initiation, config: DeepSeekConfig) -> dict[str, Any]:
        if initiation.target_id != context.responder.id:
            raise ValueError("initiation target must match response context responder")
        pair_history = [
            {
                "turn": event.turn,
                "speaker_role": "responder" if event.actor == context.responder.id else "initiator",
                "topic": event.topic,
                "approach": event.approach,
                "reaction": event.response.value,
            }
            for event in context.pair_events
        ]
        # Conversation text is pair-scoped. The public history projection is
        # useful to other presentation layers, but must not steer this pair's
        # responder with an unrelated exchange.
        public_history = context.dialogue_history

        def role(participant_id: str) -> str:
            if participant_id == context.responder.id:
                return "responder"
            if participant_id == context.initiator.id:
                return "current_initiator"
            return "other"

        dialogue_history = [
            {
                "turn": item.turn,
                "speaker_role": role(item.actor),
                "approach_speaker_role": role(item.actor),
                "topic": item.topic,
                "approach": item.approach,
                "reply": item.reply,
                "reply_speaker_role": role(item.target),
                "reaction": item.reaction,
            }
            for item in public_history[-4:]
        ]
        memory = None
        if context.online_memory is not None:
            memory = {
                "kind": context.online_memory.kind,
                "topic": context.online_memory.topic,
                "exchange_count": context.online_memory.exchange_count,
                "summary": context.online_memory.summary,
            }
        projection_source = {
                "turn": context.turn,
                "responder": {
                    "persona": context.responder.persona,
                    "interests": list(context.responder.interests),
                    "stance": context.responder.stance.value,
                    "affect_to_initiator": context.affect,
                },
                "initiation": {"topic": initiation.topic, "approach": initiation.approach},
                "dialogue_history": dialogue_history,
                "pair_history": pair_history,
                "online_memory": memory,
        }
        # The pair redactor knows the two active people. Extend it with the
        # engine's in-memory public roster so third-party names/IDs embedded
        # in a quoted utterance cannot cross the provider boundary.
        for token in sorted(context.public_participant_tokens, key=len, reverse=True):
            if token not in {
                context.responder.id,
                context.responder.name,
                context.initiator.id,
                context.initiator.name,
            }:
                def redact_third_party(value: Any) -> Any:
                    if isinstance(value, Mapping):
                        return {key: redact_third_party(item) for key, item in value.items()}
                    if isinstance(value, list):
                        return [redact_third_party(item) for item in value]
                    if isinstance(value, str):
                        return value.replace(token, "他の参加者")
                    return value
                projection_source = redact_third_party(projection_source)
        projection = _response_redact(projection_source, context)
        return {
            "model": config.model,
            "messages": [
                {"role": "system", "content": (
                    "JSON objectを厳密に返します。キーはreply、reaction、continueの3つだけです。"
                    "replyは短い自然な返答、reactionはpositive、neutral、misalignedのいずれかです。"
                    "responderのpersonaは話し方・好み・価値観の手がかりです。personaから自然に生じる感想、"
                    "『好き』『やってみたい』という希望、仮定、提案、創作的なアイデアは自由に表現できます。"
                    "入力にない相手の非公開personaは推測して渡しません。"
                    "reactionは現在のaffectと会話内容を踏まえて選びます。JSON以外は返しません。"
                    "continueはこの会話を続ける場合true、終える場合falseです。"
                    "replyは1〜2文、80文字程度までにし、結論を先にして説明を付け足しません。"
                    "直前の発話をそのままコピーしたり言い換えたりせず、新しい反応・具体的詳細・質問を1つ加えます。"
                    "自然な間や話題のまとまりではcontinue=falseにします。会話は常に続ける必要はありません。"
                    + FACT_BOUNDARY_INSTRUCTION
                    + RHYTHM_INSTRUCTION
                )},
                {"role": "user", "content": json.dumps(projection, ensure_ascii=False, separators=(",", ":"))},
            ],
            "stream": False,
            "response_format": {"type": "json_object"},
            "temperature": config.temperature,
            "max_tokens": min(config.max_tokens, MAX_REPLY_LENGTH),
            "thinking": {"type": "disabled"},
        }

    @staticmethod
    def _validate_autonomous_response(value: Any) -> ResponseOutput:
        keys = set(value) if isinstance(value, Mapping) else set()
        if keys not in ({"reply", "reaction"}, {"reply", "reaction", "continue"}):
            raise ValueError("invalid autonomous response keys")
        reply, reaction = value.get("reply"), value.get("reaction")
        if type(reply) is not str or not reply.strip() or reply != reply.strip() or len(reply) > MAX_REPLY_LENGTH:
            raise ValueError("invalid autonomous reply")
        if type(reaction) is not str or reaction not in {"positive", "neutral", "misaligned"}:
            raise ValueError("invalid autonomous reaction")
        decision = value.get("continue")
        if decision is not None and type(decision) is not bool:
            raise ValueError("invalid autonomous continuation decision")
        return ResponseOutput(reply, reaction, continue_conversation=decision)

    def respond(self, context: ResponseContext, initiation: Initiation) -> ResponseOutput:
        try:
            output = self._validate_autonomous_response(self.client.request_json(self._body(context, initiation, self.config)))
        except DeepSeekAdapterError as exc:
            return self._fallback(context, exc.category)
        except Exception:
            return self._fallback(context, "semantic_invalid")
        self.last_response_output = output
        self._turns.append({"turn": context.turn, "outcome": "deepseek", "attempts": [{"attempt": 1, "status": "success"}], "fallback_reason": None, "reaction": output.reaction, "continue": output.continue_conversation})
        return output

    def continue_episode(self, context: ResponseContext, initiation: Initiation, episode: ConversationEpisode) -> tuple[str, bool]:
        """Generate only the responder's next utterance, or signal end."""
        projection = _response_redact({
            "turn": context.turn,
            "responder": {"persona": context.responder.persona, "interests": list(context.responder.interests), "stance": context.responder.stance.value, "affect_to_initiator": context.affect},
            "initiation": {"topic": initiation.topic, "approach": initiation.approach},
            "remaining_utterances": 4 - len(episode.utterances),
            "episode": [
                {
                    "sequence": item.sequence,
                    "speaker_role": "responder" if item.speaker == context.responder.id else "initiator",
                    "text": item.text,
                }
                for item in episode.utterances
            ],
        }, context)
        body = {"model": self.config.model, "messages": [
            {"role": "system", "content": "JSON objectを厳密に返します。キーはtextとcontinueだけです。textはresponder本人の短い次の一発話です。continueは会話を続けるかどうかです。直前の発話をそのままコピーしたり言い換えたりせず、新しい反応・具体的詳細・質問を1つ加えます。自然な間や話題のまとまりではcontinue=falseにします。会話は常に続ける必要はありません。remaining_utterancesが1なら、未回答の質問を残さない短い締めにしてcontinue=falseにします。initiatorの発話やpersonaを代筆せず、事実境界の指示に従います。" + RHYTHM_INSTRUCTION + FACT_BOUNDARY_INSTRUCTION},
            {"role": "user", "content": json.dumps(projection, ensure_ascii=False, separators=(",", ":"))},
        ], "stream": False, "response_format": {"type": "json_object"}, "temperature": self.config.temperature, "max_tokens": min(self.config.max_tokens, MAX_REPLY_LENGTH), "thinking": {"type": "disabled"}}
        try:
            return _validate_continuation(self.client.request_json(body))
        except Exception:
            raise

    def describe(self) -> dict[str, Any]:
        result = super().describe()
        result.update({"name": self.name, "version": self.version, "prompt_version": PROMPT_VERSION})
        return result


__all__ = [
    "AutonomousOutputError",
    "DeepSeekAutonomousConversationInitiator",
    "DeepSeekAutonomousResponseProvider",
    "MAX_TOPIC_LENGTH",
    "PROMPT_VERSION",
]
