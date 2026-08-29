from __future__ import annotations

import json
import os
from typing import Protocol
from urllib.request import Request, urlopen

from .models import Candidate, Conversation, Decision, DecisionContext, Persona


class DecisionProvider(Protocol):
    name: str

    def decide(self, context: DecisionContext) -> Decision: ...


class ConversationProvider(Protocol):
    name: str

    def create(
        self,
        *,
        actor: Persona,
        target: Persona,
        decision: Decision,
        shared_interests: tuple[str, ...],
        venue_topic: str,
    ) -> Conversation: ...


class RuleBasedDecisionProvider:
    """Uses precomputed candidate scores; deterministic when the engine seed is fixed."""

    name = "rule_based"

    def decide(self, context: DecisionContext) -> Decision:
        if not context.candidates:
            return Decision("step_back", None, "まだ話しかける相手がいない")

        best: Candidate = max(context.candidates, key=lambda item: item.score)
        if best.relation_strength >= 2 and best.score >= 6:
            return Decision("continue", best.participant_id, "前の会話をもう少し続けたい")
        if best.score >= 5:
            return Decision("talk", best.participant_id, "話してみるきっかけがある")
        return Decision("step_back", None, "今は様子を見たい")


class TemplateConversationProvider:
    """Short replay-friendly dialogue. It is presentation, not metric input."""

    name = "template"

    def create(
        self,
        *,
        actor: Persona,
        target: Persona,
        decision: Decision,
        shared_interests: tuple[str, ...],
        venue_topic: str,
    ) -> Conversation:
        if venue_topic == "机の上の素材":
            return self._workshop_conversation(actor, target, decision, venue_topic)
        if venue_topic == "さっきの発表":
            return self._event_conversation(actor, target, decision, venue_topic)

        topic = shared_interests[0] if shared_interests else venue_topic
        if decision.action == "continue":
            first = f"さっきの{topic}、もう少し聞いてもいい？"
            second = f"もちろん。私も{topic}の話を続けたかった。"
        else:
            first = f"この{topic}、気になっていて。よかったら話しませんか？"
            second = f"いいですね。私も{topic}に興味があります。"
        return Conversation(
            topic=topic,
            speeches=(
                {"speaker_id": actor.id, "speaker_name": actor.name, "text": first},
                {"speaker_id": target.id, "speaker_name": target.name, "text": second},
            ),
        )

    @staticmethod
    def _event_conversation(
        actor: Persona, target: Persona, decision: Decision, venue_topic: str
    ) -> Conversation:
        if decision.action == "continue":
            first = "さっきの発表の話、もう少し聞いてもいい？"
            second = "もちろん。印象に残ったところを話しましょう。"
        else:
            first = "さっきの発表、印象に残りましたね。"
            second = "はい。私もそこが気になっていました。"
        return Conversation(
            topic=venue_topic,
            speeches=(
                {"speaker_id": actor.id, "speaker_name": actor.name, "text": first},
                {"speaker_id": target.id, "speaker_name": target.name, "text": second},
            ),
        )

    @staticmethod
    def _workshop_conversation(
        actor: Persona, target: Persona, decision: Decision, venue_topic: str
    ) -> Conversation:
        if decision.action == "continue":
            first = "さっき試した作り方、もう少し教えてもらってもいい？"
            second = "もちろん。一緒にもう少し試してみましょう。"
        else:
            first = "この素材、どう使うか気になりますね。"
            second = "ですね。一緒に試してみませんか？"
        return Conversation(
            topic=venue_topic,
            speeches=(
                {"speaker_id": actor.id, "speaker_name": actor.name, "text": first},
                {"speaker_id": target.id, "speaker_name": target.name, "text": second},
            ),
        )


class OllamaDecisionProvider:
    """Asks a local Ollama model for a decision and falls back on any invalid result."""

    name = "ollama"
    default_base_url = "http://localhost:11434"
    default_model = "qwen3:4b-instruct-2507-q4_K_M"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 10.0,
        fallback: DecisionProvider | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or self.default_base_url).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL") or self.default_model
        self.timeout = timeout
        self.fallback = fallback or RuleBasedDecisionProvider()
        self.fallback_count = 0

    def decide(self, context: DecisionContext) -> Decision:
        if not context.candidates:
            return self._fallback(context)

        try:
            decision = self._request_decision(context)
            self._validate_decision(decision, context)
            return decision
        except Exception:
            # An optional local model must never prevent the deterministic simulation from finishing.
            return self._fallback(context)

    def _request_decision(self, context: DecisionContext) -> Decision:
        candidate_ids = [candidate.participant_id for candidate in context.candidates]
        schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["talk", "continue", "step_back"]},
                "target_id": {
                    "anyOf": [
                        {"type": "string", "enum": candidate_ids},
                        {"type": "null"},
                    ]
                },
                "reason": {"type": "string", "minLength": 1, "maxLength": 80},
            },
            "required": ["action", "target_id", "reason"],
            "additionalProperties": False,
        }
        context_payload = {
            "step": context.step,
            "actor": {
                "id": context.actor.id,
                "name": context.actor.name,
                "interests": list(context.actor.interests),
                "visit_purpose": context.actor.visit_purpose,
                "conversation_style": context.actor.conversation_style,
            },
            "scenario": {
                "venue": context.scenario.venue_label,
                "description": context.scenario.venue_description,
                "topic": context.scenario.venue_topic,
            },
            "candidates": [
                {
                    "participant_id": candidate.participant_id,
                    "name": candidate.name,
                    "score": candidate.score,
                    "shared_interests": list(candidate.shared_interests),
                    "relation_strength": candidate.relation_strength,
                    "interests": list(candidate.interests),
                    "visit_purpose": candidate.visit_purpose,
                    "conversation_style": candidate.conversation_style,
                }
                for candidate in context.candidates
            ],
        }
        payload = {
            "model": self.model,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Choose one social action for the actor. Use continue only for an existing "
                        "relationship (relation_strength >= 2). Return exactly the requested JSON object."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(context_payload, ensure_ascii=False),
                },
            ],
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))

        content = response_payload["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("Ollama response content must be a JSON string")
        raw_decision = json.loads(content)
        if not isinstance(raw_decision, dict) or set(raw_decision) != {"action", "target_id", "reason"}:
            raise ValueError("Ollama decision must contain exactly action, target_id, and reason")
        return Decision(
            action=raw_decision["action"],
            target_id=raw_decision["target_id"],
            reason=raw_decision["reason"],
        )

    @staticmethod
    def _validate_decision(decision: Decision, context: DecisionContext) -> None:
        if decision.action not in {"talk", "continue", "step_back"}:
            raise ValueError("Unsupported action")
        if not isinstance(decision.reason, str) or not decision.reason.strip() or len(decision.reason) > 80:
            raise ValueError("Reason must be a short, non-empty string")
        if decision.action == "step_back":
            if decision.target_id is not None:
                raise ValueError("step_back cannot have a target")
            return
        if not isinstance(decision.target_id, str):
            raise ValueError("talk and continue require a target")
        candidates = {candidate.participant_id: candidate for candidate in context.candidates}
        if decision.target_id not in candidates:
            raise ValueError("Target is not an eligible candidate")
        if decision.action == "continue" and candidates[decision.target_id].relation_strength < 2:
            raise ValueError("continue requires an existing relationship")

    def _fallback(self, context: DecisionContext) -> Decision:
        self.fallback_count += 1
        return self.fallback.decide(context)


class OllamaConversationProvider:
    """Reserved adapter boundary. Implemented after the rule-based POC is validated."""

    name = "ollama"

    def create(self, **_: object) -> Conversation:
        raise NotImplementedError("Ollama provider is intentionally deferred until rule-based validation.")
