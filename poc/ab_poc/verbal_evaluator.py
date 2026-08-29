"""Replaceable verbal evaluation boundary for bounded conversation episodes.

The baseline in this module is deliberately small and deterministic.  It is
not intended to infer a latent relationship from language; it provides a
stable score while a later human or cloud-LLM evaluator is being designed.
"""

from __future__ import annotations

import re
import unicodedata

from .contracts import ConversationEpisode, VerbalEpisodeEvaluation


# A few high-precision continuation cues are enough for the baseline.  The
# list is intentionally short and can be replaced without changing the DTO or
# engine boundary.  Japanese and English cues keep fixture tests readable.
_CONTINUATION_CUES = (
    "また", "次", "ぜひ", "楽しみ", "面白", "話したい", "やってみたい",
    "またね", "again", "next", "let's", "would like", "interesting",
)


def _normalized_text(text: str) -> str:
    """Normalize wording enough to identify an exact adjacent echo."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)


class DeterministicVerbalEpisodeEvaluator:
    """Evaluate a completed episode using transparent reaction/text heuristics.

    Relationship movement is taken from the provider's reaction label.  End
    momentum starts at one for a positive reaction, gains one for reciprocal
    development (both speakers add a non-repeated contribution), and gains
    one for an explicit continuation cue near the end.  Four utterances alone
    never add a point.  Adjacent exact echoes cap momentum at one, so a forced
    four-line loop cannot look like sustained interest. Scores are clamped to
    the public 0..3 range and no simulation state is changed.
    """

    name = "deterministic-verbal-episode-evaluator"
    version = "m16_verbal_baseline_v1"

    def evaluate(self, episode: ConversationEpisode) -> VerbalEpisodeEvaluation:
        if not isinstance(episode, ConversationEpisode):
            raise ValueError("verbal evaluator requires a ConversationEpisode")
        if not episode.ended:
            raise ValueError("verbal evaluator requires a completed episode")

        utterances = episode.utterances
        normalized = tuple(_normalized_text(item.text) for item in utterances)
        movement = {"positive": 1, "neutral": 0, "misaligned": -1}[episode.reaction]
        evidence = [utterances[1].sequence]
        momentum = 1 if episode.reaction == "positive" else 0

        echo_sequences = tuple(
            (utterances[index].sequence, utterances[index + 1].sequence)
            for index in range(len(utterances) - 1)
            if normalized[index] and normalized[index] == normalized[index + 1]
        )
        if len(utterances) >= 4 and not echo_sequences:
            # Each speaker must contribute at least twice, with the second
            # contribution materially different from the first. This is a
            # small proxy for reciprocal development, not a language model.
            speakers = tuple(dict.fromkeys(item.speaker for item in utterances))
            developed = all(
                len([index for index, item in enumerate(utterances) if item.speaker == speaker]) >= 2
                and normalized[
                    [index for index, item in enumerate(utterances) if item.speaker == speaker][0]
                ] != normalized[
                    [index for index, item in enumerate(utterances) if item.speaker == speaker][1]
                ]
                for speaker in speakers
            )
            if developed and len(speakers) >= 2:
                momentum += 1

        cue_sequences = tuple(
            item.sequence
            for item in utterances[-2:]
            if any(cue in item.text.casefold() for cue in _CONTINUATION_CUES)
        )
        if cue_sequences:
            momentum += 1
            evidence.append(cue_sequences[-1])

        if echo_sequences:
            momentum = min(momentum, 1)
            evidence.extend(echo_sequences[0])

        # Keep evidence ordered and unique if the response is also the final
        # utterance in a two-line episode.
        evidence = list(dict.fromkeys(evidence))
        momentum = min(3, momentum)
        reaction_reason = {
            "positive": "反応が前向きだったため関係の動きを+1",
            "neutral": "反応が中立だったため関係の動きを0",
            "misaligned": "反応が噛み合わなかったため関係の動きを-1",
        }[episode.reaction]
        momentum_reason = {
            0: "終了時の継続サインは弱い",
            1: "終了時に最低限の関心が残った",
            2: "会話の往復または継続サインが確認できる",
            3: "会話の往復と次につながるサインが確認できる",
        }[momentum]
        if echo_sequences:
            momentum_reason = "隣接発言の反復があるため盛り上がりを抑制"

        evidence = sorted(set(evidence))
        return VerbalEpisodeEvaluation(
            relationship_movement=movement,
            end_momentum=momentum,
            reason=f"{reaction_reason}。{momentum_reason}。",
            evidence_sequences=tuple(evidence),
        )


# The rule-oriented alias makes the evaluator easy to discover for callers
# that use the existing RuleParticipantProvider naming convention.
RuleVerbalEpisodeEvaluator = DeterministicVerbalEpisodeEvaluator


__all__ = [
    "DeterministicVerbalEpisodeEvaluator",
    "RuleVerbalEpisodeEvaluator",
]
