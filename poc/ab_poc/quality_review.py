"""Small, blindable human-review layer for M5.

This module deliberately does not decide whether an LLM conversation is
good.  It only extracts the information a reviewer needs, validates human
labels, and aggregates those labels.  Keeping extraction and judgement
separate makes it possible to replace a human reviewer with another LLM
later without changing the simulation run format.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any


REVIEW_DIMENSIONS = (
    "target_choice",
    "topic_fit",
    "history_factual_consistency",
    "conversation_naturalness",
)

_BLIND_FORBIDDEN_KEYS = frozenset(
    {
        "condition",
        "condition_id",
        "conditions",
        "changed_field",
        "metrics",
        "evaluator",
        "relationship_evaluator",
        "online_known_pairs",
    }
)


class FatalIssue(str, Enum):
    """A problem that makes a turn unusable regardless of its score."""

    NONE = "none"
    UNSUPPORTED_MEMORY = "unsupported_memory"
    UNSUPPORTED_ATTRIBUTE_OR_RELATION = "unsupported_attribute_or_relation"
    WRONG_ADDRESSEE = "wrong_addressee"
    PROMPT_CONTRADICTION = "prompt_contradiction"
    META_LEAK = "meta_leak"


_FATAL_VALUES = frozenset(item.value for item in FatalIssue)
_OUTCOMES = frozenset({"llm", "fallback"})
_REVIEW_ID_RE = re.compile(r"^rvw-[0-9a-f]{16}$")


def _require_text(value: Any, field: str, *, max_length: int | None = None) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty trimmed string")
    if max_length is not None and len(value) > max_length:
        raise ValueError(f"{field} is too long")
    return value


def _check_blind_keys(value: Any) -> None:
    """Reject fields that could reveal the condition or computed result."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in _BLIND_FORBIDDEN_KEYS:
                raise ValueError(f"prompt_input contains forbidden blind key: {key}")
            _check_blind_keys(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _check_blind_keys(child)


def _copy_prompt_input(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("prompt_input must be an object")
    _check_blind_keys(value)
    return copy.deepcopy(dict(value))


@dataclass(frozen=True)
class BlindReviewItem:
    """The condition-blinded input shown to a human reviewer.

    ``review_id`` is an anonymous stable token.  No condition name or metric
    is part of this object.  ``outcome`` remains because a reviewer may need
    to distinguish a successful LLM turn from a rule fallback; fallback turns
    are excluded from scoring by :func:`aggregate_review_records`.
    """

    review_id: str
    turn_data: Mapping[str, Any]
    prompt_input: Mapping[str, Any]
    outcome: str
    retry_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.review_id, str) or not _REVIEW_ID_RE.fullmatch(self.review_id):
            raise ValueError("review_id must be a stable rvw-<16 hex> token")
        if not isinstance(self.turn_data, Mapping):
            raise ValueError("turn_data must be an object")
        required = {"turn", "actor", "target", "topic", "approach", "response"}
        if set(self.turn_data) != required:
            raise ValueError("turn_data must contain exactly the event fields")
        if isinstance(self.turn_data["turn"], bool) or not isinstance(self.turn_data["turn"], int):
            raise ValueError("turn_data.turn must be an integer")
        if self.turn_data["turn"] < 1:
            raise ValueError("turn_data.turn must be positive")
        for field in ("actor", "target", "topic", "approach", "response"):
            _require_text(self.turn_data[field], f"turn_data.{field}", max_length=256)
        if self.turn_data["actor"] == self.turn_data["target"]:
            raise ValueError("turn_data actor and target must differ")
        _copy_prompt_input(self.prompt_input)
        if self.outcome not in _OUTCOMES:
            raise ValueError("outcome must be llm or fallback")
        if isinstance(self.retry_count, bool) or not isinstance(self.retry_count, int):
            raise ValueError("retry_count must be an integer")
        if self.retry_count < 0:
            raise ValueError("retry_count cannot be negative")

    @property
    def eligible(self) -> bool:
        return self.outcome == "llm"

    def to_dict(self) -> dict[str, Any]:
        """Return a safe copy suitable for blind-review UI or JSON."""

        return {
            "review_id": self.review_id,
            "turn_data": copy.deepcopy(dict(self.turn_data)),
            "prompt_input": copy.deepcopy(dict(self.prompt_input)),
            "outcome": self.outcome,
            "retry_count": self.retry_count,
        }


def _normalise_fatal(value: Any) -> str:
    if value is None:
        return FatalIssue.NONE.value
    if isinstance(value, FatalIssue):
        value = value.value
    if not isinstance(value, str) or value not in _FATAL_VALUES:
        raise ValueError(f"fatal_issue must be one of {sorted(_FATAL_VALUES)}")
    return value


def score_label(total_score: int, fatal_issue: str | FatalIssue = FatalIssue.NONE) -> str:
    """Map a validated score to the four M5 labels; fatal always wins."""

    fatal = _normalise_fatal(fatal_issue)
    if isinstance(total_score, bool) or not isinstance(total_score, int) or not 0 <= total_score <= 8:
        raise ValueError("total_score must be an integer from 0 through 8")
    if fatal != FatalIssue.NONE.value or total_score <= 3:
        return "不採用"
    if total_score <= 5:
        return "修正候補"
    if total_score == 6:
        return "利用可能"
    return "良好"


@dataclass(frozen=True)
class ReviewRecord:
    """Strict manual review record: four 0/1/2 scores and one short reason."""

    review_id: str
    scores: Mapping[str, int]
    rationale: str
    fatal_issue: str = FatalIssue.NONE.value

    def __post_init__(self) -> None:
        if not isinstance(self.review_id, str) or not _REVIEW_ID_RE.fullmatch(self.review_id):
            raise ValueError("review_id must be a stable rvw-<16 hex> token")
        if not isinstance(self.scores, Mapping) or set(self.scores) != set(REVIEW_DIMENSIONS):
            raise ValueError("scores must contain exactly the four review dimensions")
        for dimension in REVIEW_DIMENSIONS:
            score = self.scores[dimension]
            if isinstance(score, bool) or not isinstance(score, int) or score not in (0, 1, 2):
                raise ValueError(f"score for {dimension} must be 0, 1, or 2")
        _require_text(self.rationale, "rationale", max_length=240)
        object.__setattr__(self, "fatal_issue", _normalise_fatal(self.fatal_issue))
        object.__setattr__(self, "scores", dict(self.scores))

    @property
    def total_score(self) -> int:
        return sum(self.scores.values())

    @property
    def label(self) -> str:
        return score_label(self.total_score, self.fatal_issue)

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "scores": dict(self.scores),
            "rationale": self.rationale,
            "fatal_issue": self.fatal_issue,
            "total_score": self.total_score,
            "label": self.label,
        }


def validate_review_record(value: ReviewRecord | Mapping[str, Any]) -> ReviewRecord:
    """Validate a record supplied by a person or future evaluator."""

    if isinstance(value, ReviewRecord):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("review record must be a ReviewRecord or object")
    if set(value) != {"review_id", "scores", "rationale", "fatal_issue"}:
        raise ValueError("review record contains missing or unknown fields")
    return ReviewRecord(
        review_id=value["review_id"],
        scores=value["scores"],
        rationale=value["rationale"],
        fatal_issue=value["fatal_issue"],
    )


def _stable_review_id(root_seed: Any, source: str, turn: int, actor: str) -> str:
    token = f"m5.review.v1\0{root_seed!r}\0{source}\0{turn}\0{actor}".encode("utf-8")
    return "rvw-" + hashlib.sha256(token).hexdigest()[:16]


def _event_for_turn(run: Mapping[str, Any], turn: int) -> dict[str, Any]:
    turns = run.get("turns")
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        raise ValueError("run turns must be an array")
    matches = [item for item in turns if isinstance(item, Mapping) and item.get("turn") == turn]
    if len(matches) != 1:
        raise ValueError("each audited turn must have exactly one matching event")
    event = matches[0]
    required = ("turn", "actor", "target", "topic", "approach", "response")
    if not set(required).issubset(event):
        raise ValueError("turn event is missing review fields")
    return {field: copy.deepcopy(event[field]) for field in required}


def _items_for_audit(
    audit: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    root_seed: Any,
    source: str,
) -> list[BlindReviewItem]:
    turns = audit.get("turns")
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        raise ValueError("participant audit turns must be an array")
    items: list[BlindReviewItem] = []
    seen: set[int] = set()
    for audited in turns:
        if not isinstance(audited, Mapping):
            raise ValueError("participant audit turn must be an object")
        turn = audited.get("turn")
        if isinstance(turn, bool) or not isinstance(turn, int) or turn in seen:
            raise ValueError("participant audit turns must have unique integer turn numbers")
        seen.add(turn)
        prompt_input = _copy_prompt_input(audited.get("prompt_input"))
        outcome = audited.get("outcome")
        if outcome not in _OUTCOMES:
            raise ValueError("participant audit outcome must be llm or fallback")
        attempts = audited.get("attempts")
        if not isinstance(attempts, Sequence) or isinstance(attempts, (str, bytes)) or not attempts:
            raise ValueError("participant audit attempts must be a non-empty array")
        retry_count = len(attempts) - 1
        if outcome == "llm":
            structured = audited.get("structured_output")
            if not isinstance(structured, Mapping):
                raise ValueError("successful LLM turn must include structured_output")
            event = _event_for_turn(run, turn)
            turn_data = {
                "turn": turn,
                # Actor identity is taken from the recorded event, never from
                # the model's output.  The model is only allowed to choose
                # target/topic/approach/response.
                "actor": event["actor"],
                "target": structured.get("target_id"),
                "topic": structured.get("topic"),
                "approach": structured.get("approach"),
                "response": structured.get("response"),
            }
        else:
            turn_data = _event_for_turn(run, turn)
        # Let BlindReviewItem enforce the exact event shape/types.
        item = BlindReviewItem(
            review_id=_stable_review_id(root_seed, source, turn, str(turn_data["actor"])),
            turn_data=turn_data,
            prompt_input=prompt_input,
            outcome=outcome,
            retry_count=retry_count,
        )
        items.append(item)
    return items


def extract_blind_review_items(run_result: Mapping[str, Any]) -> tuple[BlindReviewItem, ...]:
    """Extract anonymous A/B turn inputs without condition names or metrics.

    The function is pure: it makes defensive copies and never mutates the
    supplied RunResult.  Fallback items are retained for operational counts,
    but are not eligible for score aggregation.
    """

    if not isinstance(run_result, Mapping):
        raise ValueError("run_result must be an object")
    root_seed = run_result.get("root_seed")
    if root_seed is None:
        raise ValueError("run_result.root_seed is required")
    runs = run_result.get("runs")
    if isinstance(runs, Mapping):
        sources = [(str(key), value) for key, value in sorted(runs.items(), key=lambda pair: str(pair[0]))]
        audit_map = run_result.get("participant_audit")
        if not isinstance(audit_map, Mapping):
            raise ValueError("A/B run must include participant_audit")
        pairs = [(source, run, audit_map.get(source)) for source, run in sources]
    else:
        run = run_result
        pairs = [(str(run.get("condition_id", "single")), run, run_result.get("participant_audit"))]
    items: list[BlindReviewItem] = []
    for source, run, audit in pairs:
        if not isinstance(run, Mapping) or not isinstance(audit, Mapping):
            raise ValueError("each run must include a participant audit object")
        items.extend(_items_for_audit(audit, run, root_seed=root_seed, source=source))
    ids = [item.review_id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("review_id collision; source run is not uniquely identifiable")
    return tuple(items)


def aggregate_review_records(
    items: Sequence[BlindReviewItem | Mapping[str, Any]],
    records: Sequence[ReviewRecord | Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate validated labels without generating any subjective score.

    Fallback items are counted operationally but excluded from the review
    denominator and from all score distributions.  Missing eligible labels
    are reported as ``unreviewed_count`` rather than silently treated as zero.
    """

    normal_items_list: list[BlindReviewItem] = []
    for item in items:
        if isinstance(item, BlindReviewItem):
            normal_items_list.append(item)
            continue
        if not isinstance(item, Mapping) or set(item) != {
            "review_id", "turn_data", "prompt_input", "outcome", "retry_count"
        }:
            raise ValueError("review item contains missing or unknown fields")
        normal_items_list.append(BlindReviewItem(**item))
    normal_items = tuple(normal_items_list)
    item_by_id = {item.review_id: item for item in normal_items}
    if len(item_by_id) != len(normal_items):
        raise ValueError("items contain duplicate review_id")
    normal_records = tuple(validate_review_record(record) for record in records)
    record_by_id: dict[str, ReviewRecord] = {}
    for record in normal_records:
        if record.review_id in record_by_id:
            raise ValueError("records contain duplicate review_id")
        if record.review_id not in item_by_id:
            raise ValueError("record refers to an unknown review_id")
        # A fallback is a rule output, not an LLM conversation to score.
        if not item_by_id[record.review_id].eligible:
            continue
        record_by_id[record.review_id] = record

    eligible = tuple(item for item in normal_items if item.eligible)
    reviewed = tuple(record_by_id.values())
    distribution = {
        dimension: {"0": 0, "1": 0, "2": 0}
        for dimension in REVIEW_DIMENSIONS
    }
    for record in reviewed:
        for dimension in REVIEW_DIMENSIONS:
            distribution[dimension][str(record.scores[dimension])] += 1
    totals = [record.total_score for record in reviewed]
    fatal_types: dict[str, int] = {}
    for record in reviewed:
        if record.fatal_issue != FatalIssue.NONE.value:
            fatal_types[record.fatal_issue] = fatal_types.get(record.fatal_issue, 0) + 1
    acceptable = sum(
        record.total_score >= 6 and record.fatal_issue == FatalIssue.NONE.value
        for record in reviewed
    )
    review_rows = [
        {
            "review_id": record.review_id,
            "total_score": record.total_score,
            "label": record.label,
            "fatal_issue": record.fatal_issue,
        }
        for record in sorted(reviewed, key=lambda item: item.review_id)
    ]
    return {
        "target_count": len(normal_items),
        "eligible_count": len(eligible),
        "reviewed_count": len(reviewed),
        "unreviewed_count": len(eligible) - len(reviewed),
        "llm_success": sum(item.outcome == "llm" for item in normal_items),
        "retry_success": sum(item.outcome == "llm" and item.retry_count > 0 for item in normal_items),
        "fallback": sum(item.outcome == "fallback" for item in normal_items),
        "score_distribution": distribution,
        "total_median": median(totals) if totals else None,
        "acceptable_rate": (acceptable / len(reviewed)) if reviewed else None,
        "fatal_count": sum(fatal_types.values()),
        "fatal_types": fatal_types,
        "acceptable_count": acceptable,
        "reviews": review_rows,
    }


__all__ = [
    "BlindReviewItem",
    "FatalIssue",
    "REVIEW_DIMENSIONS",
    "ReviewRecord",
    "aggregate_review_records",
    "extract_blind_review_items",
    "score_label",
    "validate_review_record",
]
