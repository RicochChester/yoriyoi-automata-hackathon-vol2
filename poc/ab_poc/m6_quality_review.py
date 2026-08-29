"""M6 quality-review helpers.

M6 intentionally sits beside :mod:`quality_review` rather than changing its
M5 contract.  It reuses the M5 blind extractor and adds only the redaction and
labels needed for the multi-seed quality check.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from statistics import median
from typing import Any

from .quality_review import (
    BlindReviewItem,
    FatalIssue,
    REVIEW_DIMENSIONS,
    extract_blind_review_items,
)


class M6FatalIssue(str, Enum):
    """Fatal labels accepted by the M6 reviewer."""

    NONE = FatalIssue.NONE.value
    UNSUPPORTED_MEMORY = FatalIssue.UNSUPPORTED_MEMORY.value
    UNSUPPORTED_ATTRIBUTE_OR_RELATION = FatalIssue.UNSUPPORTED_ATTRIBUTE_OR_RELATION.value
    WRONG_ADDRESSEE = FatalIssue.WRONG_ADDRESSEE.value
    PROMPT_CONTRADICTION = FatalIssue.PROMPT_CONTRADICTION.value
    META_LEAK = FatalIssue.META_LEAK.value
    IRRELEVANT_EXTERNAL_CONTENT = "irrelevant_external_content"


M6_FATAL_ISSUES = frozenset(item.value for item in M6FatalIssue)
M6_FLUENCY_VALUES = frozenset({0, 1, 2})
M6_CONTENT_DRIFT_VALUES = frozenset({
    "none",
    "minor_topic_blur",
    "irrelevant_external_content",
})
M6_REPETITION_VALUES = frozenset({
    "none",
    "minor_similar_expression",
    "major_near_duplicate",
})


_M6_DISPLAY_METADATA_KEYS = frozenset(
    {
        "seed",
        "root_seed",
        "provider",
        "provider_id",
        "provider_name",
        "model",
        "model_id",
        "model_name",
        "ab",
        "a_b",
        "variant",
    }
)


def _redact_direct_known(value: Any) -> Any:
    """Return a deep copy without direct knowledge or display metadata."""

    if isinstance(value, Mapping):
        return {
            key: _redact_direct_known(child)
            for key, child in value.items()
            if key != "direct_known" and key not in _M6_DISPLAY_METADATA_KEYS
        }
    if isinstance(value, list):
        return [_redact_direct_known(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_redact_direct_known(child) for child in value)
    return copy.deepcopy(value)


def build_m6_blind_items(run_result: Mapping[str, Any]) -> tuple[BlindReviewItem, ...]:
    """Build M6 blind items without mutating the run or its audit.

    The M5 extractor remains the authority for source validation and for its
    existing forbidden blind keys.  M6 then removes ``direct_known`` at every
    nesting level because it is the specific signal under review in this
    milestone.  The review id, event, outcome and retry count are preserved.
    """

    source_items = extract_blind_review_items(run_result)
    result: list[BlindReviewItem] = []
    for item in source_items:
        item_data = item.to_dict()
        item_data["prompt_input"] = _redact_direct_known(item_data["prompt_input"])
        result.append(BlindReviewItem(**item_data))
    return tuple(result)


# A discoverable alias for callers that use the M5 naming convention.
extract_m6_blind_review_items = build_m6_blind_items


def _require_text(value: Any, field: str, *, max_length: int | None = None) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty trimmed string")
    if max_length is not None and len(value) > max_length:
        raise ValueError(f"{field} is too long")
    return value


def _normalise_fatal(value: Any) -> str:
    if isinstance(value, M6FatalIssue):
        value = value.value
    elif isinstance(value, FatalIssue):
        value = value.value
    if not isinstance(value, str) or value not in M6_FATAL_ISSUES:
        raise ValueError(f"fatal_issue must be one of {sorted(M6_FATAL_ISSUES)}")
    return value


@dataclass(frozen=True)
class M6ReviewRecord:
    """Strict M6 label: M5's four scores plus fluency/drift/repetition."""

    review_id: str
    scores: Mapping[str, int]
    rationale: str
    fatal_issue: str = M6FatalIssue.NONE.value
    japanese_fluency: int = 2
    content_drift: str = "none"
    repetition: str = "none"

    def __post_init__(self) -> None:
        # Reuse the stable id shape without importing M5's private validator.
        if not isinstance(self.review_id, str) or not self.review_id.startswith("rvw-"):
            raise ValueError("review_id must be a stable rvw-<16 hex> token")
        token = self.review_id[4:]
        if len(token) != 16 or any(char not in "0123456789abcdef" for char in token):
            raise ValueError("review_id must be a stable rvw-<16 hex> token")
        if not isinstance(self.scores, Mapping) or set(self.scores) != set(REVIEW_DIMENSIONS):
            raise ValueError("scores must contain exactly the four review dimensions")
        for dimension in REVIEW_DIMENSIONS:
            score = self.scores[dimension]
            if isinstance(score, bool) or not isinstance(score, int) or score not in (0, 1, 2):
                raise ValueError(f"score for {dimension} must be 0, 1, or 2")
        _require_text(self.rationale, "rationale", max_length=240)
        fatal = _normalise_fatal(self.fatal_issue)
        if (
            isinstance(self.japanese_fluency, bool)
            or not isinstance(self.japanese_fluency, int)
            or self.japanese_fluency not in M6_FLUENCY_VALUES
        ):
            raise ValueError("japanese_fluency must be 0, 1, or 2")
        if not isinstance(self.content_drift, str) or self.content_drift not in M6_CONTENT_DRIFT_VALUES:
            raise ValueError(f"content_drift must be one of {sorted(M6_CONTENT_DRIFT_VALUES)}")
        if not isinstance(self.repetition, str) or self.repetition not in M6_REPETITION_VALUES:
            raise ValueError(f"repetition must be one of {sorted(M6_REPETITION_VALUES)}")
        external = M6FatalIssue.IRRELEVANT_EXTERNAL_CONTENT.value
        if (fatal == external) != (self.content_drift == external):
            raise ValueError(
                "irrelevant_external_content fatal_issue and content_drift must agree"
            )
        object.__setattr__(self, "fatal_issue", fatal)
        object.__setattr__(self, "scores", dict(self.scores))

    @property
    def total_score(self) -> int:
        return sum(self.scores.values())

    @property
    def acceptable(self) -> bool:
        return self.total_score >= 6 and self.fatal_issue == M6FatalIssue.NONE.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "scores": dict(self.scores),
            "rationale": self.rationale,
            "fatal_issue": self.fatal_issue,
            "japanese_fluency": self.japanese_fluency,
            "content_drift": self.content_drift,
            "repetition": self.repetition,
            "total_score": self.total_score,
            "acceptable": self.acceptable,
        }


def validate_m6_review_record(value: M6ReviewRecord | Mapping[str, Any]) -> M6ReviewRecord:
    """Validate a complete M6 record, rejecting missing or unknown fields."""

    if isinstance(value, M6ReviewRecord):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("M6 review record must be an M6ReviewRecord or object")
    required = {
        "review_id",
        "scores",
        "rationale",
        "fatal_issue",
        "japanese_fluency",
        "content_drift",
        "repetition",
    }
    derived = {"total_score", "acceptable"}
    if set(value) not in (required, required | derived):
        raise ValueError("M6 review record contains missing or unknown fields")
    record = M6ReviewRecord(**{key: value[key] for key in required})
    if "total_score" in value and value["total_score"] != record.total_score:
        raise ValueError("total_score does not match scores")
    if "acceptable" in value and value["acceptable"] != record.acceptable:
        raise ValueError("acceptable does not match scores and fatal_issue")
    return record


def _normalise_item(value: BlindReviewItem | Mapping[str, Any]) -> BlindReviewItem:
    if isinstance(value, BlindReviewItem):
        return value
    required = {"review_id", "turn_data", "prompt_input", "outcome", "retry_count"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("review item contains missing or unknown fields")
    return BlindReviewItem(**dict(value))


def aggregate_m6_review_records(
    items: Sequence[BlindReviewItem | Mapping[str, Any]],
    records: Sequence[M6ReviewRecord | Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate M6 labels; fallback turns are operational but unscored."""

    normal_items = tuple(_normalise_item(item) for item in items)
    item_by_id: dict[str, BlindReviewItem] = {}
    for item in normal_items:
        if item.review_id in item_by_id:
            raise ValueError("items contain duplicate review_id")
        item_by_id[item.review_id] = item

    normal_records = tuple(validate_m6_review_record(record) for record in records)
    record_by_id: dict[str, M6ReviewRecord] = {}
    for record in normal_records:
        if record.review_id in record_by_id:
            raise ValueError("records contain duplicate review_id")
        if record.review_id not in item_by_id:
            raise ValueError("record refers to an unknown review_id")
        if item_by_id[record.review_id].eligible:
            record_by_id[record.review_id] = record

    eligible = tuple(item for item in normal_items if item.eligible)
    reviewed = tuple(record_by_id.values())
    score_distribution = {
        dimension: {"0": 0, "1": 0, "2": 0} for dimension in REVIEW_DIMENSIONS
    }
    fluency_distribution = {"0": 0, "1": 0, "2": 0}
    drift_distribution = {value: 0 for value in sorted(M6_CONTENT_DRIFT_VALUES)}
    repetition_distribution = {value: 0 for value in sorted(M6_REPETITION_VALUES)}
    for record in reviewed:
        for dimension in REVIEW_DIMENSIONS:
            score_distribution[dimension][str(record.scores[dimension])] += 1
        fluency_distribution[str(record.japanese_fluency)] += 1
        drift_distribution[record.content_drift] += 1
        repetition_distribution[record.repetition] += 1

    totals = [record.total_score for record in reviewed]
    fluencies = [record.japanese_fluency for record in reviewed]
    fatal_types: dict[str, int] = {}
    for record in reviewed:
        if record.fatal_issue != M6FatalIssue.NONE.value:
            fatal_types[record.fatal_issue] = fatal_types.get(record.fatal_issue, 0) + 1
    acceptable = sum(record.acceptable for record in reviewed)
    content_drift_count = sum(record.content_drift != "none" for record in reviewed)
    repetition_count = sum(record.repetition != "none" for record in reviewed)
    reviews = [
        {
            "review_id": record.review_id,
            "total_score": record.total_score,
            "acceptable": record.acceptable,
            "fatal_issue": record.fatal_issue,
            "japanese_fluency": record.japanese_fluency,
            "content_drift": record.content_drift,
            "repetition": record.repetition,
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
        "score_distribution": score_distribution,
        "total_median": median(totals) if totals else None,
        "acceptable_rate": acceptable / len(reviewed) if reviewed else None,
        "fatal_count": sum(fatal_types.values()),
        "fatal_types": fatal_types,
        "acceptable_count": acceptable,
        "japanese_fluency_distribution": fluency_distribution,
        "japanese_fluency_median": median(fluencies) if fluencies else None,
        "japanese_fluency_zero_count": fluency_distribution["0"],
        "content_drift_distribution": drift_distribution,
        "content_drift_count": content_drift_count,
        "repetition_distribution": repetition_distribution,
        "repetition_count": repetition_count,
        "reviews": reviews,
    }


__all__ = [
    "M6FatalIssue",
    "M6ReviewRecord",
    "M6_FATAL_ISSUES",
    "aggregate_m6_review_records",
    "build_m6_blind_items",
    "extract_m6_blind_review_items",
    "validate_m6_review_record",
]
