"""Pure directional subjective-affect state for one simulation run."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from .domain import ResponseKind


_MIN = -2
_MAX = 2


def _participant_ids(participants: Sequence[str]) -> tuple[str, ...]:
    if isinstance(participants, (str, bytes)):
        raise ValueError("participants must be a sequence of IDs")
    ids = tuple(participants)
    if len(ids) < 2:
        raise ValueError("at least two participants are required")
    if any(type(item) is not str or not item or item != item.strip() for item in ids):
        raise ValueError("participant IDs must be non-empty trimmed strings")
    if len(set(ids)) != len(ids):
        raise ValueError("participant IDs must be unique")
    return ids


def _direction(owner: str, other: str) -> tuple[str, str]:
    if type(owner) is not str or type(other) is not str or not owner or not other:
        raise ValueError("direction IDs must be non-empty strings")
    if owner == other:
        raise ValueError("owner and other must be different participants")
    return owner, other


def _checked_value(value: int, label: str = "affect value") -> int:
    if type(value) is not int or not _MIN <= value <= _MAX:
        raise ValueError(f"{label} must be an integer from -2 to +2")
    return value


def _reaction_delta(reaction: ResponseKind | str) -> tuple[ResponseKind, int]:
    if isinstance(reaction, ResponseKind):
        kind = reaction
    elif type(reaction) is str:
        labels = {
            "positive": ResponseKind.POSITIVE,
            "neutral": ResponseKind.NEUTRAL,
            "misaligned": ResponseKind.MISALIGNED,
        }
        kind = labels.get(reaction.strip().lower())
        if kind is None:
            try:
                kind = ResponseKind(reaction)
            except ValueError as exc:
                raise ValueError("reaction must be positive, neutral, or misaligned") from exc
    else:
        raise ValueError("reaction must be positive, neutral, or misaligned")
    delta = {ResponseKind.POSITIVE: 1, ResponseKind.NEUTRAL: 0, ResponseKind.MISALIGNED: -1}[kind]
    return kind, delta


def _clamp(value: int) -> int:
    return max(_MIN, min(_MAX, value))


@dataclass(frozen=True)
class DirectionalAffectState:
    """Immutable values for every explicit ``owner -> other`` direction."""

    participants: tuple[str, ...]
    values: Mapping[tuple[str, str], int] | None = None

    def __post_init__(self) -> None:
        ids = _participant_ids(self.participants)
        # Preserve participant order in the mapping so serialized run output
        # is stable across Python processes (sets have hash-randomized order).
        directions = tuple(
            (owner, other) for owner in ids for other in ids if owner != other
        )
        supplied = (
            {direction: 0 for direction in directions}
            if self.values is None
            else dict(self.values)
        )
        for direction, value in supplied.items():
            if not isinstance(direction, tuple) or len(direction) != 2:
                raise ValueError("affect direction must be an (owner, other) pair")
            owner, other = direction
            if owner not in ids or other not in ids:
                raise ValueError(f"unknown affect direction: {direction!r}")
            if owner == other:
                raise ValueError("affect direction cannot point to itself")
            _checked_value(value, f"affect value for {direction!r}")
        if set(supplied) != set(directions):
            raise ValueError("affect state must contain every non-self direction")
        object.__setattr__(self, "participants", ids)
        object.__setattr__(self, "values", MappingProxyType(dict(supplied)))

    @classmethod
    def initial(cls, participants: Sequence[str]) -> "DirectionalAffectState":
        return cls(tuple(participants))

    def value(self, owner: str, other: str) -> int:
        owner, other = _direction(owner, other)
        if owner not in self.participants or other not in self.participants:
            raise ValueError("unknown participant ID")
        try:
            return self.values[(owner, other)]  # type: ignore[index]
        except KeyError as exc:
            raise ValueError("unknown affect direction") from exc

    def _with_value(self, owner: str, other: str, value: int) -> "DirectionalAffectState":
        self.value(owner, other)
        values = dict(self.values)
        values[(owner, other)] = _checked_value(value)
        return type(self)(self.participants, values)


@dataclass(frozen=True)
class AffectTransition:
    """Audit record for one update of ``receiver -> initiator``."""

    turn: int
    receiver: str
    initiator: str
    reaction: ResponseKind
    delta: int
    before: int
    after: int

    def __post_init__(self) -> None:
        if type(self.turn) is not int or self.turn < 1:
            raise ValueError("turn must be a positive integer")
        _direction(self.receiver, self.initiator)
        if not isinstance(self.reaction, ResponseKind):
            raise ValueError("reaction must be a ResponseKind")
        expected = {ResponseKind.POSITIVE: 1, ResponseKind.NEUTRAL: 0, ResponseKind.MISALIGNED: -1}[self.reaction]
        if self.delta != expected:
            raise ValueError("transition delta does not match reaction")
        _checked_value(self.before, "transition before")
        _checked_value(self.after, "transition after")
        if self.after != _clamp(self.before + self.delta):
            raise ValueError("transition after does not match before and delta")


class AffectUpdater:
    """Stateless reaction-to-delta updater."""

    def update(
        self,
        state: DirectionalAffectState,
        turn: int,
        receiver: str,
        initiator: str,
        reaction: ResponseKind | str,
    ) -> tuple[DirectionalAffectState, AffectTransition]:
        if not isinstance(state, DirectionalAffectState):
            raise TypeError("state must be a DirectionalAffectState")
        if type(turn) is not int or turn < 1:
            raise ValueError("turn must be a positive integer")
        kind, delta = _reaction_delta(reaction)
        before = state.value(receiver, initiator)
        after = _clamp(before + delta)
        next_state = state._with_value(receiver, initiator, after)
        transition = AffectTransition(turn, receiver, initiator, kind, delta, before, after)
        return next_state, transition


__all__ = ["AffectTransition", "AffectUpdater", "DirectionalAffectState"]
