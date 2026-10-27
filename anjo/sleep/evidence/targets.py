"""Targets, results and expectations as an event log and its fold (Task 3, the ``TargetStore`` port).

Every mutation appends one event to the person's log, and the table is ``fold`` of that log, so a
replay from scratch (``replay_digest``) must equal the live state (``digest``). Each event
carries its sequence number, and a fold refuses a gap or a repeat.

- Only the person's own evidence raises a target's ``revision``: ``record_mention`` takes an
  item extracted from their turns, or imported from their own corrections and preferences,
  counts each item once and moves the revision by exactly one. Her homework and practice never
  do, and neither does a result. Every target starts at revision 1, unacted and not noted.
- An ``upsert`` without evidence changes only a target's label, status or date: it is refused
  if it would move the revision or the confidence. ``acted_revision`` moves only through
  ``mark_acted`` and ``due_noted`` only through ``settle``; an upsert keeps both.
- A homework result (``research``, with a homework id) with at least one verified source raises
  a thread's ``confidence`` by ``CONFIDENCE_STEP``, recorded in the event so a replay never
  depends on today's policy. An ``evidence`` result never does.
- Times are the night's clock, virtual in accelerated runs: ``open_targets`` reads only the
  ``as_of`` it is given, against the day each target was first mentioned.
- The person's digest covers her MIND workspace too when the store is given its log
  (``workspace``): ``digest`` and ``replay_digest`` then hash the targets' digest with the
  workspace's records digest (Task 17), so a killed-and-resumed night and an uninterrupted one
  agree on both.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Protocol, get_args

from anjo.sleep.evidence.jsonl import Codec, Guard, append_records, canonical, read_records
from anjo.sleep.types import (
    EvidenceKind,
    Evidence,
    Expectation,
    ExpectationOutcome,
    Result,
    ResultBasis,
    Target,
    TargetKind,
    TargetStatus,
)

CONFIDENCE_STEP: Final = 0.1
# Evidence kinds that come from the person: from their turns or their own stored corrections and
# preferences. ``homework`` and ``practice`` are hers.
PERSON_KINDS: Final[frozenset[EvidenceKind]] = frozenset(
    {"lesson", "correction", "preference", "tone_request"}
)
_TARGET_KINDS = frozenset(get_args(TargetKind))
_STATUSES = frozenset(get_args(TargetStatus))
_BASES = frozenset(get_args(ResultBasis))
_OUTCOMES = frozenset(get_args(ExpectationOutcome))


class TargetStoreError(ValueError):
    """A mutation the log refuses, or a stored event that can't be folded."""


def target_id_for(person_id: str, kind: TargetKind, label: str) -> str:
    """A stable id for a new target: the same person, kind and label always get the same one."""
    normal = " ".join(label.casefold().split())
    digest = hashlib.sha256(f"{person_id}\x1f{kind}\x1f{normal}".encode()).hexdigest()
    return f"t-{digest[:16]}"


@dataclass(frozen=True)
class Mention:
    """One item's mention of a target, already matched to the target's id."""

    target_id: str
    kind: TargetKind
    label: str
    check_after: datetime | None
    confidence: float | None

    def __post_init__(self) -> None:
        if self.kind not in _TARGET_KINDS:
            raise TargetStoreError(f"unknown target kind {self.kind!r}")
        if not self.label.strip():
            raise TargetStoreError("a mention needs a label")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise TargetStoreError(f"confidence must be in [0, 1], got {self.confidence}")


def mentioned_target(existing: Target | None, mention: Mention, *, person_id: str, day: date) -> Target:
    """The target after one new mention: created at revision 1, or one revision up. A mention
    never changes a target's kind, label, status or confidence; a new date moves a commitment."""
    if existing is None:
        return Target(
            id=mention.target_id, person_id=person_id, kind=mention.kind,
            label=mention.label.strip(), check_after=mention.check_after, status="open",
            revision=1, acted_revision=0,
            confidence=mention.confidence if mention.kind == "thread" else None,
            last_mentioned=day,
        )
    if existing.kind != mention.kind:
        raise TargetStoreError(f"target {existing.id} is a {existing.kind}, not a {mention.kind}")
    check_after = existing.check_after if mention.check_after is None else mention.check_after
    return replace(
        existing, revision=existing.revision + 1, last_mentioned=max(existing.last_mentioned, day),
        check_after=check_after,
    )


# ── Records ───────────────────────────────────────────────────────────────────


def _iso(value: date | datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _when(value: object) -> datetime | None:
    return None if value is None else datetime.fromisoformat(_str(value))


def _str(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("expected a string")
    return value


def _int(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("expected an integer")
    return value


def _bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError("expected a boolean")
    return value


def _choice(value: object, allowed: frozenset[str]) -> Any:
    if value not in allowed:
        raise ValueError(f"unexpected value {value!r}")
    return value


def target_record(target: Target) -> dict[str, Any]:
    return {
        "id": target.id, "person_id": target.person_id, "kind": target.kind, "label": target.label,
        "check_after": _iso(target.check_after), "status": target.status,
        "revision": target.revision, "acted_revision": target.acted_revision,
        "confidence": target.confidence, "last_mentioned": target.last_mentioned.isoformat(),
        "due_noted": target.due_noted,
    }


def target_from_record(record: Mapping[str, Any]) -> Target:
    confidence = record["confidence"]
    if confidence is not None and (not isinstance(confidence, (int, float)) or isinstance(confidence, bool)):
        raise TypeError("confidence must be a number")
    return Target(
        id=_str(record["id"]), person_id=_str(record["person_id"]),
        kind=_choice(record["kind"], _TARGET_KINDS), label=_str(record["label"]),
        check_after=_when(record["check_after"]), status=_choice(record["status"], _STATUSES),
        revision=_int(record["revision"]), acted_revision=_int(record["acted_revision"]),
        confidence=None if confidence is None else float(confidence),
        last_mentioned=date.fromisoformat(_str(record["last_mentioned"])),
        due_noted=_bool(record["due_noted"]),
    )


def result_record(result: Result) -> dict[str, Any]:
    return {
        "id": result.id, "target_id": result.target_id, "homework_id": result.homework_id,
        "verified_sources": result.verified_sources, "created_at": result.created_at.isoformat(),
        "consumed": result.consumed, "basis": result.basis, "user_cites": result.user_cites,
    }


def result_from_record(record: Mapping[str, Any]) -> Result:
    homework_id = record["homework_id"]
    result = Result(
        id=_str(record["id"]), target_id=_str(record["target_id"]),
        homework_id=None if homework_id is None else _str(homework_id),
        verified_sources=_int(record["verified_sources"]),
        created_at=datetime.fromisoformat(_str(record["created_at"])),
        consumed=_bool(record["consumed"]), basis=_choice(record["basis"], _BASES),
        user_cites=_int(record["user_cites"]),
    )
    for name, count in (("verified_sources", result.verified_sources), ("user_cites", result.user_cites)):
        if count < 0:
            raise TargetStoreError(f"{name} must be >= 0, got {count}")
    return result


def expectation_record(expectation: Expectation) -> dict[str, Any]:
    return {
        "id": expectation.id, "target_id": expectation.target_id,
        "made_at": expectation.made_at.isoformat(), "horizon": expectation.horizon.isoformat(),
        "p_needed": expectation.p_needed, "outcome": expectation.outcome,
    }


def expectation_from_record(record: Mapping[str, Any]) -> Expectation:
    outcome = record["outcome"]
    p_needed = record["p_needed"]
    if not isinstance(p_needed, (int, float)) or isinstance(p_needed, bool):
        raise TypeError("p_needed must be a number")
    return Expectation(
        id=_str(record["id"]), target_id=_str(record["target_id"]),
        made_at=datetime.fromisoformat(_str(record["made_at"])),
        horizon=datetime.fromisoformat(_str(record["horizon"])), p_needed=float(p_needed),
        outcome=None if outcome is None else _choice(outcome, _OUTCOMES),
    )


# ── The fold ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TargetState:
    """The table the log folds to. ``mentioned`` holds the evidence ids already counted."""

    seq: int
    targets: Mapping[str, Target]
    first_seen: Mapping[str, date]
    mentioned: frozenset[str]
    results: tuple[Result, ...]
    expectations: tuple[Expectation, ...]


EMPTY = TargetState(
    seq=0, targets=MappingProxyType({}), first_seen=MappingProxyType({}), mentioned=frozenset(),
    results=(), expectations=(),
)


def _known(state: TargetState, target_id: object) -> Target:
    target = state.targets.get(_str(target_id))
    if target is None:
        raise TargetStoreError(f"unknown target {target_id!r}")
    return target


def _with_target(state: TargetState, target: Target) -> TargetState:
    return replace(state, targets=MappingProxyType({**state.targets, target.id: target}))


def _check_upsert(existing: Target | None, incoming: Target, *, mention: bool) -> None:
    """What an upsert may do: start a target at revision 1, or, for an existing one, move the
    revision by exactly one on a mention and not at all otherwise, never moving confidence."""
    if existing is None:
        if (incoming.revision, incoming.acted_revision, incoming.due_noted) != (1, 0, False):
            raise TargetStoreError(
                f"a new target starts at revision 1, unacted and not noted; got revision "
                f"{incoming.revision}, acted {incoming.acted_revision}, due_noted {incoming.due_noted}"
            )
        return
    if incoming.kind != existing.kind:
        raise TargetStoreError(f"target {incoming.id} can't change its kind")
    expected = existing.revision + 1 if mention else existing.revision
    if incoming.revision != expected:
        how = "a mention moves it by one" if mention else "only a mention moves it"
        raise TargetStoreError(
            f"target {incoming.id} can't go from revision {existing.revision} to "
            f"{incoming.revision}: {how}"
        )
    if incoming.confidence != existing.confidence:
        raise TargetStoreError(
            f"target {incoming.id}'s confidence moves only through a verified homework result"
        )


def _upsert(state: TargetState, event: Mapping[str, Any]) -> TargetState:
    incoming = target_from_record(event["target"])
    evidence_id = event["evidence_id"]
    if evidence_id is not None and _str(evidence_id) in state.mentioned:
        raise TargetStoreError(f"evidence {evidence_id} was already counted")
    existing = state.targets.get(incoming.id)
    _check_upsert(existing, incoming, mention=evidence_id is not None)
    first_seen = state.first_seen
    if existing is None:
        stored = incoming
        first_seen = MappingProxyType({**first_seen, incoming.id: incoming.last_mentioned})
    else:
        stored = replace(incoming, acted_revision=existing.acted_revision, due_noted=existing.due_noted)
    mentioned = state.mentioned if evidence_id is None else state.mentioned | {evidence_id}
    return replace(_with_target(state, stored), first_seen=first_seen, mentioned=mentioned)


def _mark_acted(state: TargetState, event: Mapping[str, Any]) -> TargetState:
    target = _known(state, event["target_id"])
    revision = _int(event["revision"])
    if revision > target.revision:
        raise TargetStoreError(f"revision {revision} is beyond target {target.id}'s {target.revision}")
    if revision < target.acted_revision:
        raise TargetStoreError(f"target {target.id} can't move back to revision {revision}")
    return _with_target(state, replace(target, acted_revision=revision))


def _add_result(state: TargetState, event: Mapping[str, Any]) -> TargetState:
    result = result_from_record(event["result"])
    target = _known(state, result.target_id)
    if any(stored.id == result.id for stored in state.results):
        raise TargetStoreError(f"result {result.id} is already recorded")
    step = event["confidence_step"]
    if not isinstance(step, (int, float)) or isinstance(step, bool):
        raise TypeError("confidence_step must be a number")
    verified = (
        result.basis == "research" and result.homework_id is not None and result.verified_sources >= 1
    )
    if verified and target.kind == "thread" and target.confidence is not None:
        target = replace(target, confidence=min(1.0, round(target.confidence + step, 9)))
    return replace(_with_target(state, target), results=(*state.results, result))


def _settle(state: TargetState, event: Mapping[str, Any]) -> TargetState:
    target = _known(state, event["target_id"])
    results = tuple(
        replace(result, consumed=True) if result.target_id == target.id else result
        for result in state.results
    )
    if target.kind == "commitment":
        target = replace(target, due_noted=True)
    return replace(_with_target(state, target), results=results)


def _record_expectation(state: TargetState, event: Mapping[str, Any]) -> TargetState:
    expectation = expectation_from_record(event["expectation"])
    _known(state, expectation.target_id)
    if any(stored.id == expectation.id for stored in state.expectations):
        raise TargetStoreError(f"expectation {expectation.id} is already recorded")
    return replace(state, expectations=(*state.expectations, expectation))


def _resolve_expectation(state: TargetState, event: Mapping[str, Any]) -> TargetState:
    expectation_id = _str(event["expectation_id"])
    outcome = _choice(event["outcome"], _OUTCOMES)
    found = [stored for stored in state.expectations if stored.id == expectation_id]
    if not found:
        raise TargetStoreError(f"unknown expectation {expectation_id!r}")
    if found[0].outcome is not None:
        raise TargetStoreError(f"expectation {expectation_id} is already resolved")
    return replace(state, expectations=tuple(
        replace(stored, outcome=outcome) if stored.id == expectation_id else stored
        for stored in state.expectations
    ))


_HANDLERS: Mapping[str, Callable[[TargetState, Mapping[str, Any]], TargetState]] = MappingProxyType({
    "upsert": _upsert,
    "mark_acted": _mark_acted,
    "add_result": _add_result,
    "settle": _settle,
    "record_expectation": _record_expectation,
    "resolve_expectation": _resolve_expectation,
})


def apply(state: TargetState, event: Mapping[str, Any]) -> TargetState:
    """The state after one event; ``TargetStoreError`` when the event can't follow ``state``."""
    if event.get("seq") != state.seq:
        raise TargetStoreError(f"event seq {event.get('seq')!r} where {state.seq} was expected")
    op = event.get("op")
    handler = _HANDLERS.get(op) if isinstance(op, str) else None
    if handler is None:
        raise TargetStoreError(f"unknown event op {op!r}")
    try:
        folded = handler(state, event)
    except TargetStoreError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise TargetStoreError(f"event {state.seq} ({op}) can't be read: {exc}") from None
    return replace(folded, seq=state.seq + 1)


def fold(events: Iterable[Mapping[str, Any]]) -> TargetState:
    state = EMPTY
    for event in events:
        state = apply(state, event)
    return state


def state_document(state: TargetState) -> dict[str, Any]:
    """The derived state as JSON: what the export carries and the digest hashes."""
    return {
        "targets": [
            {**target_record(target), "first_seen": state.first_seen[target.id].isoformat()}
            for target in sorted(state.targets.values(), key=lambda target: target.id)
        ],
        "results": [result_record(result) for result in state.results],
        "expectations": [expectation_record(expectation) for expectation in state.expectations],
        "counted_evidence": sorted(state.mentioned),
    }


def state_digest(state: TargetState) -> str:
    return hashlib.sha256(canonical(state_document(state)).encode("ascii")).hexdigest()


# ── The store ─────────────────────────────────────────────────────────────────


class WorkspaceRecords(Protocol):
    """What the person's digest reads of her MIND workspace log: its records' digest."""

    def records_digest(self) -> str: ...


def _with_workspace(targets_digest: str, workspace: WorkspaceRecords | None) -> str:
    if workspace is None:
        return targets_digest
    return hashlib.sha256(f"{targets_digest}:{workspace.records_digest()}".encode("ascii")).hexdigest()


class EventLogTargetStore:
    """The ``TargetStore`` port over one log file. Before each read or write it folds in any
    events another writer appended; ``guard`` serializes writers across processes. The codec and
    the guard are always named, so a caller can't write plain text by leaving one out."""

    def __init__(
        self, path: Path, person_id: str, *, codec: Codec, guard: Guard, workspace: WorkspaceRecords | None = None,
    ) -> None:
        self.path = path
        self.person_id = person_id
        self._codec = codec
        self._guard = guard
        self._workspace = workspace
        self._state = EMPTY

    def _current(self) -> TargetState:
        events = read_records(self.path, self._codec)
        if len(events) < self._state.seq:
            raise TargetStoreError("the log is shorter than what this store already read")
        state = self._state
        for event in events[state.seq:]:
            state = apply(state, event)
        self._state = state
        return state

    def _append(self, state: TargetState, event: Mapping[str, Any]) -> None:
        record = {**event, "seq": state.seq}
        folded = apply(state, record)
        append_records(self.path, self._codec, [record])
        self._state = folded

    def _commit(self, event: Mapping[str, Any]) -> None:
        with self._guard():
            self._append(self._current(), event)

    def upsert(self, target: Target) -> None:
        """Create a target at revision 1, or change an existing one's label, status or date."""
        if target.person_id != self.person_id:
            raise TargetStoreError(f"target {target.id} is another person's")
        self._commit({"op": "upsert", "target": target_record(target), "evidence_id": None})

    def record_mention(self, mention: Mention, evidence: Evidence) -> bool:
        """Count one item's mention: create the target or raise its revision. False when the
        item was already counted, so a re-run of a night changes nothing."""
        if evidence.kind not in PERSON_KINDS:
            raise TargetStoreError(f"only the person's own evidence raises a target, not {evidence.kind}")
        if evidence.held_out:
            raise TargetStoreError("a held out item never touches a target")
        if evidence.person_id != self.person_id or evidence.target_id != mention.target_id:
            raise TargetStoreError(f"evidence {evidence.id} doesn't name this person's target")
        with self._guard():
            state = self._current()
            if evidence.id in state.mentioned:
                return False
            target = mentioned_target(
                state.targets.get(mention.target_id), mention, person_id=self.person_id,
                day=evidence.day,
            )
            self._append(state, {"op": "upsert", "target": target_record(target), "evidence_id": evidence.id})
        return True

    def get(self, target_id: str) -> Target | None:
        return self._current().targets.get(target_id)

    def targets(self) -> list[Target]:
        return sorted(self._current().targets.values(), key=lambda target: target.id)

    def open_targets(self, as_of: datetime) -> list[Target]:
        state = self._current()
        day = as_of.date()
        return [
            target for target in sorted(state.targets.values(), key=lambda target: target.id)
            if target.status == "open" and state.first_seen[target.id] <= day
        ]

    def mark_acted(self, target_id: str, revision: int) -> None:
        self._commit({"op": "mark_acted", "target_id": target_id, "revision": revision})

    def add_result(self, result: Result) -> None:
        self._commit({"op": "add_result", "result": result_record(result), "confidence_step": CONFIDENCE_STEP})

    def results(self) -> list[Result]:
        return list(self._current().results)

    def settle(self, target_id: str) -> None:
        self._commit({"op": "settle", "target_id": target_id})

    def record_expectation(self, expectation: Expectation) -> None:
        self._commit({"op": "record_expectation", "expectation": expectation_record(expectation)})

    def resolve_expectation(self, expectation_id: str, outcome: ExpectationOutcome) -> None:
        self._commit({"op": "resolve_expectation", "expectation_id": expectation_id, "outcome": outcome})

    def expectations(self) -> list[Expectation]:
        return list(self._current().expectations)

    def digest(self) -> str:
        return _with_workspace(state_digest(self._current()), self._workspace)

    def replay_digest(self) -> str:
        return _with_workspace(state_digest(fold(read_records(self.path, self._codec))), self._workspace)

    def export(self) -> dict[str, object]:
        return state_document(self._current())
