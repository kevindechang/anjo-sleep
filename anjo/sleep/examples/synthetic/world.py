"""The example's own stores and context: plain files in one folder, BM25 recall, no private Anjo.

The ports of ``anjo.sleep.ports`` are implemented with public parts. The evidence store, the
target log, the brief log and the transcript log are the engine's JSON Lines stores with the plain
codec. ``evidence_recall`` is BM25. Anjo's own memory is absent (``NoMemory``). ``LatestTrained``
stands in for the checkpoint registry: it hands the arms the adapter this run trained and
never promotes anything, because the example runs no gates (it has no crisis gate).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Final

from anjo.sleep.context.brief_store import JsonlBriefStore, ServedBriefs, TrainingBriefs
from anjo.sleep.context.builder import ComposedContextBuilder, RecallSettings
from anjo.sleep.context.recall_bm25 import Bm25Recall
from anjo.sleep.evidence.jsonl import NO_GUARD, PLAIN, append_records
from anjo.sleep.examples.synthetic.persona import DayTurn, ExamplePersona
from anjo.sleep.evidence.store import JsonlEvidenceStore
from anjo.sleep.evidence.targets import EventLogTargetStore
from anjo.sleep.practice.transcripts import JsonlTranscriptStore
from anjo.sleep.types import CheckpointRef

RECALL: Final = RecallSettings(k=6, budget_tokens=1500)
PREFERENCE_KINDS: Final = frozenset({"preference", "correction", "tone_request"})


class NoMemory:
    """Anjo's own memory is private; the example's context has none."""

    def read(self, user_id: str, message: str) -> str:
        return ""


class EvidencePreferences:
    """The person's stated preferences and corrections recorded after the brief was compiled."""

    def __init__(self, evidence: JsonlEvidenceStore) -> None:
        self._evidence = evidence

    def since(self, user_id: str, since: datetime | None) -> tuple[str, ...]:
        items = [
            item for item in self._evidence.stored(date.max)
            if item.kind in PREFERENCE_KINDS and (since is None or item.created_at > since)
        ]
        return tuple(item.text for item in sorted(items, key=lambda item: (item.created_at, item.id)))


class LatestTrained:
    """The arms' ``AdapterRegistry``: the checkpoint a person's run trained, if any.

    The name ``last_promoted`` is the port's. Here nothing is promoted: no gate ran, so this only
    says which adapter to try."""

    def __init__(self, trained: Mapping[str, CheckpointRef]) -> None:
        self._trained = dict(trained)

    def last_promoted(self, person_id: str) -> CheckpointRef | None:
        return self._trained.get(person_id)

    def with_trained(self, ckpt: CheckpointRef) -> LatestTrained:
        return LatestTrained({**self._trained, ckpt.person_id: ckpt})


@dataclass(frozen=True)
class World:
    """One person's folder of plain files, and the context built over it."""

    root: Path
    evidence: JsonlEvidenceStore
    targets: EventLogTargetStore
    briefs: JsonlBriefStore
    transcripts: JsonlTranscriptStore
    recall: Bm25Recall
    contexts: ComposedContextBuilder
    training_briefs: TrainingBriefs


def store_day(world: World, person: ExamplePersona, turns: Sequence[DayTurn]) -> None:
    """Step 1: the persona's visible part and the day's turns, as plain lines in the store. The
    held-out part is not stored: only the sealed probes and the judge read it."""
    append_records(world.root / "persona.jsonl", PLAIN, [{
        "person_id": person.person_id, "name": person.name, "age": person.age, "field": person.field,
        "line": person.line, "project": person.project_title, "day": person.day.isoformat(),
        "statements": [{"kind": s.kind, "text": s.text} for s in person.statements],
    }])
    append_records(world.root / "day.jsonl", PLAIN, [
        {"role": t.role, "at": t.at.isoformat(), "text": t.text} for t in turns
    ])


def open_world(root: Path, person_id: str, night: date) -> World:
    """Open (or start) a person's stores under ``root``. Everything is plain text on disk."""
    evidence = JsonlEvidenceStore(
        root / "evidence.jsonl", person_id, codec=PLAIN, guard=NO_GUARD, hosted_in_training=False,
    )
    targets = EventLogTargetStore(root / "targets.jsonl", person_id, codec=PLAIN, guard=NO_GUARD)
    briefs = JsonlBriefStore(root / "briefs.jsonl", person_id, codec=PLAIN, guard=NO_GUARD)
    transcripts = JsonlTranscriptStore(root / "practice.jsonl", person_id, codec=PLAIN, guard=NO_GUARD)
    recall = Bm25Recall(lambda _user: evidence, until=night)
    contexts = ComposedContextBuilder(
        memory=NoMemory(), briefs=ServedBriefs(lambda _user: briefs),
        preferences=EvidencePreferences(evidence), recall=recall, settings=RECALL,
    )
    return World(
        root=root, evidence=evidence, targets=targets, briefs=briefs, transcripts=transcripts,
        recall=recall, contexts=contexts, training_briefs=TrainingBriefs(lambda _user: briefs),
    )
