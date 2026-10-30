"""The two briefs each night folds, kept per person (Task 5, step 6).

Every fold appends one record to the person's brief log: the served brief and the training brief
after the night, each with its ancestry, the moment they were compiled on the night's clock, and
the ids of the evidence the night folded in, so the next night folds only what came after (the
night's practice items included, which are stored after its fold).

**Ancestry.** A brief derives from its previous version, from the items folded into it and from
the model that compiled it (Spark Open, called only when there are items to fold). So its ancestry
is the union of all three, and ``None`` when any part is unknown. Text built from a brief with
unknown ancestry never trains, so the training brief, which only ever folds items that may train,
always has known ancestry. ``compiler`` names the id the brief records: the one
``ANJO_SLEEP_SPARK`` selects (Baseten's by default).

``ServedBriefs`` and ``TrainingBriefs`` are the two ``BriefSource`` views the context builders
read (``anjo.sleep.app.prompt.set_context_setup``). The log is append-only JSON Lines: the app
passes encryption and the account's writer guard, tests and the public example pass ``PLAIN``.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from anjo.sleep.context.brief import DEFAULT_BUDGET_TOKENS, EMPTY_BRIEF, BriefChat, BriefError, BriefState, fold_day
from anjo.sleep.context.mind_brief import refresh
from anjo.sleep.context.builder import StoredBrief
from anjo.sleep.evidence.jsonl import Codec, Guard, LogError, append_records, read_records
from anjo.sleep.provenance import may_train
from anjo.sleep.types import Evidence


def compiler() -> str:
    """The id the brief records as its compiler: Spark Open's on the provider
    ``ANJO_SLEEP_SPARK`` selects (Baseten's by default)."""
    from anjo.sleep import spark_provider  # imported here: spark_provider imports this package

    return spark_provider.current().model


def union(parts: Iterable[frozenset[str] | None]) -> frozenset[str] | None:
    """Every model the parts derive from, or ``None`` when any part's ancestry is unknown."""
    names: set[str] = set()
    for part in parts:
        if part is None:
            return None
        names |= part
    return frozenset(names)


def folded_ancestry(
    prev: frozenset[str] | None, items: Sequence[Evidence],
) -> frozenset[str] | None:
    """A brief's ancestry after folding ``items`` into a brief of ancestry ``prev``."""
    if not items:
        return prev
    return union((prev, *(item.ancestry for item in items), frozenset({compiler()})))


@dataclass(frozen=True)
class BriefRecord:
    night: date
    served: str
    served_ancestry: frozenset[str] | None
    training: str
    training_ancestry: frozenset[str] | None
    compiled_at: datetime
    folded: tuple[str, ...]


def _names(value: frozenset[str] | None) -> list[str] | None:
    return None if value is None else sorted(value)


def _from_names(value: object) -> frozenset[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(name, str) for name in value):
        raise LogError("a brief's ancestry is a list of model names or null")
    return frozenset(value)


def brief_record(record: BriefRecord) -> dict[str, Any]:
    return {
        "night": record.night.isoformat(), "served": record.served,
        "served_ancestry": _names(record.served_ancestry), "training": record.training,
        "training_ancestry": _names(record.training_ancestry),
        "compiled_at": record.compiled_at.isoformat(), "folded": list(record.folded),
    }


def brief_from_record(record: Mapping[str, Any]) -> BriefRecord:
    try:
        return BriefRecord(
            night=date.fromisoformat(record["night"]), served=str(record["served"]),
            served_ancestry=_from_names(record["served_ancestry"]), training=str(record["training"]),
            training_ancestry=_from_names(record["training_ancestry"]),
            compiled_at=datetime.fromisoformat(record["compiled_at"]),
            folded=tuple(str(item) for item in record["folded"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LogError(f"not a brief record ({type(error).__name__})") from None


class JsonlBriefStore:
    """One person's brief log over one file."""

    def __init__(self, path: Path, person_id: str, *, codec: Codec, guard: Guard) -> None:
        self.path = path
        self.person_id = person_id
        self._codec = codec
        self._guard = guard

    def records(self) -> list[BriefRecord]:
        return [brief_from_record(record) for record in read_records(self.path, self._codec)]

    def latest(self) -> BriefRecord | None:
        found = self.records()
        return found[-1] if found else None

    def folded_ids(self) -> frozenset[str]:
        return frozenset(item for record in self.records() for item in record.folded)

    def unfolded(self, stored: Sequence[Evidence]) -> list[Evidence]:
        """The stored items no night has folded yet, oldest first."""
        done = self.folded_ids()
        return sorted((item for item in stored if item.id not in done), key=lambda e: (e.created_at, e.id))

    def fold(
        self, night: date, new: Sequence[Evidence], *, at: datetime, hosted_in_training: bool,
        chat: BriefChat | None = None,
        mind_refresh: bool = False,
    ) -> BriefRecord:
        """Fold ``new`` into both briefs, store the result and return it."""
        if any(item.person_id != self.person_id for item in new):
            raise ValueError("evidence for another person")
        prev = self.latest()
        state = BriefState(
            served=prev.served if prev else None, training=prev.training if prev else None,
        )
        try:
            folded = fold_day(state, new, hosted_in_training=hosted_in_training, chat=chat)
        except BriefError:
            homework = [item for item in new if item.kind == "homework" and item.id.startswith("mind-")]
            if not mind_refresh or not homework:
                raise
            served, included = refresh(state.served, homework, DEFAULT_BUDGET_TOKENS)
            # A fallback folds only complete homework; other evidence is still owed a normal fold.
            new = included
            trainable_homework = [item for item in new if may_train(
                item.ancestry, hosted_in_training=hosted_in_training)]
            training = state.training
            if trainable_homework:
                training, _ = refresh(training, trainable_homework, DEFAULT_BUDGET_TOKENS)
            folded = BriefState(served, training if training is not None else EMPTY_BRIEF)
        trainable = [item for item in new if may_train(item.ancestry, hosted_in_training=hosted_in_training)]
        record = BriefRecord(
            night=night, served=str(folded.served), training=str(folded.training),
            served_ancestry=folded_ancestry(prev.served_ancestry if prev else frozenset(), list(new)),
            training_ancestry=folded_ancestry(prev.training_ancestry if prev else frozenset(), trainable),
            compiled_at=at, folded=tuple(item.id for item in new),
        )
        with self._guard():
            append_records(self.path, self._codec, [brief_record(record)])
        return record

    def export(self) -> dict[str, object]:
        return {"briefs": [brief_record(record) for record in self.records()]}


BriefStores = Callable[[str], JsonlBriefStore]


class ServedBriefs:
    """``BriefSource`` over the served briefs: what served turns and arms 2C and 4 read."""

    def __init__(self, stores: BriefStores) -> None:
        self._stores = stores

    def served(self, user_id: str) -> StoredBrief | None:
        latest = self._stores(user_id).latest()
        if latest is None:
            return None
        return StoredBrief(latest.served, latest.compiled_at, latest.served_ancestry)


class TrainingBriefs:
    """``BriefSource`` over the training briefs: what cleared turns and training inputs read."""

    def __init__(self, stores: BriefStores) -> None:
        self._stores = stores

    def served(self, user_id: str) -> StoredBrief | None:
        latest = self._stores(user_id).latest()
        if latest is None:
            return None
        return StoredBrief(latest.training, latest.compiled_at, latest.training_ancestry)
