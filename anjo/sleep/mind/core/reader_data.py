"""The thought channel's shared pretraining rows: moments of generated lives where MIND's state
should, or should not, make Glimmer raise a problem (torch-free, pure apart from ``open``).

**Only core-train.** ``TrainingExport.open`` reads a data export (``mind-core-events-v1``) whose split
is ``core-train`` and whose labels aren't sealed, and nothing else: a core-val or core-test export,
a sealed one, or the counterfactual pairs (test material) stop it before any label is read. The
target writer (``reader_targets``) takes only a ``TrainingExport``, so test labels can't reach it.

**A moment** is one of the person's messages, and the core's state just before it (the reading's
key is ``<life>:<event id>``). A problem is *open* at a moment when its knowable event came before
the message and the person hasn't raised it yet (the message's day is before its surface day). In
each life (``moments_of``, seeded per life):

- one **raise moment** per planted problem: an ordinary message (no item's evidence) while it is
  open; its reply should raise every open problem, citing their events (``cite_ids``);
- one **quiet moment** before anything is knowable, one per decoy or reversal once it is knowable
  while no problem is open (it looks like a problem and isn't), and one **settled moment** per
  problem after the day the person raised it, while nothing else is open (the state still holds
  the thread's events, but it is no longer live); a life with nothing planted gives two quiet
  moments. A problem's surface day itself (the person raising it) is never used.

**The channel carries the current state; the weights carry stable knowledge.** So a raise moment's
reply raises a live thread at the right time, and every other reply recalls nothing: a settled
thread, a decoy, or a state from before the problem existed all earn the plain reply.

**A row** (``plan_rows``, ``training_row``) is a night dataset row (prompt, completion) with the
state it was made under (``core``, a ``night.core_snapshot.CoreSnapshot``). A raise moment's row
also carries its **contrast**: the same message under another moment's state where nothing is open
and the plain reply for it: the same life's settled moment first (the same thread's events, no
longer live), then an earlier quiet one, else another life's quiet moment; for
``drop_share`` of them the other state is the empty reading (``EMPTY_KEY``), so zero tokens learn
silence. The training job turns these into the state-conditional margin (``train.channel_train``).
"""
from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

from anjo.sleep.night.core_snapshot import CoreSnapshot
from anjo.sleep.serving import framing

EXPORT_SCHEMA: Final = "mind-core-events-v1"
TRAIN_SPLITS: Final = frozenset({"core-train"})
EMPTY_KEY: Final = "empty"
READER_SYSTEM: Final = ("You are Anjo, this person's own AI. You know their life from what they have told you. "
                        "Reply to their message.")
QUIET_EMPTY_LIFE: Final = 2
READER_DAY: Final = date(2026, 11, 1)  # the date the reader's rows are rendered on, in training and in every arm


class ReaderDataError(ValueError):
    """An export the reader may not train on, or rows that don't fit together."""


@dataclass(frozen=True)
class Event:
    event_id: str
    day: date
    text: str

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Event:
        return cls(str(record["event_id"]), date.fromisoformat(str(record["day"])), str(record["text"]))


@dataclass(frozen=True)
class Item:
    """One planted item of a labelled life (``labels/<life>.json``)."""

    item_id: str
    family: str
    actionable: bool
    topic: str
    resolution: str
    origin_ids: tuple[str, ...]
    disambiguating_ids: tuple[str, ...]
    knowable_event_id: str
    surface_day: date

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Item:
        return cls(str(record["item_id"]), str(record["family"]), bool(record["actionable"]),
                   str(record.get("topic", "")), str(record.get("resolution", "")),
                   tuple(str(i) for i in record["origin_ids"]), tuple(str(i) for i in record.get("disambiguating_ids", ())),
                   str(record["knowable_event_id"]), date.fromisoformat(str(record["surface_day"])))


@dataclass(frozen=True)
class Life:
    life_id: str
    field: str
    events: tuple[Event, ...]
    items: tuple[Item, ...]


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ReaderDataError(f"{path.name} can't be read: {type(error).__name__}") from None


def _ordered(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(records, key=lambda e: (str(e["t"]), int(e.get("seq", 0))))


@dataclass(frozen=True)
class TrainingExport:
    """A core-train export, read whole: its lives, their events in order, and their planted items."""

    split: str
    lives: tuple[Life, ...]

    @classmethod
    def open(cls, folder: Path) -> TrainingExport:
        if not (folder / "manifest.json").is_file():
            raise ReaderDataError("not a split's export (the counterfactual pairs are test material, never trained on)")
        manifest = _json(folder / "manifest.json")
        if manifest.get("schema") != EXPORT_SCHEMA:
            raise ReaderDataError(f"the export isn't {EXPORT_SCHEMA}")
        if manifest.get("split") not in TRAIN_SPLITS or manifest.get("sealed") is not False:
            raise ReaderDataError(f"the reader trains on {', '.join(sorted(TRAIN_SPLITS))} only, never a sealed "
                                  f"or held-out split ({manifest.get('split')!r})")
        lives = []
        for entry in manifest["lives"]:
            life_id = str(entry["life_id"])
            lines = (folder / "events" / f"{life_id}.jsonl").read_text(encoding="utf-8").splitlines()
            events = tuple(Event.from_record(e) for e in _ordered([json.loads(line) for line in lines if line.strip()]))
            label_file = folder / "labels" / f"{life_id}.json"
            items = tuple(Item.from_record(r) for r in (_json(label_file)["items"] if label_file.is_file() else ()))
            lives.append(Life(life_id, str(entry.get("field", "")), events, items))
        return cls(str(manifest["split"]), tuple(lives))

    def life(self, life_id: str) -> Life:
        found = next((life for life in self.lives if life.life_id == life_id), None)
        if found is None:
            raise ReaderDataError(f"no life {life_id} in this export")
        return found


@dataclass(frozen=True)
class Moment:
    life_id: str
    position: int  # the message's place in its life
    event_id: str
    day: date
    message: str
    open_items: tuple[str, ...]
    cite_ids: tuple[str, ...]

    @property
    def key(self) -> str:
        """The reading's key: the core's state just before this message."""
        return f"{self.life_id}:{self.event_id}"


def open_items(life: Life, at: int) -> tuple[Item, ...]:
    """The planted problems open at the life's ``at``-th message: knowable before it, not yet raised."""
    where = {e.event_id: n for n, e in enumerate(life.events)}
    day = life.events[at].day
    return tuple(i for i in life.items if i.actionable and where.get(i.knowable_event_id, len(life.events)) < at
                 and day < i.surface_day)


def _moment(life: Life, at: int) -> Moment:
    found = open_items(life, at)
    cites = tuple(dict.fromkeys(eid for item in found for eid in item.origin_ids))
    event = life.events[at]
    return Moment(life.life_id, at, event.event_id, event.day, event.text, tuple(i.item_id for i in found), cites)


def moments_of(life: Life, *, seed: int) -> tuple[Moment, ...]:
    """The life's raise and quiet moments (see the module), in life order."""
    rng = random.Random(f"{seed}:{life.life_id}")
    where = {e.event_id: n for n, e in enumerate(life.events)}
    evidence = {eid for item in life.items for eid in (*item.origin_ids, *item.disambiguating_ids)}
    raised_on = {i.surface_day for i in life.items if i.actionable}
    ordinary = [n for n, e in enumerate(life.events) if n > 0 and e.event_id not in evidence and e.day not in raised_on]
    quiet = [n for n in ordinary if not open_items(life, n)]
    chosen: set[int] = set()
    for item in sorted(life.items, key=lambda i: i.item_id):
        after = where.get(item.knowable_event_id, len(life.events))
        pool = [n for n in (ordinary if item.actionable else quiet) if n > after and life.events[n].day < item.surface_day
                and (not item.actionable or open_items(life, n))]
        settled = [n for n in quiet if item.actionable and life.events[n].day > item.surface_day]
        for found in (pool, settled):
            if found:
                chosen.add(rng.choice(found))
    first_knowable = min((where.get(i.knowable_event_id, len(life.events)) for i in life.items), default=len(life.events))
    before = [n for n in quiet if n < first_knowable]
    for _ in range(QUIET_EMPTY_LIFE if not life.items else 1):
        left = [n for n in before if n not in chosen]
        if left:
            chosen.add(rng.choice(left))
    return tuple(_moment(life, n) for n in sorted(chosen))


def all_moments(export: TrainingExport, *, seed: int) -> dict[str, tuple[Moment, ...]]:
    return {life.life_id: moments_of(life, seed=seed) for life in export.lives}


@dataclass(frozen=True)
class Plan:
    """One row to be: its moment, and for a raise moment the reading key of its contrast state."""

    row_id: str
    moment: Moment
    negative: str | None


def plan_rows(moments: Mapping[str, Sequence[Moment]], *, seed: int, drop_share: float) -> tuple[Plan, ...]:
    """Every moment's row; a raise moment's contrast state is a quiet moment of the same life (a
    settled one after it first, then an earlier one), else another life's, or the empty reading for
    ``drop_share`` of them."""
    if not 0.0 <= drop_share <= 1.0:
        raise ReaderDataError("drop_share is in [0, 1]")
    rng = random.Random(seed)
    quiet = {life: [m for m in found if not m.open_items] for life, found in moments.items()}
    plans = []
    for life in sorted(moments):
        for moment in moments[life]:
            negative: str | None = None
            if moment.open_items:
                elsewhere = [m for other in sorted(quiet) if other != life for m in quiet[other]]
                settled = [m for m in quiet[life] if m.position > moment.position]
                pool = settled or [m for m in quiet[life] if m.position < moment.position] or elsewhere
                dropped = rng.random() < drop_share
                negative = EMPTY_KEY if dropped or not pool else rng.choice(pool).key
            row_id = "rd-" + hashlib.sha256(f"{seed}:{moment.key}".encode()).hexdigest()[:16]
            plans.append(Plan(row_id, moment, negative))
    return tuple(plans)


@dataclass(frozen=True)
class Targets:
    """A moment's replies: the plain one (answers the message, raises nothing) and, at a raise
    moment, the one that also raises its open problems."""

    plain: str
    raised: str | None


def prompt_messages(message: str) -> list[dict[str, str]]:
    """The reader's prompt, the same in training and in every arm of the evaluation."""
    return framing.clean_messages([{"role": "system", "content": READER_SYSTEM}, {"role": "user", "content": message}])


def training_row(plan: Plan, readings: Mapping[str, CoreSnapshot], targets: Mapping[str, Targets]) -> dict[str, Any]:
    """The dataset row for ``plan``: the night's row form, with ``core`` and, at a raise moment, ``contrast``."""
    moment = plan.moment
    target = targets.get(moment.key)
    own = readings.get(moment.key)
    if target is None or own is None:
        raise ReaderDataError(f"moment {moment.key} has no target or no reading")
    row: dict[str, Any] = {"id": plan.row_id, "group": moment.life_id, "provenance": "teacher",
                           "prompt": prompt_messages(moment.message), "core": own.to_json()}
    if not moment.open_items:
        return {**row, "completion": [{"role": "assistant", "content": framing.clean_text(target.plain)}]}
    if target.raised is None or plan.negative is None:
        raise ReaderDataError(f"raise moment {moment.key} has no raise reply or no contrast state")
    other = None if plan.negative == EMPTY_KEY else readings.get(plan.negative)
    if plan.negative != EMPTY_KEY and other is None:
        raise ReaderDataError(f"the contrast state {plan.negative} has no reading")
    return {**row, "completion": [{"role": "assistant", "content": framing.clean_text(target.raised)}],
            "contrast": {"core": None if other is None else other.to_json(), "plain": framing.clean_text(target.plain)}}


# ── Files between the steps ───────────────────────────────────────────────────


def write_plans(path: Path, plans: Sequence[Plan]) -> None:
    rows = [{"row_id": p.row_id, "negative": p.negative, "life_id": p.moment.life_id, "position": p.moment.position,
             "event_id": p.moment.event_id, "day": p.moment.day.isoformat(), "message": p.moment.message,
             "open_items": list(p.moment.open_items), "cite_ids": list(p.moment.cite_ids)} for p in plans]
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def read_plans(path: Path) -> tuple[Plan, ...]:
    found = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            moment = Moment(str(r["life_id"]), int(r["position"]), str(r["event_id"]), date.fromisoformat(r["day"]),
                            str(r["message"]), tuple(r["open_items"]), tuple(r["cite_ids"]))
            found.append(Plan(str(r["row_id"]), moment, r.get("negative")))
    return tuple(found)


def moments_by_life(plans: Sequence[Plan]) -> dict[str, tuple[Moment, ...]]:
    lives: dict[str, list[Moment]] = {}
    for plan in plans:
        lives.setdefault(plan.moment.life_id, []).append(plan.moment)
    return {life: tuple(found) for life, found in lives.items()}


def write_targets(path: Path, targets: Mapping[str, Targets]) -> None:
    path.write_text("".join(json.dumps({"key": k, "plain": t.plain, "raise": t.raised}, ensure_ascii=False) + "\n"
                            for k, t in sorted(targets.items())), encoding="utf-8")


def append_target(path: Path, key: str, targets: Targets) -> None:
    """One more line of a targets file (the last line for a key wins on reading)."""
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"key": key, "plain": targets.plain, "raise": targets.raised}, ensure_ascii=False) + "\n")


def read_targets(path: Path) -> dict[str, Targets]:
    if not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {str(r["key"]): Targets(str(r["plain"]), r.get("raise")) for r in rows}


def write_readings(path: Path, readings: Mapping[str, tuple[CoreSnapshot, Sequence[Sequence[str]]]]) -> None:
    """Each key's snapshot and its routed slots' cited event ids (the text arm's memory, never on a row)."""
    path.write_text("".join(json.dumps({"key": k, "snapshot": s.to_json(), "cites": [list(c) for c in cites]}) + "\n"
                            for k, (s, cites) in sorted(readings.items())), encoding="utf-8")


def read_readings(path: Path) -> dict[str, tuple[CoreSnapshot, tuple[tuple[str, ...], ...]]]:
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            found[str(r["key"])] = (CoreSnapshot.from_json(r["snapshot"]), tuple(tuple(c) for c in r["cites"]))
    return found
