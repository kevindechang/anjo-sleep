"""The example's invented person and her one fixed day (``persona.json``, ``day1.jsonl``).

Nothing here is a real person, and none of it is one of the registered test personas. The file
has a visible part (what the day reveals), and a held-out part that only the sealed probes and the
judge read: three preferences the day never states, and three pieces of new work that no training
input contains. Loading refuses a file that breaks those rules, with a message that names the field.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Final, get_args

from anjo.sleep.evidence.extract import Turn
from anjo.sleep.types import Evidence, EvidenceKind, Probe

HERE: Final = Path(__file__).parent
SCHEMA: Final = "anjo-sleep-example/1"
VISIBLE_KINDS: Final = frozenset(get_args(EvidenceKind)) - {"homework", "practice", "agent_note"}
MIN_NEW_WORK: Final = 3


class ExampleInputError(ValueError):
    """The persona or the day file breaks a rule. The message names the file and the field."""


@dataclass(frozen=True)
class Statement:
    kind: EvidenceKind
    text: str


@dataclass(frozen=True)
class HeldPreference:
    id: str
    text: str


@dataclass(frozen=True)
class ExamplePersona:
    person_id: str
    name: str
    age: int
    field: str
    line: str
    project_title: str
    logline: str
    constraints: tuple[str, ...]
    day: date
    statements: tuple[Statement, ...]
    held_out_preferences: tuple[HeldPreference, ...]
    new_work: tuple[str, ...]


@dataclass(frozen=True)
class DayTurn:
    role: str
    text: str
    at: datetime


def resolve(name: str | Path) -> Path:
    """The file ``name`` points at: as given when it exists, else the bundled file of that name."""
    given = Path(name)
    if given.is_file():
        return given
    bundled = HERE / given.name
    return bundled if bundled.is_file() else given


def _need(data: dict[str, Any], key: str, where: str) -> Any:
    if key not in data:
        raise ExampleInputError(f"{where}: missing '{key}'")
    return data[key]


def _text(data: dict[str, Any], key: str, where: str) -> str:
    value = _need(data, key, where)
    if not isinstance(value, str) or not value.strip():
        raise ExampleInputError(f"{where}: '{key}' must be non-empty text")
    return value.strip()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        raise ExampleInputError(f"{path}: can't be read") from None
    except ValueError:
        raise ExampleInputError(f"{path}: not JSON") from None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ExampleInputError(f"{path}: not an {SCHEMA} persona file")
    return data


def load_persona(path: Path) -> ExamplePersona:
    data = _read_json(path)
    where = path.name
    visible, held = _need(data, "visible", where), _need(data, "held_out", where)
    project = _need(data, "project", where)
    statements = tuple(_statement(row, where) for row in _need(visible, "statements", where))
    prefs = tuple(
        HeldPreference(_text(row, "id", where), _text(row, "text", where))
        for row in _need(held, "preferences", where)
    )
    work = tuple(str(text).strip() for text in _need(held, "new_work", where) if str(text).strip())
    if len(work) < MIN_NEW_WORK:
        raise ExampleInputError(f"{where}: held_out.new_work needs at least {MIN_NEW_WORK} items")
    if not prefs:
        raise ExampleInputError(f"{where}: held_out.preferences is empty")
    try:
        day = date.fromisoformat(_text(data, "day", where))
    except ValueError:
        raise ExampleInputError(f"{where}: 'day' must be YYYY-MM-DD") from None
    return ExamplePersona(
        person_id=_text(data, "person_id", where), name=_text(data, "name", where),
        age=int(_need(data, "age", where)), field=_text(data, "field", where),
        line=_text(data, "line", where), project_title=_text(project, "title", where),
        logline=_text(project, "logline", where),
        constraints=tuple(str(item) for item in _need(project, "constraints", where)),
        day=day, statements=statements, held_out_preferences=prefs, new_work=work,
    )


def _statement(row: dict[str, Any], where: str) -> Statement:
    kind = _text(row, "kind", where)
    if kind not in VISIBLE_KINDS:
        raise ExampleInputError(f"{where}: statement kind '{kind}' isn't one the day can show")
    return Statement(kind=kind, text=_text(row, "text", where))  # type: ignore[arg-type]


def load_day(path: Path, day: date) -> tuple[DayTurn, ...]:
    """The day's turns in order: ``{"role", "at": "HH:MM", "text"}`` per line, on ``day`` (UTC)."""
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        raise ExampleInputError(f"{path}: can't be read") from None
    turns: list[DayTurn] = []
    for number, line in enumerate(lines, start=1):
        where = f"{path.name} line {number}"
        try:
            row = json.loads(line)
            at = datetime.combine(day, time.fromisoformat(row["at"]), tzinfo=timezone.utc)
            role, text = row["role"], row["text"]
        except (ValueError, KeyError, TypeError):
            raise ExampleInputError(f"{where}: needs role, at (HH:MM) and text") from None
        if role not in {"user", "assistant"} or not isinstance(text, str) or not text.strip():
            raise ExampleInputError(f"{where}: role is user or assistant, and text is not empty")
        turns.append(DayTurn(role=role, text=text.strip(), at=at))
    if not any(turn.role == "user" for turn in turns):
        raise ExampleInputError(f"{path.name}: the day has no message from the person")
    return tuple(turns)


def turns_of(day_turns: tuple[DayTurn, ...], person_id: str) -> list[Turn]:
    """The day as extraction reads it. The turns are hand-written, so their ancestry is empty."""
    return [
        Turn(source=f"{person_id}:{index}", role=turn.role, text=turn.text, ancestry=frozenset())
        for index, turn in enumerate(day_turns)
    ]


def _evidence(person: ExamplePersona, ident: str, kind: EvidenceKind, text: str, *, held: bool) -> Evidence:
    return Evidence(
        id=ident, person_id=person.person_id, day=person.day, kind=kind, text=text,
        source=f"persona:{ident}", held_out=held,
        created_at=datetime.combine(person.day, time(0), tzinfo=timezone.utc),
        model=None, ancestry=frozenset(),
    )


def visible_evidence(person: ExamplePersona) -> list[Evidence]:
    """What the day reveals, as the ground the written probes stand on."""
    return [
        _evidence(person, f"{person.person_id}-v{index}", item.kind, item.text, held=False)
        for index, item in enumerate(person.statements)
    ]


def held_out_evidence(person: ExamplePersona) -> list[Evidence]:
    return [
        _evidence(person, f"{person.person_id}-{item.id}", "preference", item.text, held=True)
        for item in person.held_out_preferences
    ]


def new_work_probes(person: ExamplePersona) -> list[Probe]:
    """The held-out new work, sealed as ``new_work`` probes: each is the person's own message."""
    return [
        Probe(
            id=f"{person.person_id}-new_work-{index + 1:02d}", person_id=person.person_id,
            kind="new_work", prompt=text, options=None, expected=None,
            source_ids=(f"{person.person_id}-nw{index + 1}",),
        )
        for index, text in enumerate(person.new_work)
    ]


def judge_brief(person: ExamplePersona) -> str:
    """The judge's brief: the person as they really are, held-out preferences included. Only the
    judge reads it, never an arm."""
    visible = "\n".join(f"- {item.text}" for item in person.statements)
    unseen = "\n".join(f"- {item.text}" for item in person.held_out_preferences)
    rules = "\n".join(f"- {item}" for item in person.constraints)
    return "\n\n".join((
        f"{person.name}, {person.age}. Field: {person.field}. {person.line}",
        f"## Their project\n{person.project_title}: {person.logline}\nConstraints:\n{rules}",
        f"## What the assistants saw\n{visible}",
        f"## Preferences no assistant saw\n{unseen}",
        "Never mark a `new_work` reply down for a missing date or name: exact facts are kept "
        "elsewhere by design.",
    ))
