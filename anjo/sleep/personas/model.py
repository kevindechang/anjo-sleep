"""Persona files: the visible part a simulator may play, and the held-out part only tests read.

One `PersonFile` shape serves the film student, her donor and every expanded life
(`expand_life`, Task 12a, must emit exactly this). Only `generate_probes` and `judge_brief`
read `held_out` (RF5).

Plan.day is the day on which the person brings the plan up in conversation.
NewWork.day is the day the new item arrives in the person's life.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PLANNED_NIGHTS = 8
NEW_WORK_FIRST_DAY = 2
MIN_PREFERENCES_TOTAL = 75
MIN_HELD_OUT_PREFERENCES = 15
MIN_HELD_OUT_SHARE = 0.20
MIN_NEW_WORK = 25
MIN_PLANS = 3
MIN_TONE_REQUESTS = 3
MIN_PROJECT_CONSTRAINTS = 3
MIN_ADULT_AGE = 18


class PersonaError(ValueError):
    """A persona file broke a rule."""


@dataclass(frozen=True)
class Statement:
    id: str
    text: str
    domain: str = ""


@dataclass(frozen=True)
class Plan:
    day: int
    text: str


@dataclass(frozen=True)
class Project:
    title: str
    logline: str
    constraints: tuple[str, ...]


@dataclass(frozen=True)
class Persona:
    """The visible part: everything the simulator may play and a conversation may reveal."""

    person_id: str
    name: str
    age: int
    field: str
    field_domains: tuple[str, ...]
    line: str
    project: Project
    preferences: tuple[Statement, ...]
    tone_requests: tuple[Statement, ...]
    stated_views: tuple[Statement, ...]
    plans: tuple[Plan, ...]


@dataclass(frozen=True)
class TasteItem:
    """A held-out forced choice. `pick` is the index of the option this person prefers."""

    id: str
    pref_id: str
    prompt: str
    options: tuple[str, str]
    pick: int


@dataclass(frozen=True)
class NewWork:
    id: str
    day: int
    parent_project: str
    text: str


@dataclass(frozen=True)
class HeldOut:
    preferences: tuple[Statement, ...]
    taste_ab: tuple[TasteItem, ...]
    new_work: tuple[NewWork, ...]


@dataclass(frozen=True)
class PersonFile:
    visible: Persona
    held_out: HeldOut


# ---------------------------------------------------------------- parsing


def _req(d: dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise PersonaError(f"{where}: missing '{key}'")
    return d[key]


def _text(d: dict[str, Any], key: str, where: str) -> str:
    v = _req(d, key, where)
    if not isinstance(v, str) or not v.strip():
        raise PersonaError(f"{where}: '{key}' must be non-empty text")
    return v


def _statements(rows: Any, where: str) -> tuple[Statement, ...]:
    if not isinstance(rows, list):
        raise PersonaError(f"{where}: must be a list")
    return tuple(
        Statement(_text(r, "id", where), _text(r, "text", where), str(r.get("domain", "")))
        for r in rows
    )


def _int(d: dict[str, Any], key: str, where: str) -> int:
    v = _req(d, key, where)
    if isinstance(v, bool) or not isinstance(v, int):
        raise PersonaError(f"{where}: '{key}' must be an integer")
    return v


def _persona_from(d: dict[str, Any]) -> Persona:
    proj = _req(d, "project", "visible")
    return Persona(
        person_id=_text(d, "person_id", "visible"),
        name=_text(d, "name", "visible"),
        age=_int(d, "age", "visible"),
        field=_text(d, "field", "visible"),
        field_domains=tuple(str(x) for x in _req(d, "field_domains", "visible")),
        line=_text(d, "line", "visible"),
        project=Project(
            _text(proj, "title", "project"),
            _text(proj, "logline", "project"),
            tuple(str(c) for c in _req(proj, "constraints", "project")),
        ),
        preferences=_statements(_req(d, "preferences", "visible"), "preferences"),
        tone_requests=_statements(_req(d, "tone_requests", "visible"), "tone_requests"),
        stated_views=_statements(_req(d, "stated_views", "visible"), "stated_views"),
        plans=tuple(
            Plan(_int(p, "day", "plan"), _text(p, "text", "plan"))
            for p in _req(d, "plans", "visible")
        ),
    )


def _held_out_from(d: dict[str, Any]) -> HeldOut:
    taste = []
    for t in _req(d, "taste_ab", "held_out"):
        opts = _req(t, "options", "taste_ab")
        if not isinstance(opts, list) or len(opts) != 2:
            raise PersonaError("taste_ab: options must be a pair")
        taste.append(
            TasteItem(
                _text(t, "id", "taste_ab"),
                _text(t, "pref_id", "taste_ab"),
                _text(t, "prompt", "taste_ab"),
                (str(opts[0]), str(opts[1])),
                _int(t, "pick", "taste_ab"),
            )
        )
    work = tuple(
        NewWork(
            _text(w, "id", "new_work"),
            _int(w, "day", "new_work"),
            _text(w, "parent_project", "new_work"),
            _text(w, "text", "new_work"),
        )
        for w in _req(d, "new_work", "held_out")
    )
    return HeldOut(
        preferences=_statements(_req(d, "preferences", "held_out"), "held_out.preferences"),
        taste_ab=tuple(taste),
        new_work=work,
    )


def person_file_from_dict(data: dict[str, Any]) -> PersonFile:
    return PersonFile(
        _persona_from(_req(data, "visible", "file")),
        _held_out_from(_req(data, "held_out", "file")),
    )


# ---------------------------------------------------------------- validation


def validate_person_file(pf: PersonFile, *, planned_nights: int = PLANNED_NIGHTS) -> None:
    """Raise PersonaError unless the file meets Task 2's counts and the leak rules."""
    v, h = pf.visible, pf.held_out
    if not v.field_domains or any(
        not d.strip() or d != d.lower() for d in v.field_domains
    ):
        raise PersonaError("field_domains must be non-empty lowercase tags")
    if v.age < MIN_ADULT_AGE:
        raise PersonaError("persona must be an adult")
    if len(v.plans) < MIN_PLANS or any(p.day < 0 for p in v.plans):
        raise PersonaError(f"need at least {MIN_PLANS} dated plans")
    if len(v.tone_requests) < MIN_TONE_REQUESTS:
        raise PersonaError("need tone requests")
    if len(v.project.constraints) < MIN_PROJECT_CONSTRAINTS:
        raise PersonaError("project needs constraints")

    total = len(v.preferences) + len(h.preferences)
    if total < MIN_PREFERENCES_TOTAL:
        raise PersonaError(f"need at least {MIN_PREFERENCES_TOTAL} preferences, got {total}")
    if len(h.preferences) < MIN_HELD_OUT_PREFERENCES:
        raise PersonaError(f"need at least {MIN_HELD_OUT_PREFERENCES} held-out preferences")
    if len(h.preferences) < math.ceil(MIN_HELD_OUT_SHARE * total):
        raise PersonaError("held-out preferences must be at least 20% of all preferences")

    ids = [s.id for s in (*v.preferences, *h.preferences)]
    if len(set(ids)) != len(ids):
        raise PersonaError("preference ids must be unique")
    visible_texts = {s.text.strip().lower() for s in v.preferences}
    for s in h.preferences:
        if s.text.strip().lower() in visible_texts:
            raise PersonaError(f"held-out preference {s.id} also appears in the visible part")

    held_ids = {s.id for s in h.preferences}
    if len(h.taste_ab) < len(h.preferences):
        raise PersonaError("every held-out preference needs a taste_ab item")
    seen_prefs: set[str] = set()
    for t in h.taste_ab:
        if t.pref_id not in held_ids or t.pref_id in seen_prefs:
            raise PersonaError(f"taste item {t.id} must map to one distinct held-out preference")
        seen_prefs.add(t.pref_id)
        if t.pick not in (0, 1) or t.options[0] == t.options[1]:
            raise PersonaError(f"taste item {t.id} is malformed")

    _validate_new_work(pf, planned_nights)


def _validate_new_work(pf: PersonFile, planned_nights: int) -> None:
    v, h = pf.visible, pf.held_out
    if len(h.new_work) < MIN_NEW_WORK:
        raise PersonaError(f"need at least {MIN_NEW_WORK} new-work items")
    last_day = planned_nights + 7
    ids = [w.id for w in h.new_work]
    if len(set(ids)) != len(ids):
        raise PersonaError("new-work ids must be unique")
    visible_blob = json.dumps(to_visible_dict(v)).lower()
    for w in h.new_work:
        if not NEW_WORK_FIRST_DAY <= w.day <= last_day:
            raise PersonaError(f"new-work {w.id} arrives on day {w.day}, outside 2..{last_day}")
        if w.parent_project != v.project.title:
            raise PersonaError(f"new-work {w.id} has an unknown parent project")
        if w.text.strip().lower() in visible_blob:
            raise PersonaError(f"new-work {w.id} appears in the visible part")


def load_person_file(
    path: str | Path, *, planned_nights: int = PLANNED_NIGHTS
) -> PersonFile:
    """Read, parse and validate a persona or life file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pf = person_file_from_dict(data)
    validate_person_file(pf, planned_nights=planned_nights)
    return pf


# ---------------------------------------------------------------- serialisation


def _stmt(s: Statement) -> dict[str, str]:
    return {"id": s.id, "text": s.text, "domain": s.domain}


def to_visible_dict(v: Persona) -> dict[str, Any]:
    return {
        "person_id": v.person_id,
        "name": v.name,
        "age": v.age,
        "field": v.field,
        "field_domains": list(v.field_domains),
        "line": v.line,
        "project": {
            "title": v.project.title,
            "logline": v.project.logline,
            "constraints": list(v.project.constraints),
        },
        "preferences": [_stmt(s) for s in v.preferences],
        "tone_requests": [_stmt(s) for s in v.tone_requests],
        "stated_views": [_stmt(s) for s in v.stated_views],
        "plans": [{"day": p.day, "text": p.text} for p in v.plans],
    }


def to_held_out_dict(h: HeldOut) -> dict[str, Any]:
    return {
        "preferences": [_stmt(s) for s in h.preferences],
        "taste_ab": [
            {
                "id": t.id,
                "pref_id": t.pref_id,
                "prompt": t.prompt,
                "options": list(t.options),
                "pick": t.pick,
            }
            for t in h.taste_ab
        ],
        "new_work": [
            {"id": w.id, "day": w.day, "parent_project": w.parent_project, "text": w.text}
            for w in h.new_work
        ],
    }


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
