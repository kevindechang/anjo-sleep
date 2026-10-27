"""The APD generation gates (Task 17), run before any arm and frozen.

Each gate reads pre-surface text only: the days before an item's surface day, never the day the
person raises it, and never the item's resolution or template (``PlantedItem.topic`` is the
neutral name the readers get).

1. **Leak** (``leak_v1.md``): no single day and no two adjacent days, read alone, lets the reader
   flag the item.
2. **Discoverability** (``discover_v1.md``): a reader given the whole pre-surface prefix and the
   topic finds the item, citing every day its origin evidence sits on.
3. **Script audit:** the auditor checks the whole script against the sealed problem file.

A life passes only when every verdict passed. A reply that isn't the JSON asked for raises
``GateError``: a reader that failed to read is no verdict. No gate looks at an arm.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Final

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.context.brief import ChatFn
from anjo.sleep.mind.apd.types import GateVerdict, Life, PlantedItem, ProblemFile, Session

LEAK_PATH: Final = Path(__file__).parent / "prompts" / "leak_v1.md"
DISCOVER_PATH: Final = Path(__file__).parent / "prompts" / "discover_v1.md"
READER_MODEL: Final = SPARK_OPEN
AUDITOR_MODEL: Final = "muse-spark"
TIMEOUT_S: Final = 180.0
TEMPERATURE: Final = 0.0
AUDIT_PROMPT: Final = """# audit_v1

You audit a made-up person's scripted life against its sealed problem file. You see every day's
messages, dated, and the problem file: for each planted item its family, topic, surface date,
the origin facts and the disambiguating facts (each with its date and text) and the resolution.

Return one JSON object and nothing else: {"ok": true, "issues": []}.

Set "ok" to false and list "issues" (short sentences) if any of these holds, else set it to true:

- an item's origin or disambiguating fact is missing from its date, or is changed in substance;
- a contradiction's two halves are named together, or any item's facts are connected to each
  other, called a problem, or solved before the surface date;
- the surface date's messages do not raise the item's topic, or already contain its resolution;
- the script contains a problem, conflict, missing step or opening that is not in the problem
  file, or a fact that settles an item differently from its resolution;
- the script contradicts itself outside the file, or describes the person's health, feelings or
  character.

A problem file with no items means the script plants nothing: set "ok" to false if the script
contains any deliberate problem.
"""


class GateError(ValueError):
    """A reader's or the auditor's reply isn't the JSON asked for."""


def pre_surface(life: Life, item: PlantedItem) -> tuple[Session, ...]:
    """The sessions before ``item``'s surface day: the only text any gate reads for it."""
    return tuple(session for session in life.sessions if session.day < item.surface_day)


def check(
    life: Life, problems: ProblemFile, *, reader: ChatFn, auditor: ChatFn,
) -> tuple[GateVerdict, ...]:
    """Leak and discoverability for every item, then the script audit for the life."""
    if problems.life_id != life.id:
        raise ValueError("the problem file belongs to another life")
    verdicts: list[GateVerdict] = []
    for item in problems.items:
        days = _days(pre_surface(life, item))
        verdicts.append(_leak(item, days, reader))
        verdicts.append(_discoverability(life, item, days, reader))
    verdicts.append(_audit(life, problems, auditor))
    return tuple(verdicts)


def passed(verdicts: Sequence[GateVerdict]) -> bool:
    """Whether the life passes: every verdict passed."""
    return all(verdict.passed for verdict in verdicts)


def _days(sessions: Sequence[Session]) -> list[dict[str, str]]:
    """The sessions as one entry per day, oldest first."""
    by_day: dict[date, list[str]] = {}
    for session in sessions:
        by_day.setdefault(session.day, []).append(session.text)
    return [{"date": day.isoformat(), "text": "\n\n".join(by_day[day])} for day in sorted(by_day)]


def _windows(days: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    """Every single day, then every two adjacent days."""
    pairs = [
        [first, second] for first, second in zip(days, days[1:])
        if (date.fromisoformat(second["date"]) - date.fromisoformat(first["date"])).days == 1
    ]
    return [*([day] for day in days), *pairs]


def _leak(item: PlantedItem, days: list[dict[str, str]], reader: ChatFn) -> GateVerdict:
    for window in _windows(days):
        reply = _ask(reader, READER_MODEL, LEAK_PATH.read_text(encoding="utf-8"),
                     {"topic": item.topic, "days": window})
        flag = reply.get("flag")
        if not isinstance(flag, bool):
            raise GateError("the leak reply has no flag")
        if flag:
            return GateVerdict("leak", item.id, False, f"flagged from {window[0]['date']}")
    return GateVerdict("leak", item.id, True)


def _discoverability(
    life: Life, item: PlantedItem, days: list[dict[str, str]], reader: ChatFn,
) -> GateVerdict:
    reply = _ask(reader, READER_MODEL, DISCOVER_PATH.read_text(encoding="utf-8"),
                 {"topic": item.topic, "days": days})
    found, cited = reply.get("found"), reply.get("days")
    if not isinstance(found, bool) or not isinstance(cited, list):
        raise GateError("the discoverability reply has no verdict")
    origin = _evidence_days(life, item.origin_ids)
    if origin is None:
        return GateVerdict("discoverability", item.id, False, "origin evidence is not in the life")
    if not found:
        return GateVerdict("discoverability", item.id, False, "the reader found nothing")
    if not origin <= {str(day) for day in cited}:
        return GateVerdict("discoverability", item.id, False, "the reader missed an origin day")
    return GateVerdict("discoverability", item.id, True)


def _evidence_days(life: Life, ids: Sequence[str]) -> set[str] | None:
    by_id = {evidence.id: evidence.day.isoformat() for evidence in life.evidence}
    if any(evidence_id not in by_id for evidence_id in ids):
        return None
    return {by_id[evidence_id] for evidence_id in ids}


def _audit(life: Life, problems: ProblemFile, auditor: ChatFn) -> GateVerdict:
    reply = _ask(auditor, AUDITOR_MODEL, AUDIT_PROMPT, {
        "problem_file": [_sealed(life, item) for item in problems.items],
        "days": _days(life.sessions),
    })
    ok, issues = reply.get("ok"), reply.get("issues", [])
    if not isinstance(ok, bool) or not isinstance(issues, list):
        raise GateError("the audit reply has no verdict")
    return GateVerdict("script_audit", None, ok, None if ok else f"{len(issues)} issue(s) found")


def _sealed(life: Life, item: PlantedItem) -> dict[str, Any]:
    by_id = {evidence.id: evidence for evidence in life.evidence}

    def facts(ids: Sequence[str]) -> list[dict[str, str]]:
        return [{"date": by_id[i].day.isoformat(), "text": by_id[i].text} for i in ids if i in by_id]

    return {
        "item": item.id, "family": item.family, "topic": item.topic,
        "surface_date": item.surface_day.isoformat(), "origin": facts(item.origin_ids),
        "disambiguating": facts(item.disambiguating_ids), "resolution": item.resolution,
    }


def _ask(chat: ChatFn, model: str, system: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    return json_object(chat(model, messages, None, TIMEOUT_S, temperature=TEMPERATURE).text)


def json_object(text: str) -> dict[str, Any]:
    """The JSON object in a model's reply, allowing one code fence around it."""
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rsplit("```", 1)[0]
    try:
        value = json.loads(body)
    except ValueError:
        raise GateError("the reply is not JSON") from None
    if not isinstance(value, dict):
        raise GateError("the reply is not a JSON object")
    return value
