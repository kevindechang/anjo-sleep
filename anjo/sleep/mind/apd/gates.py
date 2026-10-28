"""The APD generation gates (Task 17), run before any arm and frozen.

Each gate reads pre-surface text only: the days before an item's surface day, never the day the
person raises it, and never the item's resolution or template (``PlantedItem.topic`` is the
neutral name the readers get).

1. **Leak** (``leak_v1.md``): no single day and no two adjacent days lets the reader, told the
   item's family, see the whole item. The reader checks both what the arms see (each window's
   extracted evidence and target labels) and the sessions they came from.
2. **Discoverability** (``discover_v1.md``): a reader given the whole pre-surface prefix and the
   topic finds the item: it cites every origin day and its statement, judged by the auditor
   (``identify_v1.md``) against the sealed item, identifies the planted item.
3. **Script audit** (``audit_v1.md``): the auditor checks the whole script against the sealed
   problem file, which holds the scheduled template facts as well as the extracted evidence.

Claude's read of 4 lives is manual; ``ClaudeAudit`` is its cohort-level record and
``release_cohort`` refuses to hand lives to the arms without it.

A life passes only when every verdict passed. A reply that isn't the JSON asked for raises
``GateError``: a reader that failed to read is no verdict. No gate looks at an arm.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.context.brief import ChatFn
from anjo.sleep.mind.apd.core_families import template as find_template
from anjo.sleep.mind.apd.families import Template
from anjo.sleep.mind.apd.retry import REPLY_MAX_TOKENS, reply_timeout_s
from anjo.sleep.mind.apd.types import GateVerdict, Life, PlantedItem, ProblemFile, Session
from anjo.sleep.mind.inquiry._check import check_text

LEAK_PATH: Final = Path(__file__).parent / "prompts" / "leak_v1.md"
DISCOVER_PATH: Final = Path(__file__).parent / "prompts" / "discover_v1.md"
READER_MODEL: Final = SPARK_OPEN
AUDITOR_MODEL: Final = "muse-spark"
TIMEOUT_S: Final = reply_timeout_s(REPLY_MAX_TOKENS)  # the wait a capped reply needs (``retry``)
TEMPERATURE: Final = 0.0
AUDIT_PATH: Final = Path(__file__).parent / "prompts" / "audit_v1.md"
IDENTIFY_PATH: Final = Path(__file__).parent / "prompts" / "identify_v1.md"
CLAUDE_AUDIT_LIVES: Final = 4  # Claude reads this many lives of a cohort, or all of a smaller one


class GateError(ValueError):
    """A reader's or the auditor's reply isn't the JSON asked for."""


def pre_surface(life: Life, item: PlantedItem) -> tuple[Session, ...]:
    """The sessions before ``item``'s surface day: the only text any gate reads for it."""
    return tuple(session for session in life.sessions if session.day < item.surface_day)


def check(
    life: Life, problems: ProblemFile, *, reader: ChatFn, auditor: ChatFn, model: str = READER_MODEL,
) -> tuple[GateVerdict, ...]:
    """Leak and discoverability for every item, then the script audit for the life. The readers
    ask for ``model``: Spark Open's id on the experiment's provider."""
    if problems.life_id != life.id:
        raise ValueError("the problem file belongs to another life")
    verdicts: list[GateVerdict] = []
    for item in problems.items:
        days = _days(pre_surface(life, item))
        views = {"sessions": days, "evidence": _arm_days(life, item, days)}
        verdicts.append(_leak(item, views, reader, model))
        verdicts.append(_discoverability(life, item, days, reader, auditor, model))
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


def _arm_days(life: Life, item: PlantedItem, days: list[dict[str, str]]) -> list[dict[str, str]]:
    """What the arms see of each pre-surface day: its extracted evidence and the labels of the
    targets mentioned on it."""
    shown = {day["date"] for day in days}
    lines: dict[str, list[str]] = {}
    for evidence in life.evidence:
        if evidence.day.isoformat() in shown:
            lines.setdefault(evidence.day.isoformat(), []).append(evidence.text)
    for day in life.days:
        for target in life.targets[day]:
            if target.last_mentioned == day and day.isoformat() in shown:
                due = "" if target.check_after is None else f", due {target.check_after.date().isoformat()}"
                lines.setdefault(day.isoformat(), []).append(f"Target ({target.kind}): {target.label}{due}")
    return [{"date": day, "text": "\n".join(lines[day])} for day in sorted(lines)]


def _leak(item: PlantedItem, views: Mapping[str, list[dict[str, str]]], reader: ChatFn,
          model: str = READER_MODEL) -> GateVerdict:
    for view, days in views.items():
        for window in _windows(days):
            reply = _ask(reader, model, LEAK_PATH.read_text(encoding="utf-8"), {
                "topic": item.topic, "family": item.family, "view": view, "days": window})
            flag = reply.get("flag")
            if not isinstance(flag, bool):
                raise GateError("the leak reply has no flag")
            if flag:
                return GateVerdict("leak", item.id, False, f"{view} flagged from {window[0]['date']}")
    return GateVerdict("leak", item.id, True)


def _discoverability(
    life: Life, item: PlantedItem, days: list[dict[str, str]], reader: ChatFn, auditor: ChatFn, model: str,
) -> GateVerdict:
    reply = _ask(reader, model, DISCOVER_PATH.read_text(encoding="utf-8"),
                 {"topic": item.topic, "days": days})
    found, cited, statement = reply.get("found"), reply.get("days"), reply.get("statement", "")
    if not isinstance(found, bool) or not isinstance(cited, list) or not isinstance(statement, str):
        raise GateError("the discoverability reply has no verdict")
    origin = _evidence_days(life, item.origin_ids)
    if origin is None:
        return GateVerdict("discoverability", item.id, False, "origin evidence is not in the life")
    if not found or not statement.strip():
        return GateVerdict("discoverability", item.id, False, "the reader found nothing")
    if not origin <= {str(day) for day in cited}:
        return GateVerdict("discoverability", item.id, False, "the reader missed an origin day")
    judged = _ask(auditor, AUDITOR_MODEL, IDENTIFY_PATH.read_text(encoding="utf-8"),
                  {"statement": statement.strip(), "item": _sealed(life, item)})
    match = judged.get("match")
    if not isinstance(match, bool):
        raise GateError("the identify reply has no verdict")
    if not match:
        return GateVerdict("discoverability", item.id, False, "the statement isn't the planted item")
    return GateVerdict("discoverability", item.id, True)


def _evidence_days(life: Life, ids: Sequence[str]) -> set[str] | None:
    by_id = {evidence.id: evidence.day.isoformat() for evidence in life.evidence}
    if any(evidence_id not in by_id for evidence_id in ids):
        return None
    return {by_id[evidence_id] for evidence_id in ids}


def _audit(life: Life, problems: ProblemFile, auditor: ChatFn) -> GateVerdict:
    reply = _ask(auditor, AUDITOR_MODEL, AUDIT_PATH.read_text(encoding="utf-8"), {
        "problem_file": [_sealed(life, item) for item in problems.items],
        "days": _days(life.sessions),
    })
    ok, issues = reply.get("ok"), reply.get("issues")
    if not isinstance(ok, bool) or not isinstance(issues, list):
        raise GateError("the audit reply has no verdict")
    if not all(isinstance(issue, str) and issue.strip() for issue in issues) or ok == bool(issues):
        raise GateError("the audit reply contradicts itself")
    return GateVerdict("script_audit", None, ok, None if ok else f"{len(issues)} issue(s) found")


def _template(item: PlantedItem) -> Template | None:
    return find_template(item.template)


def _sealed(life: Life, item: PlantedItem) -> dict[str, Any]:
    """The sealed item: the scheduled template facts, and the evidence extracted for them."""
    by_id = {evidence.id: evidence for evidence in life.evidence}
    template = _template(item)

    def facts(ids: Sequence[str]) -> list[dict[str, str]]:
        return [{"date": by_id[i].day.isoformat(), "text": by_id[i].text} for i in ids if i in by_id]

    return {
        "item": item.id, "family": item.family, "topic": item.topic,
        "surface_date": item.surface_day.isoformat(),
        "origin_facts": [] if template is None else list(template.origins),
        "disambiguating_fact": None if template is None else template.disambiguating,
        "origin": facts(item.origin_ids), "disambiguating": facts(item.disambiguating_ids),
        "resolution": item.resolution,
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


# ── Claude's read of the cohort ───────────────────────────────────────────────


class CohortNotReleased(RuntimeError):
    """A cohort can't go to the arms yet."""


@dataclass(frozen=True)
class ClaudeAudit:
    """The registered record of Claude's manual read of a cohort: which lives, who read them, when,
    whether they were sound, and notes. Nothing here calls Claude."""

    life_ids: tuple[str, ...]
    reviewer: str
    day: date
    ok: bool
    notes: str

    def __post_init__(self) -> None:
        if isinstance(self.life_ids, str):
            raise TypeError("life_ids must be a sequence of ids, not a string")
        object.__setattr__(self, "life_ids", tuple(self.life_ids))
        for life_id in self.life_ids:
            check_text("life_ids", life_id)
        if not self.life_ids or len(set(self.life_ids)) != len(self.life_ids):
            raise ValueError("the audit names its lives, each once")
        check_text("reviewer", self.reviewer)
        if not isinstance(self.ok, bool) or not isinstance(self.notes, str):
            raise TypeError("ok is a bool and notes a string")

    def to_json(self) -> str:
        return json.dumps({"life_ids": list(self.life_ids), "reviewer": self.reviewer,
                           "date": self.day.isoformat(), "ok": self.ok, "notes": self.notes}, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> ClaudeAudit:
        try:
            raw = json.loads(text)
            return cls(tuple(raw["life_ids"]), raw["reviewer"], date.fromisoformat(raw["date"]), raw["ok"],
                       raw["notes"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("the Claude audit record is malformed") from None


def read_claude_audit(path: Path) -> ClaudeAudit | None:
    """The record at ``path``, or ``None`` when none has been registered."""
    return ClaudeAudit.from_json(path.read_text(encoding="utf-8")) if path.is_file() else None


def release_cohort(lives: Sequence[Life], audit: ClaudeAudit | None) -> tuple[Life, ...]:
    """The cohort, for the arms to run, once Claude's audit of it is registered and sound: it
    names ``CLAUDE_AUDIT_LIVES`` of the cohort's lives (all of a smaller cohort)."""
    if audit is None:
        raise CohortNotReleased("no Claude audit is registered for the cohort")
    if not audit.ok:
        raise CohortNotReleased("the Claude audit did not pass")
    ids = {life.id for life in lives}
    if not set(audit.life_ids) <= ids:
        raise CohortNotReleased("the Claude audit names a life outside the cohort")
    if len(audit.life_ids) < min(CLAUDE_AUDIT_LIVES, len(ids)):
        raise CohortNotReleased(f"the Claude audit reads fewer than {CLAUDE_AUDIT_LIVES} lives")
    return tuple(lives)
