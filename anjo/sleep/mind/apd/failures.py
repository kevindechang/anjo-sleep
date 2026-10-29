"""Why a generation attempt failed, with none of any model's text (MIND's data).

An attempt fails at a rule (``generate.GenerationFailed``: the writer's day, the extractor's
reply, a planted fact, a pair), at a gate (a verdict that didn't pass), or on a call that still
fails after its tries or a reader's reply that can't be read (an error out of the attempt), or is
cut short by a kill (``killed``). Each
becomes one ``AttemptFailure``: the step, the day (1 to ``LIFE_DAYS``) and the item (its number in
the life) where known, the error's class or the gate's verdict, the code's own reason, and for a
reply that couldn't be read the field it lacked, the rule that refused it (``generators.Refused``),
its length, its answer's length and its finish reason. The reasons are
the code's own strings and the dates in them; nothing a model wrote is kept, and never an item's
topic. ``gen_batch`` saves each to ``failed.jsonl``, keeps it in the life's call log and prints its
``line``.

Pure.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Final

from anjo.sleep.mind.apd.calls import Call
from anjo.sleep.mind.apd.generate import GenerationFailed
from anjo.sleep.mind.apd.types import GateVerdict, Life

STEPS: Final = {"write": "writer", "extract": "extractor", "leak": "leak reader", "discover": "discoverability reader",
                "identify": "auditor (identify)", "audit": "auditor (script audit)"}
_DATE: Final = re.compile(r"\d{4}-\d{2}-\d{2}")
_ITEM: Final = re.compile(r"-i(\d+)$")


@dataclass(frozen=True)
class AttemptFailure:
    attempt: int  # 0-based, as the holds and the call log count
    step: str
    error: str  # the error's class, or ``verdict`` for a gate that didn't pass
    reason: str  # the code's own words
    day: int | None = None
    item: int | None = None
    field: str | None = None
    rule: str | None = None  # the rule that refused the reply (``generators.Refused``)
    reply_chars: int | None = None
    answer_chars: int | None = None  # what of the reply was read as its answer
    finish_reason: str | None = None
    rate_limits: int = 0  # the attempt's tries answered 429
    waits_s: tuple[float, ...] = ()  # the waits they were tried again after
    killed: bool = False  # cut short by a kill (``backend.Halted``), not failed
    missing: int | None = None  # a day's beats its ``placed`` left out (``generate.GenerationFailed``)
    extra: int | None = None  # the keys its ``placed`` added that are no beat

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> AttemptFailure:
        return cls(int(raw["attempt"]), str(raw["step"]), str(raw["error"]), str(raw["reason"]), _int(raw.get("day")),
                   _int(raw.get("item")), _str(raw.get("field")), _str(raw.get("rule")), _int(raw.get("reply_chars")),
                   _int(raw.get("answer_chars")), _str(raw.get("finish_reason")), int(raw.get("rate_limits") or 0),
                   tuple(float(wait) for wait in raw.get("waits_s") or ()), bool(raw.get("killed", False)),
                   _int(raw.get("missing")), _int(raw.get("extra")))


def _int(found: object) -> int | None:
    return None if found is None else int(str(found))


def _str(found: object) -> str | None:
    return None if found is None else str(found)


def of_rule(attempt: int, failed: GenerationFailed) -> AttemptFailure:
    return AttemptFailure(attempt, failed.step or "rule", type(failed).__name__, str(failed), failed.day, failed.item,
                          failed.field, failed.rule, failed.reply_chars, failed.answer_chars, failed.finish_reason,
                          missing=failed.missing, extra=failed.extra)


def of_verdicts(attempt: int, life: Life, verdicts: Sequence[GateVerdict]) -> tuple[AttemptFailure, ...]:
    """Each verdict that didn't pass, by gate, item and (a leak's) first flagged day."""
    found = []
    for verdict in verdicts:
        if verdict.passed:
            continue
        number = None if verdict.item_id is None else _ITEM.search(verdict.item_id)
        dated = _DATE.search(verdict.reason or "")
        day = None if dated is None else (date.fromisoformat(dated.group()) - life.days[0]).days + 1
        found.append(AttemptFailure(attempt, verdict.gate, "verdict", verdict.reason or "did not pass", day=day,
                                    item=None if number is None else int(number.group(1))))
    return tuple(found)


def of_error(attempt: int, error: Exception, last: Call | None) -> AttemptFailure:
    """A call that still failed after its tries, or a reply that couldn't be read: at the step of
    the attempt's last call, with that reply's length and finish reason when it came back."""
    step = "writer" if last is None else STEPS.get(last.kind, last.kind)
    return AttemptFailure(attempt, step, type(error).__name__, str(error) or type(error).__name__,
                          day=None if last is None or last.kind != "write" else last.days,
                          item=None if last is None else last.item,
                          reply_chars=None if last is None or last.failed else last.reply_chars,
                          finish_reason=None if last is None or last.failed else last.finish)


def recorded(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """``failed.jsonl``'s failed attempts, one printed line each."""
    return [line(str(raw.get("name")), AttemptFailure.from_json(raw)) for raw in records
            if str(raw.get("what", "")).endswith("-attempt")]


def line(name: str, failure: AttemptFailure) -> str:
    """One printed line: where the attempt failed and why."""
    where = [f"day {failure.day}" if failure.day is not None else "",
             f"item {failure.item}" if failure.item is not None else "",
             f"field {failure.field}" if failure.field is not None else ""]
    answered = "" if failure.answer_chars is None else f" (answer {failure.answer_chars})"
    reply = "" if failure.reply_chars is None else (
        f"; reply {failure.reply_chars} chars{answered}, finish {failure.finish_reason or 'unknown'}")
    rule = "" if failure.rule is None else f" [rule {failure.rule}]"
    waited = ", ".join(f"{wait:g}" for wait in failure.waits_s) or "nothing"
    limits = "" if not failure.rate_limits else f"; {failure.rate_limits} x HTTP 429, waited {waited} s"
    limits += "; killed" if failure.killed else ""
    at = ", ".join(part for part in where if part)
    return (f"FAILED attempt {failure.attempt + 1} of {name} at the {failure.step}"
            f"{' (' + at + ')' if at else ''}: {failure.error}: {failure.reason}{rule}{reply}{limits}")
