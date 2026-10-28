"""Where a generated life's calls go, as they were made (MIND's data).

Each try of each generation call (``backend.CountedChat``) is logged with its kind (the prompt it
was sent: ``write``, ``extract``, ``leak``, ``discover``, ``identify``, ``audit``), the attempt,
the item it read for (numbered in the order the attempt first read it: never its topic, which is
the sealed file's), how many days it read, its tokens in and out as the provider counted them,
what it was priced at, and its reply's length and finish reason (never its text).
``DataDir.save_calls`` keeps a life's log, every attempt included, and a skipped, killed or stopped
life's too, one per run (``failed_log_id``); ``breakdown`` prints it per gate, per days read and
per item, which is what the plan's read shapes are checked against. The log also keeps why each
failed attempt failed (``failures.AttemptFailure``).

Pure.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Final

KINDS: Final = MappingProxyType({
    "# life_v1": "write", "# extract_open_v1": "extract", "# leak_v1": "leak", "# discover_v1": "discover",
    "# identify_v1": "identify", "# audit_v1": "audit",
})
READS: Final = ("leak", "discover")  # the reader's calls
OTHER: Final = "other"


@dataclass(frozen=True)
class Call:
    """One try of one call."""

    kind: str
    attempt: int
    item: int | None  # the item it read for, by the order the attempt first read it
    days: int | None  # the days it read (a read, the audit) or the day it wrote (a write)
    tokens_in: int
    tokens_out: int
    usd: float
    failed: bool = False
    reply_chars: int | None = None  # the reply's length, never its text
    finish: str | None = None  # its finish reason, as the provider gave it
    rate_limited: bool = False  # a failed try answered 429
    wait_s: float | None = None  # the wait before the call was tried again, after this try failed

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> Call:
        chars, finish, wait = raw.get("reply_chars"), raw.get("finish"), raw.get("wait_s")
        return cls(str(raw["kind"]), int(raw["attempt"]), None if raw["item"] is None else int(raw["item"]),
                   None if raw["days"] is None else int(raw["days"]), int(raw["tokens_in"]), int(raw["tokens_out"]),
                   float(raw["usd"]), bool(raw.get("failed", False)), None if chars is None else int(chars),
                   None if finish is None else str(finish), bool(raw.get("rate_limited", False)),
                   None if wait is None else float(wait))


def kind_of(messages: Sequence[Mapping[str, Any]]) -> str:
    """The call's kind, by its system prompt's first line."""
    first = str(messages[0].get("content", "")).split("\n", 1)[0].strip() if messages else ""
    return KINDS.get(first, OTHER)


def _payload(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    try:
        found = json.loads(str(messages[-1].get("content", "")))
    except ValueError:
        return {}
    return found if isinstance(found, dict) else {}


@dataclass
class CallLog:
    """One life's calls, every attempt's, as they are made (``backend.CountedChat`` adds them)."""

    attempt: int = 0
    calls: list[Call] = field(default_factory=list)
    failures: list[Any] = field(default_factory=list)  # ``failures.AttemptFailure``, one per failed attempt
    on_failure: Callable[[Any], None] | None = None  # each failure as it is kept (``gen_batch.recorder``)
    _items: dict[tuple[int, str], int] = field(default_factory=dict)

    def add(self, messages: Sequence[Mapping[str, Any]], tokens_in: int, tokens_out: int, usd: float, *,
            failed: bool = False, reply_chars: int | None = None, finish: str | None = None,
            rate_limited: bool = False, wait_s: float | None = None) -> None:
        kind, payload = kind_of(messages), _payload(messages)
        self.calls.append(Call(kind, self.attempt, self._item(kind, payload), _days(kind, payload), tokens_in,
                               tokens_out, usd, failed, reply_chars, finish, rate_limited, wait_s))

    def rate_limits(self) -> tuple[int, tuple[float, ...]]:
        """The attempt's tries answered 429, and the waits they were tried again after (0.1 s)."""
        mine = [call for call in self.calls if call.attempt == self.attempt and call.rate_limited]
        return len(mine), tuple(round(call.wait_s, 1) for call in mine if call.wait_s is not None)

    def last(self) -> Call | None:
        """The attempt's last call, if it made one."""
        mine = [call for call in self.calls if call.attempt == self.attempt]
        return mine[-1] if mine else None

    def _item(self, kind: str, payload: Mapping[str, Any]) -> int | None:
        sealed = payload.get("item")
        topic = payload.get("topic") if kind in READS else sealed.get("topic") if isinstance(sealed, dict) else None
        if not isinstance(topic, str):
            return None
        key = (self.attempt, topic)
        if key not in self._items:
            self._items[key] = 1 + sum(1 for at, _ in self._items if at == self.attempt)
        return self._items[key]


def _days(kind: str, payload: Mapping[str, Any]) -> int | None:
    if kind == "write" and isinstance(payload.get("day"), int):
        return int(payload["day"])
    days = payload.get("days")
    return len(days) if isinstance(days, list) else None


# ── The breakdown ─────────────────────────────────────────────────────────────


def _sum(calls: Iterable[Call]) -> tuple[int, int, int, float]:
    found = list(calls)
    return len(found), sum(c.tokens_in for c in found), sum(c.tokens_out for c in found), sum(c.usd for c in found)


def _line(label: str, calls: Iterable[Call]) -> str:
    found = list(calls)
    count, tokens_in, tokens_out, usd = _sum(found)
    mean_in, mean_out = (tokens_in / count, tokens_out / count) if count else (0.0, 0.0)
    worst = [call for call in found if call.failed and call.usd > 0]  # failed tries, priced at their worst case
    held = "" if not worst else (f" (of it ${sum(c.usd for c in worst):.4f} for {len(worst)} failed tries priced at "
                                 f"their worst case)")
    return (f"  {label}: {count} call(s), {tokens_in:,} in / {tokens_out:,} out tokens (each {mean_in:,.0f} / "
            f"{mean_out:,.0f}), ${usd:.4f}{held}")


def breakdown(logs: Mapping[str, Sequence[Call]]) -> list[str]:
    """Where the calls of every life in ``logs`` (an id to its calls) went: per kind, the reads
    per days read and per item, and per life."""
    every = [call for calls in logs.values() for call in calls]
    lines = [f"{len(logs)} li{'fe' if len(logs) == 1 else 'ves'} logged, {len(every)} call(s) in all "
             f"(failed tries {sum(c.failed for c in every)}):", "Per gate (kind):"]
    for kind in sorted({c.kind for c in every}):
        lines.append(_line(kind, (c for c in every if c.kind == kind)))
    reads = [c for c in every if c.kind in READS]
    lines.append("Reads per days read:")
    for kind in READS:
        for days in sorted({c.days for c in reads if c.kind == kind and c.days is not None}):
            lines.append(_line(f"{kind}, {days} day(s)", (c for c in reads if c.kind == kind and c.days == days)))
    lines.append("Reads per item (attempt, item):")
    for life_id, calls in sorted(logs.items()):
        for attempt, item in sorted({(c.attempt, c.item) for c in calls if c.kind in READS and c.item is not None}):
            lines.append(_line(f"{life_id} attempt {attempt + 1} item {item}",
                               (c for c in calls if c.kind in READS and (c.attempt, c.item) == (attempt, item))))
    lines.append("Per life:")
    for life_id, calls in sorted(logs.items()):
        lines.append(_line(life_id, calls))
    return lines


def failed_log_id(life_id: str, taken: Collection[str] = ()) -> str:
    """The id a life's log is kept under when it wasn't saved: its first attempt's life id and its
    run (``-1``, ``-2``...), the first not ``taken``, so a rerun that fails again keeps both logs."""
    run = 1
    while f"{life_id}.failed-{run}" in taken:
        run += 1
    return f"{life_id}.failed-{run}"
