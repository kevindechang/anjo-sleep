"""Repairing a core split's failed attempt (MIND's data; ``--repair``).

A core split's attempt that fails only a leak or the script audit need not be thrown away whole:
the days the failure names are written again, the rest kept.

- **What is repaired** (``fixes``): a leak verdict names the first day of the window it flagged
  (``gates.flagged``), and that day is written again, told which view gave the item away; a failed
  script audit names its days by date in its issues, and each is written again with the issues
  that name it. A discoverability verdict, an issue that names no day of the life, or more than
  ``REPAIR_DAYS`` days in one round is left to a fresh attempt, as before.
- **How a day is written again** (``rewritten``): the writer gets ``life_v1.md``, the same beats
  and ``repair_v1.md``, with the days before it as they now stand, the day as it was, why, and the
  days after it, which it keeps to. The day is extracted again on the targets held before it, its
  messages' sources tagged by the round so no evidence id is reused, and every day's targets are
  folded again; the life and its sealed file are built again from the whole draft, so a planted
  fact that left no evidence still fails the attempt.
- **What is read again** (``Memo``): the life is gated again in full, but a request sent before in
  the same attempt is answered from its reply without a call. Only the windows whose text changed,
  the ones never read (past a leak's first flag), and the discoverability, identify and audit
  calls whose input changed are paid for.
- **Held like an attempt:** each round is held before it starts, after the run's ceiling is
  checked (``gen_batch``), for its writes and a full read again at worst (``round_calls``), and
  priced and logged as an attempt is, its calls marked with the round.
- **At most ``ROUNDS`` rounds an attempt.** A round that fails again for a reason a round can fix
  starts the next; after the last, or on a reason it can't fix, the attempt fails as it did
  before, recorded with its last check's verdicts (or the rule a rewritten day broke).

APD's own splits are regenerated whole, as registered; ``--repair`` is for the core splits only.
"""
from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Final

from anjo.sleep.context.brief import ChatFn
from anjo.sleep.mind.apd import gates, generate
from anjo.sleep.mind.apd.backend import GenCalls, life_calls
from anjo.sleep.mind.apd.generate import Beat, Draft, ItemPlan, WrittenDay
from anjo.sleep.mind.apd.types import GateVerdict, Life, ProblemFile
from anjo.sleep.model_api import ChatResult

ROUNDS: Final = 2  # the repair rounds one attempt may take
REPAIR_DAYS: Final = 3  # the most days one round writes again; a failure naming more is a fresh attempt
REPAIRABLE: Final = frozenset({"leak", "script_audit"})
_SEEN: Final = {"evidence": "the notes taken from", "sessions": "the messages of"}
LEAK_REASON: Final = ('A reader shown {seen} this day, alone or with the next day, could already see the whole '
                      '{family} about "{topic}" before the person raises it.')
AUDIT_REASON: Final = "The script audit found: {issue}"
_DATE: Final = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass
class Memo:
    """A gate chat that answers a request it already sent in this attempt from its reply, without
    a call; every new reply kept in ``replies``, which the attempt's holds share."""

    inner: ChatFn
    replies: dict[str, ChatResult]

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
                 timeout_s: float, *, temperature: float | None = None) -> ChatResult:
        key = json.dumps([str(model), messages, tools, temperature], ensure_ascii=False, sort_keys=True)
        found = self.replies.get(key)
        if found is None:
            found = self.inner(model, messages, tools, timeout_s, temperature=temperature)
            self.replies[key] = found
        return found


@dataclass(frozen=True)
class Tried:
    """One gated draft of an attempt: the draft and its plan, the life and sealed file it makes,
    the gates' verdicts and, when the script audit failed, the auditor's issues."""

    draft: Draft
    plans: tuple[ItemPlan, ...]
    life: Life
    problems: ProblemFile
    verdicts: tuple[GateVerdict, ...]
    issues: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return gates.passed(self.verdicts)

    def fixes(self) -> dict[int, tuple[str, ...]] | None:
        return fixes(self.life, self.problems, self.verdicts, self.issues)

    def why(self, attempt: int, round_: int) -> dict[str, Any]:
        """Why it failed, as ``--keep-failed`` keeps it beside its life."""
        return {"attempt": attempt, "round": round_, "verdicts": [dataclasses.asdict(v) for v in self.verdicts],
                "issues": list(self.issues)}


def fixes(life: Life, problems: ProblemFile, verdicts: Sequence[GateVerdict],
          issues: Sequence[str]) -> dict[int, tuple[str, ...]] | None:
    """The days to write again (by index, from 0), each with why; ``None`` when nothing failed or
    a rewrite can't mend what did."""
    failed = [verdict for verdict in verdicts if not verdict.passed]
    if not failed or any(verdict.gate not in REPAIRABLE for verdict in failed):
        return None
    items = {item.id: item for item in problems.items}
    found: dict[int, list[str]] = {}
    for verdict in (verdict for verdict in failed if verdict.gate == "leak"):
        where, item = gates.flagged(verdict), items.get(verdict.item_id or "")
        if where is None or item is None or where[1] not in life.days:
            return None
        found.setdefault(life.days.index(where[1]), []).append(
            LEAK_REASON.format(seen=_SEEN[where[0]], family=item.family, topic=item.topic))
    if any(verdict.gate == "script_audit" for verdict in failed):
        for issue in issues:
            days = sorted({life.days.index(day) for day in _dates(issue) if day in life.days})
            if not days:
                return None
            for index in days:
                found.setdefault(index, []).append(AUDIT_REASON.format(issue=issue))
    if not found or len(found) > REPAIR_DAYS:
        return None
    return {index: tuple(reasons) for index, reasons in sorted(found.items())}


def _dates(text: str) -> list[date]:
    found = []
    for match in _DATE.findall(text):
        try:
            found.append(date.fromisoformat(match))
        except ValueError:
            continue
    return found


def round_calls(prevalence: int, days: int) -> GenCalls:
    """The most one repair round calls: a writer and an extractor call for each day it writes
    again, and the gates read again in full, as an attempt's are (at worst every window changed)."""
    full = life_calls(prevalence)
    return GenCalls(2 * days, full.reads, full.muse)


def rewritten(tried: Tried, found: Mapping[int, Sequence[str]], round_: int,
              chat: ChatFn) -> tuple[Draft, tuple[ItemPlan, ...]]:
    """``tried``'s draft with each day ``found`` names written again, in order, told why."""
    beats = generate.beats_of(tried.plans)
    draft = tried.draft
    for index, reasons in sorted(found.items()):
        draft = _rewrite(draft, index, beats.get(index, []), " ".join(reasons), chat, f"r{round_}")
    return draft, tried.plans


def _rewrite(draft: Draft, index: int, beats: Sequence[Beat], reason: str, chat: ChatFn, branch: str) -> Draft:
    old = draft.written[index]
    asked = {"reason": reason, "before": list(old.texts),
             "later": [{"day": done.index + 1, "messages": list(done.texts)} for done in draft.written[index + 1:]]}
    new = refolded(draft, draft.written[:index]).write(chat, index, old.day, beats, branch=branch, repair=asked)
    return refolded(draft, (*draft.written[:index], new, *draft.written[index + 1:]))


def refolded(draft: Draft, days: Sequence[WrittenDay]) -> Draft:
    """``draft``'s life holding ``days``, its targets folded again from the first."""
    found = Draft(draft.life_id, draft.field, draft.seed, draft.model)
    for done in days:
        found = found.add(done)
    return found
