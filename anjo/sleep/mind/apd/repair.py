"""Repairing a core split's failed attempt (MIND's data; ``--repair``).

A core split's attempt that fails only a leak, a missed origin day or the script audit need not be
thrown away whole: the days the failure names are written again, the rest kept.

- **What is repaired** (``fixes``): a leak verdict names the window it flagged (``gates.flagged``,
  ``gates.flagged_window``: one day, or two), and its days are written again, told which view
  gave the item away; a failed script audit names its days by date in its issues, and each is
  written again with the issues that name it (an issue naming no date names the one day whose
  messages hold what it quotes, ``quoted_day``); a discoverability reader that missed an origin day
  (``gates.missed_origin_days``) has each origin day it passed over written again, told to make the
  planted fact findable without stating a conclusion. Any other discoverability verdict, an issue
  that names no day of the life and quotes none, or more than ``REPAIR_DAYS`` days in one round is
  left to a fresh attempt.
- **How a day is written again** (``rewritten``): the writer gets ``life_v1.md``, the same beats
  and ``repair_v1.md``, with the days before it as they now stand, the day as it was, why, and the
  days after it, which it keeps to. The day is extracted again on the targets held before it, its
  messages' sources tagged by the round so no evidence id is reused. Every later day is extracted
  again too, its messages unchanged, on the targets then held: a target's label is fixed when it
  is made and comes back with each mention, so a label the rewrite took away must not return on
  a later day. The life and its sealed file are built again from the whole draft, so a planted
  fact that left no evidence still fails the attempt.
- **What is read again** (``Memo``): the life is gated again in full, but a request sent before in
  the same attempt is answered from its reply without a call. Only the windows whose text changed,
  the ones never read (past a leak's first flag), and the discoverability, identify and audit
  calls whose input changed are paid for.
- **Held like an attempt:** each round is held before it starts, after the run's ceiling is
  checked (``gen_batch``), for its writes and extractions and a full read again at worst
  (``round_calls``), and priced and logged as an attempt is, its calls marked with the round.
- **What a round costs** depends on how early its first day falls and what the extractions change.
  On the fakes, a round that writes day 6 of a six-item life again costs about 63% of the
  attempt's first check (``test_apd_repair``); live, the Muse identify and audit calls weigh more.
- **At most ``ROUNDS`` rounds an attempt.** A round that fails again for a reason a round can fix
  starts the next; after the last, or on a reason it can't fix, the attempt fails as it did
  before, recorded with its last check's verdicts (or the rule a rewritten day broke).

APD's own splits are regenerated whole, as registered; ``--repair`` is for the core splits only.
"""
from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Final

from anjo.sleep.context.brief import ChatFn
from anjo.sleep.mind.apd import gates, generate
from anjo.sleep.mind.apd.backend import GenCalls, life_calls
from anjo.sleep.mind.apd.generate import Draft, ItemPlan, WrittenDay
from anjo.sleep.mind.apd.types import LIFE_DAYS, GateVerdict, Life, ProblemFile
from anjo.sleep.model_api import ChatResult

ROUNDS: Final = 2  # the repair rounds one attempt may take
REPAIR_DAYS: Final = 3  # the most days one round writes again; a failure naming more is a fresh attempt
REPAIRABLE: Final = frozenset({"leak", "script_audit"})  # and discoverability's ``gates.MISSED_ORIGIN`` alone
_SEEN: Final = {"evidence": "the notes taken from", "sessions": "the messages of"}
LEAK_REASON: Final = 'A reader shown {seen} {window} could already see the whole {family} about "{topic}" before ' \
                     "the person raises it."
AUDIT_REASON: Final = "The script audit found: {issue}"
DISCOVER_REASON: Final = 'A reader looking for the {family} about "{topic}" did not cite this day, though it rests on ' \
                         "what the person says here. Make that fact findable: say it plainly, in a message of its " \
                         "own, without stating any conclusion. Apart from this day's `plant` and `ask` beats, write " \
                         "nothing about {topic} on this day."
_DATE: Final = re.compile(r"\d{4}-\d{2}-\d{2}")
QUOTED_WORDS: Final = 3  # the fewest words a quotation in an undated issue has to name the day holding it
UNQUOTED_WORDS: Final = 6  # the fewest of an unquoted undated issue's words in a row that name the day holding them
_WORD: Final = re.compile(r"[\w']+")
_QUOTED: Final = re.compile(r'"([^"]+)"|\u201c([^\u201d]+)\u201d|\u2018([^\u2019]+)\u2019|(?<!\w)\'([^\']+)\'(?!\w)')
_ELLIPSIS: Final = re.compile(r"\.\.\.|\u2026")


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
    windows: Mapping[str, tuple[date, ...]] = dataclasses.field(default_factory=dict)  # a flagged item's window
    missed: Mapping[str, tuple[date, ...]] = dataclasses.field(default_factory=dict)  # an item's uncited origin days

    @property
    def passed(self) -> bool:
        return gates.passed(self.verdicts)

    def fixes(self) -> dict[int, tuple[str, ...]] | None:
        return fixes(self.life, self.problems, self.verdicts, self.issues, self.windows, missed=self.missed)

    def why(self, attempt: int, round_: int) -> dict[str, Any]:
        """Why it failed, as ``--keep-failed`` keeps it beside its life."""
        return {"attempt": attempt, "round": round_, "verdicts": [dataclasses.asdict(v) for v in self.verdicts],
                "issues": list(self.issues),
                "missed": {item: [day.isoformat() for day in days] for item, days in self.missed.items()}}


def fixes(life: Life, problems: ProblemFile, verdicts: Sequence[GateVerdict], issues: Sequence[str],
          windows: Mapping[str, tuple[date, ...]] | None = None, *,
          missed: Mapping[str, tuple[date, ...]] | None = None) -> dict[int, tuple[str, ...]] | None:
    """The days to write again (by index, from 0), each with why; ``None`` when nothing failed or
    a rewrite can't mend what did. ``windows`` gives a flagged item's window (by item id); one not
    given is its first day alone. ``missed`` gives the origin days an item's discoverability reader
    did not cite; an item that missed one with none given is left to a fresh attempt."""
    failed = [verdict for verdict in verdicts if not verdict.passed]
    if not failed or not all(_repairable(verdict) for verdict in failed):
        return None
    items = {item.id: item for item in problems.items}
    found: dict[int, list[str]] = {}
    for verdict in (verdict for verdict in failed if verdict.gate == "discoverability"):
        origins, item = (missed or {}).get(verdict.item_id or "", ()), items.get(verdict.item_id or "")
        if not origins or item is None or not set(origins) <= set(life.days):
            return None
        for day in origins:
            found.setdefault(life.days.index(day), []).append(DISCOVER_REASON.format(family=item.family, topic=item.topic))
    for verdict in (verdict for verdict in failed if verdict.gate == "leak"):
        where, item = gates.flagged(verdict), items.get(verdict.item_id or "")
        window = (windows or {}).get(verdict.item_id or "") or (() if where is None else (where[1],))
        if where is None or item is None or not set(window) <= set(life.days):
            return None
        said = "this day alone" if len(window) == 1 else f"{' and '.join(d.isoformat() for d in window)} together"
        for day in window:
            found.setdefault(life.days.index(day), []).append(
                LEAK_REASON.format(seen=_SEEN[where[0]], window=said, family=item.family, topic=item.topic))
    if any(verdict.gate == "script_audit" for verdict in failed):
        for issue in issues:
            days = sorted({life.days.index(day) for day in _dates(issue) if day in life.days})
            if not days and (quoted := quoted_day(life, issue)) is not None:
                days = [life.days.index(quoted)]
            if not days:
                return None
            for index in days:
                found.setdefault(index, []).append(AUDIT_REASON.format(issue=issue))
    if not found or len(found) > REPAIR_DAYS:
        return None
    return {index: tuple(reasons) for index, reasons in sorted(found.items())}


def quoted_day(life: Life, issue: str) -> date | None:
    """The day an issue naming no date quotes, case and punctuation aside. An issue quoting at
    least ``QUOTED_WORDS`` words (a quotation split at an ellipsis) names the one day whose messages
    hold its quotations; its other words are not matched, so a paraphrase can't name a day by a
    common phrase. One without such a quotation (a scare-quoted word or two is none) names the one
    day holding the longest run of its words, at least ``UNQUOTED_WORDS`` long. ``None`` when
    nothing is found, or it is on more than one day."""
    texts = _day_texts(life)
    quotes = [" ".join(words) for found in _QUOTED.finditer(issue)
              for part in _ELLIPSIS.split(next(group for group in found.groups() if group))
              if len(words := _words(part)) >= QUOTED_WORDS]
    if quotes:
        return _one({day for day, text in texts.items() for quote in quotes if f" {quote} " in text})
    words = _words(issue)
    for size in range(len(words), UNQUOTED_WORDS - 1, -1):
        runs = {" ".join(words[start:start + size]) for start in range(len(words) - size + 1)}
        days = {day for day, text in texts.items() if any(f" {run} " in text for run in runs)}
        if days:
            return _one(days)
    return None


def _day_texts(life: Life) -> dict[date, str]:
    """Each day's messages as one run of words, padded so a phrase matches only whole words."""
    said: dict[date, list[str]] = {}
    for session in life.sessions:
        said.setdefault(session.day, []).extend(_words(session.text))
    return {day: f" {' '.join(found)} " for day, found in said.items() if day in life.days}


def _one(days: set[date]) -> date | None:
    return next(iter(days)) if len(days) == 1 else None


def _words(text: str) -> list[str]:
    return _WORD.findall(text.replace("\u2019", "'").lower())


def _repairable(verdict: GateVerdict) -> bool:
    return verdict.gate in REPAIRABLE or (verdict.gate == "discoverability" and verdict.reason == gates.MISSED_ORIGIN)


def _dates(text: str) -> list[date]:
    found = []
    for match in _DATE.findall(text):
        try:
            found.append(date.fromisoformat(match))
        except ValueError:
            continue
    return found


def round_calls(prevalence: int, days: Collection[int], *, day_retries: int = 0) -> GenCalls:
    """The most one repair round writing ``days`` (indexes) again calls: a writer and an extractor
    call for each of them and ``day_retries`` more writer calls each (``generate.day_retries``), an
    extractor call for each other day after the first of them, and the gates read again in full, as
    an attempt's are (at worst every window changed)."""
    full = life_calls(prevalence)
    return GenCalls(len(days) * (1 + day_retries) + LIFE_DAYS - min(days), full.reads, full.muse)


def rewritten(tried: Tried, found: Mapping[int, Sequence[str]], round_: int,
              chat: ChatFn) -> tuple[Draft, tuple[ItemPlan, ...]]:
    """``tried``'s draft from the first day ``found`` names on: each day it names written again,
    told why, and every other later day extracted again, on the targets held before it."""
    beats, first = generate.beats_of(tried.plans), min(found)
    draft = refolded(tried.draft, tried.draft.written[:first])
    for done in tried.draft.written[first:]:
        if done.index not in found:
            draft = draft.add(draft.extracted_again(chat, done))
            continue
        asked = {"reason": " ".join(found[done.index]), "before": list(done.texts),
                 "later": [{"day": d.index + 1, "messages": list(d.texts)} for d in tried.draft.written[done.index + 1:]]}
        draft = draft.add(draft.write(chat, done.index, done.day, beats.get(done.index, []), branch=f"r{round_}",
                                      repair=asked))
    return draft, tried.plans


def refolded(draft: Draft, days: Sequence[WrittenDay]) -> Draft:
    """``draft``'s life holding ``days``, its targets folded again from the first."""
    found = Draft(draft.life_id, draft.field, draft.seed, draft.model, keep_off=draft.keep_off)
    for done in days:
        found = found.add(done)
    return found
