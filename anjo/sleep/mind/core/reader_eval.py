"""The thought channel's evaluation: does Glimmer, reading MIND's state as soft tokens, raise the
right problem at the right time, and only then? Pure; the live run is ``tools.thought_channel_eval``.

**Cases.** On core-test lives, one per person-day after the first: the day's first message, and the
core's state just before it (``life_cases``). On the counterfactual pairs, the **transplant**
(``pair_cases``): the same last-day message under the state of the history where the commitment
failed (she should raise it) and of the one where it was kept (she shouldn't). Nothing but the
state differs: the message, the system turn and the sampling seed are the same.

**Arms** (``arm_prompts``), one prompt each per case, the same seed in every arm:

- ``state``: the reader (Glimmer and the reader LoRA) with the case's own state as K soft tokens;
- ``zero``: the same, with the empty reading (the channel's K positions, nothing in them);
- ``shuffled``: the same, with another life's state (a seeded pick from a different life);
- ``text``: Glimmer as today, the same facts pasted as text (the events the routed slots cite,
  ``notes_for``) in the system turn, no soft tokens and no reader adapter.

If ``zero`` or ``shuffled`` does as well as ``state``, the channel carries nothing.

**Scores.** A deterministic pre-score (``prescore``: the reply names at least two of the problem's
distinctive words, ``keywords``) and a blind judge (``judge_prompt``) asked twice, the problems in
opposite orders, counted only when both answers agree (``Verdict``). Labels are read only to score,
after every reply is written. False raises are counted per 100 person-days; the transplant's flip
rate is the share of pairs where the failed history's reply raises the commitment and the kept one's
doesn't (``flips``).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Final, Literal

from anjo.sleep.mind.core import reader_data
from anjo.sleep.mind.core.projector import Reading

ARMS: Final = ("state", "zero", "shuffled", "text")
Arm = Literal["state", "zero", "shuffled", "text"]
EMPTY: Final = reader_data.EMPTY_KEY
CaseKind = Literal["life", "failed", "kept"]
Expect = Literal["raise", "wait", "unknown"]
NOTES_HEADER: Final = "\n\nWhat they told you on earlier days:\n"
MIN_HITS: Final = 2
_WORD: Final = re.compile(r"[a-z0-9][a-z0-9'-]*")
STOPWORDS: Final = frozenset(
    "about after again also back been before being both but came can could does done each even every from "
    "going good have here into just know like made make many more most much need only other over said same "
    "should some still such take than that their them then there these they thing this those through today "
    "very want well were what when where which while will with would your yours you're i'm it's we're".split())


@dataclass(frozen=True)
class Case:
    case_id: str
    life_id: str  # the person: a core-test life, or a pair (both its histories are one person)
    kind: CaseKind
    day: date
    event_id: str
    message: str
    reading_key: str
    expect: Expect = "unknown"
    pair_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "life_id": self.life_id, "kind": self.kind, "day": self.day.isoformat(),
                "event_id": self.event_id, "message": self.message, "reading_key": self.reading_key,
                "expect": self.expect, "pair_id": self.pair_id}

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> Case:
        return cls(str(obj["case_id"]), str(obj["life_id"]), obj["kind"], date.fromisoformat(str(obj["day"])),
                   str(obj["event_id"]), str(obj["message"]), str(obj["reading_key"]), obj.get("expect", "unknown"),
                   obj.get("pair_id"))


def _events(records: Sequence[Mapping[str, Any]]) -> list[reader_data.Event]:
    return [reader_data.Event.from_record(r) for r in sorted(records, key=lambda e: (str(e["t"]), int(e.get("seq", 0))))]


def life_cases(life_id: str, records: Sequence[Mapping[str, Any]]) -> tuple[Case, ...]:
    """One case per person-day after the first: that day's first message."""
    events = _events(records)
    first_day = events[0].day if events else None
    seen: set[date] = set()
    cases = []
    for event in events:
        if event.day == first_day or event.day in seen:
            continue
        seen.add(event.day)
        cases.append(Case(f"{life_id}:{event.event_id}", life_id, "life", event.day, event.event_id, event.text,
                          f"{life_id}:{event.event_id}"))
    return tuple(cases)


def pair_cases(record: Mapping[str, Any], failed: Sequence[Mapping[str, Any]],
               kept: Sequence[Mapping[str, Any]]) -> tuple[Case, Case]:
    """The transplant: the last day's first message, under each history's state just before it."""
    pair_id, final = str(record["pair_id"]), str(record["final_ids"][0])
    texts = {side: {e.event_id: e for e in _events(rows)} for side, rows in (("failed", failed), ("kept", kept))}
    a, b = texts["failed"].get(final), texts["kept"].get(final)
    if a is None or b is None or (a.text, a.day) != (b.text, b.day):
        raise ValueError(f"pair {pair_id}'s histories don't share their last day's message word for word")
    decisions = record.get("decisions", {"failed": "speak", "kept": "wait"})
    expect: dict[str, Expect] = {side: "raise" if decisions[side] == "speak" else "wait" for side in ("failed", "kept")}
    sides: tuple[CaseKind, CaseKind] = ("failed", "kept")
    failed_case, kept_case = (Case(f"{pair_id}.{side}", pair_id, side, a.day, final, a.text, f"{pair_id}.{side}:{final}",
                                   expect[side], pair_id) for side in sides)
    return failed_case, kept_case


@dataclass(frozen=True)
class ArmPrompt:
    case_id: str
    arm: Arm
    messages: list[dict[str, str]]
    reading: str | None  # the reading key whose soft tokens go in, or None for none
    seed: int
    reader: bool  # served with the reader adapter (every arm but ``text``)


def case_seed(seed: int, case: Case) -> int:
    """The sampling seed: the person and the message, so a pair's two histories share it."""
    return int(hashlib.sha256(f"{seed}:{case.life_id}:{case.event_id}".encode()).hexdigest()[:8], 16)


def _shuffled(cases: Sequence[Case], seed: int) -> dict[str, str]:
    """Each case's stand-in state: a pick among the cases of other people, seeded by the person and
    the message, so a pair's two histories get the same stand-in."""
    found = {}
    for case in cases:
        others = [c for c in cases if c.life_id != case.life_id]
        if not others:
            raise ValueError("the shuffled arm needs cases from at least two people")
        found[case.case_id] = random.Random(f"{seed}:{case.life_id}:{case.event_id}").choice(others).reading_key
    return found


def arm_prompts(cases: Sequence[Case], notes: Mapping[str, Sequence[str]], *, seed: int) -> tuple[ArmPrompt, ...]:
    shuffled = _shuffled(cases, seed)
    prompts = []
    for case in cases:
        plain = reader_data.prompt_messages(case.message)
        pasted = "".join(f"- {line}\n" for line in notes.get(case.case_id, ()))
        text = [{"role": "system", "content": plain[0]["content"] + (NOTES_HEADER + pasted if pasted else "")}, plain[-1]]
        s = case_seed(seed, case)
        prompts += [ArmPrompt(case.case_id, "state", plain, case.reading_key, s, True),
                    ArmPrompt(case.case_id, "zero", plain, EMPTY, s, True),
                    ArmPrompt(case.case_id, "shuffled", plain, shuffled[case.case_id], s, True),
                    ArmPrompt(case.case_id, "text", text, None, s, False)]
    return tuple(prompts)


def notes_for(reading: Reading, events: Mapping[str, reader_data.Event]) -> tuple[str, ...]:
    """The text arm's memory: the events the reading's routed slots cite, once each, in time order."""
    cited = {eid for row in reading.cites for eid in row if eid in events}
    return tuple(f"{events[e].day.isoformat()}: {events[e].text}" for e in sorted(cited, key=lambda e: (events[e].day, e)))


# ── Scoring ───────────────────────────────────────────────────────────────────


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.casefold()) if (len(w) >= 4 or any(c.isdigit() for c in w))
            and w not in STOPWORDS}


def keywords(texts: Sequence[str], others: Sequence[str]) -> frozenset[str]:
    """A problem's distinctive words: in its events' texts, and in at most one of the life's other texts."""
    seen = [_words(t) for t in others]
    return frozenset(w for w in set().union(*(_words(t) for t in texts)) if sum(w in s for s in seen) <= 1)


def prescore(reply: str, words: frozenset[str]) -> bool:
    """Whether ``reply`` names the problem: at least ``MIN_HITS`` of its distinctive words (all, if fewer)."""
    return bool(words) and len(_words(reply) & words) >= min(MIN_HITS, len(words))


def flips(pairs: Sequence[tuple[Case, Case]], replies: Mapping[str, str], *,
          keywords: Mapping[str, frozenset[str]]) -> tuple[int, int]:
    """(pairs whose failed history's reply raises the commitment and kept one's doesn't, pairs scored)."""
    scored = [(a, b) for a, b in pairs if a.case_id in replies and b.case_id in replies]
    flipped = sum(prescore(replies[a.case_id], keywords[str(a.pair_id)])
                  and not prescore(replies[b.case_id], keywords[str(b.pair_id)]) for a, b in scored)
    return flipped, len(scored)


JUDGE: Final = """You read a reply an AI assistant wrote to a person's message, and a numbered list of
problems in that person's plans. Say which listed problem the reply raises: warns about, points out, or
suggests acting on. A reply that only answers the message raises none.

Answer with one JSON object and nothing else: {"raises": N, "other": B}, where N is the number of the
problem it raises (0 for none) and B is true when it warns about a risk or problem not in the list.

Message: {message}

Reply: {reply}

Problems:
{problems}"""


def judge_prompt(message: str, reply: str, problems: Sequence[str]) -> str:
    listed = "\n".join(f"{n}. {p}" for n, p in enumerate(problems, 1)) or "(none)"
    return JUDGE.replace("{message}", message).replace("{reply}", reply).replace("{problems}", listed)


def parse_judge(text: str, count: int) -> tuple[int, bool]:
    try:
        data = json.loads(text.strip().removeprefix("```json").removeprefix("```").removesuffix("```"))
        raised, other = int(data["raises"]), bool(data.get("other", False))
    except (ValueError, KeyError, TypeError, AttributeError):
        raise ValueError("the judge's answer isn't the JSON asked for") from None
    if not 0 <= raised <= count:
        raise ValueError("the judge named a problem that isn't listed")
    return raised, other


@dataclass(frozen=True)
class Verdict:
    """The judge's two answers, the list in opposite orders: the problem raised (its index in the
    list as given, or None), a risk not in the list, and whether the two answers agree."""

    raised: int | None
    other: bool
    agree: bool


def judged(ask: Callable[[str], str], message: str, reply: str, problems: Sequence[str]) -> Verdict:
    """Ask twice, ``problems`` forward then reversed; an answer that can't be read counts as a disagreement."""
    n = len(problems)
    answers = []
    for order in (list(range(n)), list(reversed(range(n)))):
        try:
            raised, other = parse_judge(ask(judge_prompt(message, reply, [problems[i] for i in order])), n)
        except ValueError:
            return Verdict(None, False, False)
        answers.append((None if raised == 0 else order[raised - 1], other))
    (first, other_a), (second, other_b) = answers
    agree = first == second and other_a == other_b
    return Verdict(first if agree else None, other_a and other_b, agree)
