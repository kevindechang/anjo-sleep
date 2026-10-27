# mypy: disallow-untyped-decorators=False
"""Fakes and builders for MIND's inquiry tests (Task 17): no network, no model, no clock."""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from anjo.sleep.mind.inquiry import kernel
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.mind.inquiry.log import MemoryWorkspaceLog
from anjo.sleep.mind.inquiry.ports import InquiryPorts
from anjo.sleep.types import Evidence, EvidenceKind, Target

PERSON = "life-film"
DAY0 = date(2026, 10, 1)
THINKER_MODEL = "glimmer-fake"


def day(n: int) -> date:
    return DAY0 + timedelta(days=n)


def noon(n: int) -> datetime:
    return datetime.combine(day(n), time(12, 0), tzinfo=timezone.utc)


def night(n: int) -> datetime:
    """The night clock of day ``n``: every step that night is stamped with it."""
    return datetime.combine(day(n), time(23, 0), tzinfo=timezone.utc)


def ev(
    id: str, text: str, *, n: int, kind: EvidenceKind = "lesson", target: str | None = None,
    held_out: bool = False, model: str | None = "spark-open-fake", ancestry: frozenset[str] | None = frozenset(),
    source: str | None = None,
) -> Evidence:
    return Evidence(
        id=id, person_id=PERSON, day=day(n), kind=kind, text=text, source=source or f"message:{id}",
        held_out=held_out, created_at=noon(n), model=model, ancestry=ancestry, target_id=target,
    )


def ref(item: Evidence) -> it.EvidenceRef:
    return it.EvidenceRef(id=item.id, day=item.day, created_at=item.created_at)


def commitment(id: str, label: str, *, due: datetime | None, n: int = 0, due_noted: bool = False) -> Target:
    return Target(
        id=id, person_id=PERSON, kind="commitment", label=label, check_after=due, status="open", revision=1,
        acted_revision=0, confidence=None, last_mentioned=day(n), due_noted=due_noted,
    )


def thread(id: str, label: str, *, n: int = 0) -> Target:
    return Target(
        id=id, person_id=PERSON, kind="thread", label=label, check_after=None, status="open", revision=1,
        acted_revision=0, confidence=0.5, last_mentioned=day(n),
    )


# ── Applying steps by hand (the kernel tests) ─────────────────────────────────


class Ws:
    """Applies steps one at a time, as the loop logs them, and keeps the events."""

    def __init__(self, rules: it.KernelRules = it.FULL_RULES) -> None:
        self.ws = it.Workspace()
        self.rules = rules
        self.events: list[it.WorkspaceEvent] = []

    def step(self, entry: Any, *, code: it.RejectCode | None = None, cycle: int | None = None,
             focus: str | None = None, tokens: int = 0, at: datetime | None = None) -> it.Applied | it.Rejected:
        step = it.Step(self.ws.seq, at or night(0), entry, code, cycle, focus, tokens)
        result = kernel.apply(self.ws, step, rules=self.rules)
        if isinstance(result, it.Applied):
            self.ws = result.ws
            self.events.append(result.event)
        return result

    def ok(self, entry: Any, **changes: Any) -> it.Workspace:
        result = self.step(entry, **changes)
        assert isinstance(result, it.Applied), result
        return result.ws

    def code(self, entry: Any) -> it.RejectCode:
        result = self.step(entry)
        assert isinstance(result, it.Rejected), "expected a rejection"
        return result.code

    def start(self, n: int = 0, *, rules: it.KernelRules | None = None, wipe: bool = False) -> None:
        rules = rules or self.rules
        self.rules = rules
        self.ok(it.RunStart(day(n), night(n), "inquiry", rules, "priority", None, it.Budget(), wipe), at=night(n))

    def arrive(self, *items: Evidence, at: datetime | None = None) -> None:
        self.ok(it.Retrieved(None, "new_evidence", "", tuple(ref(i) for i in items)), at=at)

    def found(self, question_id: str, *items: Evidence, tool: it.RetrieveTool = "evidence_search") -> None:
        self.ok(it.Retrieved(question_id, tool, "q", tuple(ref(i) for i in items)))

    def credences(self, question_id: str) -> dict[str, float]:
        return {h.id: h.credence for h in self.ws.hypotheses_of(question_id)}

    def ids(self, question_id: str) -> list[str]:
        return [h.id for h in self.ws.hypotheses_of(question_id)]


def notice_op(origin: Sequence[str], *, kind: it.QuestionKind = "contradiction", targets: Sequence[str] = ("t-scene4",),
              subject: str = "Scene 4's long-lens look against the kitchen it is shot in") -> it.NoticeOp:
    return it.NoticeOp(kind=kind, subject=subject, target_ids=tuple(targets), origin_evidence=tuple(origin))


def hypothesize(question_id: str, *priors: float, catch_all: float = 0.2) -> it.HypothesizeOp:
    return it.HypothesizeOp(
        question_id, tuple(it.Alternative(f"alternative {n} about the plan", p) for n, p in enumerate(priors, 1)),
        catch_all,
    )


# ── Port fakes ────────────────────────────────────────────────────────────────


class MemorySearch:
    """``EvidenceSearch`` over a list: lexical overlap, time-masked, held-out items never leave."""

    def __init__(self, items: Sequence[Evidence]) -> None:
        self.items = list(items)
        self.queries: list[tuple[str, datetime]] = []

    def search(self, query: str, as_of: datetime, k: int) -> list[Evidence]:
        self.queries.append((query, as_of))
        words = set(re.findall(r"\w+", query.casefold()))
        scored = [
            (len(words & set(re.findall(r"\w+", item.text.casefold()))), item) for item in self.items
            if not item.held_out and item.created_at <= as_of
        ]
        ranked = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], -s[1].created_at.timestamp(), s[1].id))
        return [item for _, item in ranked[:k]]

    def get(self, ids: Sequence[str], as_of: datetime) -> tuple[Evidence, ...]:
        held = {i.id: i for i in self.items if not i.held_out and i.created_at <= as_of}
        return tuple(held[i] for i in ids if i in held)


class Kill(BaseException):
    """A process kill: raised past every handler."""


@dataclass
class ScriptedThinker:
    """Replies from a script, in order. A reply may be a callable of the rendered prompt. A call
    that raises (``kill_at``) consumes nothing, so a resumed run gets the same reply."""

    script: list[Any]
    tokens: int = 50
    prompts: list[str] = field(default_factory=list)
    kill_at: int | None = None

    def propose(self, rendered: str, max_tokens: int) -> it.ModelReply:
        if self.kill_at is not None and len(self.prompts) == self.kill_at:
            self.kill_at = None
            raise Kill()
        self.prompts.append(rendered)
        reply = self.script.pop(0) if self.script else '{"op": "PARK", "question_id": "none"}'
        text = reply(rendered) if callable(reply) else reply if isinstance(reply, str) else json.dumps(reply)
        return it.ModelReply(text=text, tokens=min(self.tokens, max_tokens), model=THINKER_MODEL)


@dataclass
class FakeScreen:
    refuse: Callable[[str, it.ScreenPurpose], bool] = lambda text, purpose: False
    tokens: int = 5
    seen: list[tuple[str, it.ScreenPurpose]] = field(default_factory=list)

    def screen(self, text: str, purpose: it.ScreenPurpose, max_tokens: int) -> it.Screened:
        self.seen.append((text, purpose))
        return it.Screened(allowed=not self.refuse(text, purpose), tokens=min(self.tokens, max_tokens))


@dataclass
class FakeWorld:
    answer: str = "An 85 mm lens needs about 3 m of distance for a medium shot."
    tokens: int = 30
    asked: list[str] = field(default_factory=list)

    def ask(self, text: str, max_tokens: int) -> it.ModelReply:
        self.asked.append(text)
        return it.ModelReply(text=self.answer, tokens=min(self.tokens, max_tokens), model=THINKER_MODEL)


@dataclass
class FakeResolver:
    """Answers each claim with the first verdict whose key appears in any item's text."""

    verdicts: dict[str, it.Verdict] = field(default_factory=dict)
    calls: list[tuple[str, tuple[Evidence, ...]]] = field(default_factory=list)

    def resolve(self, claim: str, evidence: Sequence[Evidence]) -> it.ResolverReply:
        self.calls.append((claim, tuple(evidence)))
        for key, verdict in self.verdicts.items():
            if any(key in item.text for item in evidence):
                return it.ResolverReply(verdict, 20)
        return it.ResolverReply("undetermined", 20)


@dataclass
class FakeResearch:
    sources: tuple[Evidence, ...] = ()
    queries: list[str] = field(default_factory=list)

    def research(self, question_id: str, query: str) -> tuple[Evidence, ...]:
        self.queries.append(query)
        return self.sources


class KillingLog(MemoryWorkspaceLog):
    """Kills the process right after the event at ``kill_after`` (a step seq) is logged."""

    def __init__(self, kill_after: int | None = None) -> None:
        super().__init__()
        self.kill_after = kill_after

    def append(self, event: it.WorkspaceEvent) -> None:
        super().append(event)
        if self.kill_after is not None and event.step.seq == self.kill_after:
            self.kill_after = None
            raise Kill()


def ports(
    items: Sequence[Evidence], script: list[Any], *, screen: FakeScreen | None = None,
    resolver: FakeResolver | None = None, world: FakeWorld | None = None, research: FakeResearch | None = None,
    log: MemoryWorkspaceLog | None = None, thinker: ScriptedThinker | None = None,
) -> InquiryPorts:
    return InquiryPorts(
        thinker=thinker or ScriptedThinker(script), screen=screen or FakeScreen(), search=MemorySearch(items),
        world=world or FakeWorld(), resolver=resolver or FakeResolver(), log=log or MemoryWorkspaceLog(),
        research=research,
    )
