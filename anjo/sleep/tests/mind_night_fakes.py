# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""Fakes for MIND in the night (Task 17): a thinker that reads the rendered workspace and works one
question through notice, hypotheses, a search, one assessment and a conclusion; a research port that
stores what it finds; and ``MindNight`` over a night ``World``'s files. No model, no network."""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from anjo.sleep.evidence.jsonl import NO_GUARD, PLAIN
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.mind.inquiry.log import JsonlWorkspaceLog
from anjo.sleep.mind.inquiry.ops import op_to_json
from anjo.sleep.night.deps import MindCall
from anjo.sleep.night.meter import SpendMeter, WonderMeter
from anjo.sleep.night.mind import MindModels, MindNight, SparkWrap
from anjo.sleep.provenance import GLIMMER
from anjo.sleep.tests.inquiry_fakes import FakeResolver, FakeScreen, FakeWorld
from anjo.sleep.tests.night_fakes import World
from anjo.sleep.tests.sleep_fakes import WONDER
from anjo.sleep.types import Evidence

STATEMENT = "The plan lacks a rehearsal before the date it is due"
RESEARCH_URL = "https://example.org/rehearsal-study"
_ITEM = re.compile(r"^- \[([^\]]+)\]")


def _section(prompt: str, heading: str) -> list[str]:
    lines = prompt.split("\n")
    start = lines.index(heading) + 1
    out = []
    for line in lines[start:]:
        if line.startswith("## ") or line.startswith("Reply with"):
            break
        out.append(line)
    return out


@dataclass
class LifeThinker:
    """Opens ``notices`` questions on ``targets`` (a gap, raised by the day's first new item), then
    per cycle: HYPOTHESIZE, INVESTIGATE with ``tool`` (one ``queries`` entry each time), ASSESS the
    first unassessed hit for the leader, and CONCLUDE a resolution citing every item shown. ``hook``
    runs on each cycle prompt (a delete, say). Every reply costs ``tokens``."""

    targets: tuple[str, ...]
    tool: it.Tool = "evidence_search"
    queries: list[str] = field(default_factory=lambda: ["sorting line pilot"])
    notices: int = 1
    tokens: int = 50
    hook: Callable[[int], None] | None = None
    prompts: list[str] = field(default_factory=list)

    def propose(self, rendered: str, max_tokens: int) -> it.ModelReply:
        self.prompts.append(rendered)
        text = self._notice(rendered) if "NOTICE operations." in rendered else self._cycle(rendered)
        return it.ModelReply(text=text, tokens=min(self.tokens, max_tokens), model=GLIMMER)

    def _notice(self, prompt: str) -> str:
        new = [m.group(1) for line in _section(prompt, "## Today's new evidence") if (m := _ITEM.match(line))]
        if self.notices <= 0 or not new:
            return "[]"
        self.notices -= 1
        op = it.NoticeOp(kind="gap", subject="the plan's missing rehearsal step", target_ids=self.targets,
                         origin_evidence=(new[0],))
        return json.dumps([op_to_json(op)])

    def _cycle(self, prompt: str) -> str:
        if self.hook is not None:
            self.hook(len(self.prompts))
        qid = _section(prompt, "## Your question")[0].split(" ", 1)[0]
        held = [line[2:].split(":", 1) for line in _section(prompt, "## Hypotheses (credences are heuristic)")
                if line.startswith("- ")]
        if not held:
            alternatives = (it.Alternative("No rehearsal is planned before the date", 0.6),
                            it.Alternative("A rehearsal is planned but unmentioned", 0.2))
            return json.dumps(op_to_json(it.HypothesizeOp(qid, alternatives, 0.2)))
        ids = [h for h, _ in held]
        credences = {h: float(rest.split()[0]) for h, rest in held}
        evidence = _section(prompt, "## Evidence")
        shown = [m.group(1) for line in evidence if (m := _ITEM.match(line))]
        leader = max(ids, key=lambda h: credences[h])
        if credences[leader] >= it.RESOLVE_MIN:
            op: it.Op = it.ConcludeOp(qid, leader, "resolution", STATEMENT, tuple(shown))
        elif hit := next((m.group(1) for line in evidence if "not yet assessed" in line and (m := _ITEM.match(line))), None):
            first = next(h for h in ids if not h.endswith(".else"))
            op = it.AssessOp(qid, hit, likelihoods={h: 4.0 if h == first else 0.25 for h in ids})
        else:
            op = it.InvestigateOp(qid, self.tool, self.queries.pop(0) if len(self.queries) > 1 else self.queries[0])
        return json.dumps(op_to_json(op))


def _question(prompt: str) -> str:
    return _section(prompt, "## Your question")[0].split(" ", 1)[0]


@dataclass
class Suspects(LifeThinker):
    """A ``LifeThinker`` that reports a suspicion where it would resolve, so the question stays live."""

    def _cycle(self, prompt: str) -> str:
        reply = json.loads(super()._cycle(prompt))
        return json.dumps({**reply, "kind": "suspicion"} if reply["op"] == "CONCLUDE" else reply)


@dataclass
class TwoQuestions(LifeThinker):
    """Opens two questions on the same targets, works the first to its close, then fails (a reply
    the night can't use) on the second's first cycle."""

    first: str | None = None

    def _notice(self, prompt: str) -> str:
        ops = json.loads(super()._notice(prompt))
        return json.dumps([*ops, {**ops[0], "subject": "the plan's missing venue"}] if ops else ops)

    def _cycle(self, prompt: str) -> str:
        self.first = self.first or _question(prompt)
        if _question(prompt) != self.first:
            raise ValueError("a reply the night can't use")
        return super()._cycle(prompt)


@dataclass
class Mumbles(LifeThinker):
    """Opens its question, then answers every cycle with something no operation parses from."""

    def _cycle(self, prompt: str) -> str:
        return "I'm not sure yet."


@dataclass
class Retracts:
    """Withdraws the question it is asked about."""

    prompts: list[str] = field(default_factory=list)

    def propose(self, rendered: str, max_tokens: int) -> it.ModelReply:
        self.prompts.append(rendered)
        op = it.RetractOp(_question(rendered), "The rehearsal is booked after all", ())
        return it.ModelReply(text=json.dumps(op_to_json(op)), tokens=10, model=GLIMMER)


def mind_call(world: World, uid: str, night: date, as_of: datetime) -> MindCall:
    """A step's call outside a night's run: no Wonder, no paid action, never cancelled."""
    return MindCall(user_id=uid, night=night, as_of=as_of, at=as_of, store=world.evidence(uid), targets=world.targets(uid),
                    spend=SpendMeter(spark_in_per_m=0.2, spark_out_per_m=0.6), wonder=WonderMeter(0, 0.0),
                    actions_left=0, job=f"test:{uid}", cancelled=lambda: False)


@dataclass
class StoreResearch:
    """``research`` that finds one verified source, stores it as hosted homework (Wonder wrote it)
    and debits one Wonder call; every query it was asked is kept."""

    call: MindCall
    queries: list[str] = field(default_factory=list)

    def research(self, question_id: str, query: str) -> tuple[Evidence, ...]:
        self.queries.append(query)
        self.call.wonder.require()
        self.call.wonder.debit(0.05)
        item = Evidence(
            id=f"wr-{question_id}-{len(self.queries)}", person_id=self.call.user_id, day=self.call.night,
            kind="homework", text="Teams that rehearse a demo twice the day before ship it on time.",
            source=RESEARCH_URL, held_out=False, created_at=self.call.at - timedelta(minutes=10), model=WONDER,
            ancestry=frozenset({WONDER}),
        )
        self.call.store.add([item])
        return (item,)


def workspace(world: World, uid: str) -> JsonlWorkspaceLog:
    return JsonlWorkspaceLog(world.root / uid / "workspace.jsonl", uid, codec=PLAIN, guard=NO_GUARD)


@dataclass
class Researches:
    """Builds a ``StoreResearch`` per step, and keeps them."""

    made: list[StoreResearch] = field(default_factory=list)

    def __call__(self, call: MindCall) -> StoreResearch:
        self.made.append(StoreResearch(call))
        return self.made[-1]

    @property
    def queries(self) -> list[str]:
        return [q for r in self.made for q in r.queries]


def mind_night(
    world: World, thinker: Any, *, screen: Any = None, research: Callable[[MindCall], Any] | None = None, **kw: Any,
) -> MindNight:
    """``MindNight`` over ``world``'s files with fake models; ``screen`` may take the step's Spark
    wrapper (``screen(spark)``) to build a real screen on a fake chat."""

    def models(user_id: str, spark: SparkWrap) -> MindModels:
        built = screen(spark) if callable(screen) else screen or FakeScreen()
        return MindModels(thinker=thinker, world=FakeWorld(), screen=built, resolver=FakeResolver(), model=GLIMMER)

    return MindNight(models=models, log=lambda uid: workspace(world, uid), prices=world.deps().prices,
                     research=research, timer=world.tick, **kw)
