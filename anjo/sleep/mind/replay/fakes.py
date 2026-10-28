"""Offline stand-ins for replay's models, for ``--fake`` and the tests (Task 17b).

They answer with fixed, deterministic text and spend nothing. Their scores mean nothing; they
exist so the whole replay path (sample, resolve, score, both datasets, the training plan and the
arm comparison) runs with no keys and no cost. They share the real ports' types, so the code that
runs is the code that runs live, on the real kernel and loop.

- ``fake_life``: a small synthetic APD life, the film student's scene 4 in miniature: an 85 mm
  medium shot (day 0), a 3 m kitchen and a rental shop's word that the lens needs four metres
  (day 1), a courtyard window (day 2) and the courtyard locked (day 3), every item written by
  "Spark Open" and stamped within its own day.
- ``FakeGlimmer``: a thinker that steers by the rendered workspace, as Glimmer is asked to. At
  temperature 0 it always searches the same words, which never find the rental shop's item; at
  temperature above 0 it picks its words, priors and buckets with noise drawn from its ``seed`` and
  the prompt, so a branch is reproducible, branches with different seeds differ, and some find
  the item that settles the question.
- ``FakeLabelChat``: Spark Open labelling a start state from the later evidence: the courtyard
  locked settles the kitchen hypothesis; anything else is undetermined.
- ``FakeJudge``: the APD judge, matching a finding by its cites.

The fakes report the real model names, so the provenance rules run as they do live.
"""
from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Final

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.mind.apd.score import Ruling
from anjo.sleep.mind.apd.types import ArmModels, FindingReport, Life, PlantedItem, ProblemFile, Session, Split
from anjo.sleep.mind.inquiry.types import (
    ASSESS_BUCKETS,
    ModelReply,
    ResolverReply,
    Screened,
    ScreenPurpose,
)
from anjo.sleep.mind.replay.sample import ReplayModels
from anjo.sleep.mind.replay.types import BRANCH_TEMPERATURE, STOPPED, CallEnd
from anjo.sleep.model_api import ChatResult
from anjo.sleep.provenance import GLIMMER
from anjo.sleep.types import Evidence, Target

FAKE_DAY0: Final = date(2026, 9, 7)  # every APD life's first day (``generate.LIFE_START``)
SCENE: Final = "t-scene4"
DAYS: Final[tuple[tuple[tuple[str, str | None], ...], ...]] = (
    (("For scene 4 I want the compressed 85 mm medium shot on full frame.", SCENE),
     ("I bought gaffer tape and two lamps.", None)),
    (("Scene 4 is shot in my 3 m kitchen, handheld.", SCENE),
     ("The rental shop says that lens needs about four metres for a medium shot.", None)),
    (("Maybe I shoot scene 4 through the courtyard window.", SCENE),),
    (("The courtyard is locked on Thursday, so no window shot for scene 4.", SCENE),),
)
KITCHEN: Final = "The kitchen leaves no room for the 85 mm medium shot"
WINDOW: Final = "The 85 mm shot is taken from the courtyard window"
QUERIES: Final = ("scene 4 kitchen", "courtyard window scene 4", "85 mm lens shot")
WORLD_QUESTION: Final = "How much room does an 85 mm medium shot need?"
_EVIDENCE: Final = re.compile(r"^- \[(?P<id>[^\]]+)\] \S+?(?:, (?P<label>[^:]+))?: (?P<text>.*)$")
_HYPOTHESIS: Final = re.compile(r"^- (?P<id>\S+): (?P<credence>[0-9.]+)  (?P<statement>.*)$")
_TARGET: Final = re.compile(r"^- (?P<id>\S+) \(")
_LETTER: Final = re.compile(r"^(?P<letter>[A-Z])\. (?P<statement>.*)$")


# ── A life ────────────────────────────────────────────────────────────────────


def _noon(day: date) -> datetime:
    return datetime.combine(day, time(12, 0), tzinfo=timezone.utc)


def fake_life(seed: int, *, split: Split = "development") -> Life:
    """Scene 4 over ``len(DAYS)`` virtual days, the life's own ids throughout."""
    life_id = f"{split}-fake-film-{seed}"
    days = tuple(FAKE_DAY0 + timedelta(days=n) for n in range(len(DAYS)))
    evidence = tuple(
        Evidence(
            id=f"{life_id}:d{n}:e{i}", person_id=life_id, day=day, kind="lesson", text=text,
            source=f"session:{life_id}:d{n}", held_out=False, created_at=_noon(day) + timedelta(minutes=i),
            model=SPARK_OPEN, ancestry=frozenset({SPARK_OPEN}), target_id=target,
        )
        for n, (day, items) in enumerate(zip(days, DAYS, strict=True)) for i, (text, target) in enumerate(items)
    )
    shoot = Target(
        id=SCENE, person_id=life_id, kind="commitment", label="Shoot scene 4 on Thursday",
        check_after=_noon(days[0] + timedelta(days=5)), status="open", revision=1, acted_revision=0,
        confidence=None, last_mentioned=days[0],
    )
    sessions = tuple(Session(day, " ".join(text for text, _ in items), SPARK_OPEN) for day, items in zip(days, DAYS))
    return Life(
        id=life_id, field="indie film", seed=seed, split=split, days=days, sessions=sessions, evidence=evidence,
        targets={day: (shoot,) for day in days},
    )


def fake_problems(life: Life) -> ProblemFile:
    """The life's one planted contradiction, settled by the locked courtyard."""
    ids = [item.id for item in life.evidence]
    item = PlantedItem(
        id=f"{life.id}-i1", family="contradiction", template="fake-lens-room", topic="scene 4's shoot",
        surface_day=life.days[-1], origin_ids=(ids[0], ids[2]), disambiguating_ids=(ids[-1],),
        resolution="The kitchen is too small for the 85 mm medium shot, and the courtyard is locked.",
    )
    return ProblemFile(life_id=life.id, items=(item,))


# ── Reading a rendered prompt ─────────────────────────────────────────────────


def section(text: str, heading: str) -> list[str]:
    """The lines under ``heading`` (a ``## `` line) up to the next heading."""
    lines = text.splitlines()
    if heading not in lines:
        return []
    out: list[str] = []
    for line in lines[lines.index(heading) + 1:]:
        if line.startswith("## "):
            break
        out.append(line)
    return out


def _evidence(lines: Sequence[str]) -> list[tuple[str, str | None, str]]:
    found = (_EVIDENCE.match(line) for line in lines)
    return [(m["id"], m["label"], m["text"]) for m in found if m]


def _question(text: str) -> tuple[str, list[tuple[str, float, str]], list[str]]:
    """The focal question's id, its hypotheses and the items that raised it."""
    head = section(text, "## Your question")
    question = head[0].split(" ", 1)[0]
    raised = head[1].rsplit("Raised by: ", 1)[1].rstrip(".").split(", ") if len(head) > 1 else []
    found = (_HYPOTHESIS.match(line) for line in section(text, "## Hypotheses (credences are heuristic)"))
    return question, [(m["id"], float(m["credence"]), m["statement"]) for m in found if m], raised


# ── The thinker ───────────────────────────────────────────────────────────────


def _likelihoods(text: str, statements: Sequence[str]) -> list[float]:
    """What an item says about each hypothesis: the lock, the kitchen and the four metres back the
    kitchen; the window backs the window."""
    words = text.casefold()
    out = []
    for statement in statements:
        kitchen, window = statement == KITCHEN, statement == WINDOW
        if "locked" in words:
            out.append(4.0 if kitchen else 0.25 if window else 1.0)
        elif "window" in words:
            out.append(0.5 if kitchen else 4.0 if window else 1.0)
        elif "kitchen" in words or "metres" in words:
            out.append(2.0 if kitchen else 0.5 if window else 1.0)
        else:
            out.append(1.0)
    return out


@dataclass(frozen=True)
class FakeGlimmer:
    """The thinker: a NOTICE once the lens and the kitchen are both in view; then HYPOTHESIZE,
    ASSESS each search hit, CONCLUDE once the leader reaches 0.8, and otherwise search again, so a
    night with nothing new stalls (K5) and leaves the question live. Every reply is drawn from the
    seed, the temperature and the prompt alone. ``calls`` records how each call ended, as the
    served thinker does: a reply longer than the call's cap is cut there (``"length"``)."""

    seed: int = 0
    temperature: float = 0.0
    calls: list[CallEnd] = field(default_factory=list, compare=False)

    def propose(self, rendered: str, max_tokens: int) -> ModelReply:
        if max_tokens <= 0:
            self.calls.append(CallEnd(None))
            return ModelReply(text="", tokens=0, model=GLIMMER)
        rng = random.Random(f"{self.seed}:{self.temperature}:{rendered}")
        if "## Today's new evidence" in rendered:
            text = json.dumps(self._notice(rendered))
        else:
            text = json.dumps(self._cycle(rendered, rng))
        tokens = 150 if self.temperature == 0 else 120 + rng.randrange(240)
        self.calls.append(CallEnd("length" if tokens > max_tokens else STOPPED))
        return ModelReply(text=text, tokens=min(tokens, max_tokens), model=GLIMMER)

    def _notice(self, text: str) -> list[dict[str, Any]]:
        if section(text, "## Questions you hold") != ["(none)"]:
            return []
        items = _evidence(section(text, "## Earlier evidence you hold") + section(text, "## Today's new evidence"))
        lens = [item for item, _, words in items if "85 mm" in words]
        room = [item for item, _, words in items if "kitchen" in words]
        targets = [m["id"] for m in (_TARGET.match(line) for line in section(text, "## Open targets")) if m]
        if not (lens and room and targets):
            return []
        return [{"op": "NOTICE", "kind": "contradiction", "subject": "Scene 4's 85 mm medium shot against a 3 m kitchen",
                 "target_ids": targets[:1], "origin_evidence": [lens[0], room[0]]}]

    def _noise(self, rng: random.Random) -> float:
        return rng.uniform(-1.0, 1.0) * 0.15 * self.temperature

    def _cycle(self, text: str, rng: random.Random) -> dict[str, Any]:
        question, held, raised = _question(text)
        if not held:
            first = 0.45 + self._noise(rng)
            return {"op": "HYPOTHESIZE", "question_id": question, "catch_all": 0.2, "hypotheses": [
                {"statement": KITCHEN, "credence": round(first, 3)},
                {"statement": WINDOW, "credence": round(0.8 - first, 3)}]}
        evidence = _evidence(section(text, "## Evidence"))
        waiting = [(item, words) for item, label, words in evidence if label == "search hit, not yet assessed"]
        ids, statements = [h for h, _, _ in held], [s for _, _, s in held]
        if waiting:
            values = _likelihoods(waiting[0][1], statements)
            noisy = [rng.choice(ASSESS_BUCKETS) if rng.random() < self.temperature / 2 else v for v in values]
            return {"op": "ASSESS", "question_id": question, "evidence_id": waiting[0][0],
                    "likelihoods": dict(zip(ids, noisy, strict=True))}
        leader = max(held, key=lambda h: h[1])
        if leader[1] >= 0.8:
            return {"op": "CONCLUDE", "question_id": question, "hypothesis_id": leader[0], "kind": "resolution",
                    "statement": leader[2], "cites": raised}
        if rng.random() < self.temperature / 4:
            return {"op": "INVESTIGATE", "question_id": question, "tool": "world_query", "query": WORLD_QUESTION}
        query = QUERIES[0] if self.temperature == 0 else rng.choice(QUERIES)
        return {"op": "INVESTIGATE", "question_id": question, "tool": "evidence_search", "query": query}


# ── The other ports ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FakeWorld:
    """``world_query`` on the thinker's model (``model`` names another to test provenance)."""

    model: str = GLIMMER

    def ask(self, text: str, max_tokens: int) -> ModelReply:
        return ModelReply(text="About four metres from the subject.", tokens=min(30, max_tokens), model=self.model)


@dataclass(frozen=True)
class FakeScreen:
    def screen(self, text: str, purpose: ScreenPurpose, max_tokens: int) -> Screened:
        return Screened(allowed=True, tokens=0)


@dataclass(frozen=True)
class FakeResolver:
    """The blind prediction resolver; the fake thinker never predicts, so it is never asked."""

    def resolve(self, claim: str, evidence: Sequence[Evidence]) -> ResolverReply:
        return ResolverReply(verdict="undetermined", tokens=0)


def fake_models(thinker: FakeGlimmer | None = None) -> ArmModels:
    """The M arm's models: the thinker at temperature 0 unless given."""
    return ArmModels(thinker=thinker or FakeGlimmer(), world=FakeWorld(), screen=FakeScreen(), resolver=FakeResolver())


def _branch_thinker(seed: int, night: date) -> FakeGlimmer:
    return FakeGlimmer(seed, BRANCH_TEMPERATURE)


@dataclass
class FakeResolverChat:
    """Spark Open as the resolver's ``ChatFn``: always undetermined, under ``model``. It keeps
    every prompt it was sent."""

    model: str = SPARK_OPEN
    sent: list[list[dict[str, Any]]] = field(default_factory=list)

    def __call__(self, messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        self.sent.append(messages)
        return ChatResult(text='{"verdict": "undetermined"}', tool_calls=(), model=self.model,
                          input_tokens=len(str(messages[-1]["content"])) // 4, output_tokens=8, finish_reason="stop")


def fake_replay_models() -> ReplayModels:
    """The branches' models: the thinker at ``BRANCH_TEMPERATURE`` with each branch's seed."""
    return ReplayModels(thinker_for=_branch_thinker, world=FakeWorld(), screen=FakeScreen(),
                        resolver_chat=FakeResolverChat())


@dataclass
class FakeLabelChat:
    """Spark Open's labeller (a ``ChatFn``): the locked courtyard settles the kitchen hypothesis;
    nothing else settles anything. It keeps every prompt it was sent."""

    sent: list[list[dict[str, Any]]] = field(default_factory=list)

    def __call__(self, messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        self.sent.append(messages)
        user = str(messages[-1]["content"])
        letters = {m["statement"]: m["letter"] for m in (_LETTER.match(line) for line in user.splitlines()) if m}
        evidence = user.split("Evidence:", 1)[-1].casefold()
        holds = letters.get(KITCHEN) if "locked" in evidence and KITCHEN in letters else None
        text = json.dumps({"holds": holds or "undetermined"})
        return ChatResult(text=text, tool_calls=(), model=SPARK_OPEN, input_tokens=len(user) // 4,
                          output_tokens=min(12, max_tokens), finish_reason="stop")


@dataclass(frozen=True)
class FakeJudge:
    """The APD judge: a finding is the item's problem when it cites one of the item's origin ids,
    and it holds when it names the kitchen."""

    def rule(self, report: FindingReport, evidence: Sequence[Evidence], item: PlantedItem | None) -> Ruling:
        same = item is not None and bool(set(report.cites) & set(item.origin_ids))
        return Ruling(same_problem=same, holds="kitchen" in report.conclusion.casefold())
