# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""An accelerated rig on fakes (Task 12a): the night's fake world, a simulated day that stamps every
line with a fake clock, and an extractor that reads back only the lines inside the night's window.
No model, no GPU, no network. ``python -m anjo.sleep.sim.accelerated --fake`` runs on it; with
``mind_from`` (``--mind``), MIND's inquiry runs in each night on a fake thinker (Task 17), over a
thread and a commitment due nine days after that date."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from anjo.sleep.personas.model import Persona
from anjo.sleep.provenance import GLIMMER
from anjo.sleep.sim.accelerated import Rig
from anjo.sleep.sim.day import DayRun, run_day
from anjo.sleep.night.deps import Prices
from anjo.sleep.probes.arms import ArmDeps
from anjo.sleep.probes.morning import Morning, run_morning
from anjo.sleep.probes.score import Verdict
from anjo.sleep.tests.mind_night_fakes import LifeThinker, mind_night
from anjo.sleep.tests.night_fakes import World
from anjo.sleep.tests.sleep_fakes import FakeContexts, FakeLogprobs, FakeModel, StoreSpy
from anjo.sleep.train.lora import objective_used
from anjo.sleep.train.objective import night_cfg
from anjo.sleep.types import CheckpointRef, Evidence, NightResult, Probe, Target

FAKE_SHA = "0123456789abcdef0123456789abcdef01234567"
CLOCK_START = datetime(2026, 10, 28, 8, 0, tzinfo=timezone.utc)


class FakeClock:
    """Every reading is a minute on, so each simulated day gets its own, later window."""

    def __init__(self) -> None:
        self.now = CLOCK_START

    def __call__(self) -> datetime:
        self.now += timedelta(minutes=1)
        return self.now


class Line:
    def complete(self, system: str, messages: list[dict[str, str]], *, temperature: float, seed: int) -> str:
        return f"Line {len(messages)} of seed {seed} about the pilot."


class Replies:
    def of_reply(self, session_id: str) -> frozenset[str] | None:
        return frozenset({GLIMMER})


class Verdicts:
    """A deterministic fake judge: the verdict is a hash of the two replies."""

    def verdict(self, probe: Probe, first: str, second: str) -> Verdict:
        return ("A", "B", "tie")[int(hashlib.sha256(f"{probe.id}|{first}|{second}".encode()).hexdigest(), 16) % 3]


class Adapters:
    """The serving checkpoint of each life (the night's registry), and a promoted donor for each."""

    def __init__(self, world: World) -> None:
        self.world = world

    def last_promoted(self, person_id: str) -> CheckpointRef | None:
        if person_id.startswith(DONOR_PREFIX):
            return CheckpointRef(f"{person_id}-n1", person_id, date(2026, 10, 28), f"fake://{person_id}", None)
        return self.world.registry.last_promoted(person_id)


DONOR_PREFIX = "donor-of-"
MIND_DUE_DAYS = 9  # the life's commitment is due this many days after its first night


def life_targets(user_id: str, first: date) -> tuple[Target, Target]:
    """A MIND life's thread and the commitment it bears on."""
    def target(suffix: str, kind: str, label: str, due: datetime | None) -> Target:
        return Target(id=f"{user_id}-{suffix}", person_id=user_id, kind=kind, label=label, check_after=due,  # type: ignore[arg-type]
                      status="open", revision=1, acted_revision=0, confidence=None, last_mentioned=first)
    due = datetime.combine(first + timedelta(days=MIND_DUE_DAYS), time(9), tzinfo=timezone.utc)
    return target("t1", "thread", "the pilot", None), target("c1", "commitment", "the pilot demo", due)


def sealed_probes(user_id: str) -> list[Probe]:
    kinds = ("new_work", "vocab", "tone_hold", "pushback")
    probes = [Probe(f"{user_id}-{kind}-{i}", user_id, kind, f"A {kind} question {i}?", None, None, ())  # type: ignore[arg-type]
              for kind in kinds for i in range(2)]
    options = ("keep it short", "go deep")
    return [*probes, *(Probe(f"{user_id}-taste-{i}", user_id, "taste_ab", f"Which do you prefer {i}?", options,
                             options[0], ()) for i in range(2))]


@dataclass
class Heard:
    at: datetime
    text: str
    ancestry: frozenset[str] | None


class ObjectiveTrainer:
    """The fake world's trainer with the receipt the real one gives: the objective the setting names, or
    ``sft`` for a night with no pair, and the pair count (``train/lora.py``'s ``objective_used``)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def train(self, user_id: str, night: date, prev: CheckpointRef | None, data: Any, name: str) -> Any:
        ckpt, receipt = self._inner.train(user_id, night, prev, data, name)
        used = objective_used(night_cfg(os.environ), data)
        return ckpt, {**receipt, "objective": used, "pairs": len(data.pairs) if used == "sft_dpo" else 0}


class PairedPractice:
    """The fake world's practice, as full practice writes it: each scene also keeps its worst candidate,
    so the night has (best, worst) pairs and trains ``sft_dpo``."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.max_spark_calls = inner.max_spark_calls

    def run(self, *args: Any, **kw: Any) -> Any:
        done = self._inner.run(*args, **kw)
        worse = tuple(replace(t, rejected=f"{t.reply} (flatter)", rejected_from="glimmer") for t in done.transcripts)
        return replace(done, transcripts=worse, pairs=len(worse))


@dataclass
class FakeAccelerated:
    root: Path
    world: World = field(init=False)
    clock: FakeClock = field(default_factory=FakeClock)
    heard: dict[str, list[Heard]] = field(default_factory=dict)
    mind_from: date | None = None  # MIND's inquiry on, from this first night (Task 17)
    thinker: LifeThinker | None = None
    resolver: Any = None  # MIND's prediction resolver; the default answers undetermined

    def __post_init__(self) -> None:
        self.world = World(self.root)

    # the person's own lines, as the history source reads them from the window and stamps them: at
    # the window's end or the night's clock, whichever is earlier
    def extract_day(self, user_id: str, day: date, window: tuple[datetime, datetime],
                    as_of: datetime | None = None) -> list[Evidence]:
        self.world.events.append(("extract", user_id, day, window))
        if user_id in self.world.extract_fails:
            raise self.world.extract_fails[user_id]
        items = [
            Evidence(id=f"{user_id}-{h.at.isoformat()}", person_id=user_id, day=day, kind="lesson", text=h.text,
                     source=f"message:{h.at.isoformat()}", held_out=False,
                     created_at=min(window[1], as_of or window[1]), ancestry=h.ancestry)
            for h in self.heard.get(user_id, []) if window[0] <= h.at < window[1]
        ]
        self.world.evidence(user_id).add(items)
        return items

    def simulate(self, user_id: str, person: Persona, day_index: int, seed: int) -> DayRun:
        outer = self

        class Transport:
            def send(self, session_id: str, text: str, *, cleared: bool) -> str:
                outer.heard.setdefault(user_id, []).append(Heard(outer.clock(), text, None))
                return f"Reply to {len(text.split())} words."

        if self.mind_from is not None and day_index == 0:
            for target in life_targets(user_id, self.mind_from):
                self.world.targets(user_id).upsert(target)
        before = len(self.heard.get(user_id, []))
        day = run_day(person, day_index, seed, model=Line(), transport=Transport(), replies=Replies(), clock=self.clock)
        users = [t for t in day.turns if t.role == "user"]
        for line, turn in zip(self.heard.get(user_id, [])[before:], users, strict=True):
            line.ancestry = turn.ancestry
        return day

    def scores_dir(self, user_id: str) -> Path:
        return self.root / "results" / user_id / "scores"

    def morning(self, user_id: str) -> Morning:
        arms = ArmDeps(model=FakeModel(), adapters=Adapters(self.world), contexts=FakeContexts(),
                       evidence=lambda uid: StoreSpy([]), logprobs=FakeLogprobs(lambda adapter, messages: "A"),
                       donors={user_id: f"{DONOR_PREFIX}{user_id}"}, until=CLOCK_START.date())
        return Morning(arms=arms, sealed=sealed_probes, judge=Verdicts(),
                       prices=Prices(spark_in_per_m=0.2, spark_out_per_m=0.6, train_gpu_per_hour=2.5,
                                     serve_gpu_per_hour=1.95),
                       root=self.root / "results")

    def score(self, user_id: str, night: date, label: str, result: NightResult) -> Any:
        return run_morning(user_id, night, result.checkpoint, morning=self.morning(user_id), label=label,
                           promoted=result.status == "promoted")

    def rig(self, sha: str = FAKE_SHA) -> Rig:
        deps = self.world.deps(extractor=lambda chat: self,
                               practice=lambda chat, night: PairedPractice(self.world.practice(chat, night)),
                               trainer=ObjectiveTrainer(self.world.trainer))
        if self.mind_from is not None:
            self.thinker = self.thinker or LifeThinker(targets=("fake-life-t1", "fake-life-c1"), queries=["pilot"])
            mind = mind_night(self.world, self.thinker, resolver=self.resolver)
            deps = replace(deps, settings=replace(deps.settings, mind_inquiry=True), inquiry=mind, notice=mind)
        return Rig(deps_for=lambda night: deps, simulate=self.simulate, sha=sha, score=self.score,
                   now=lambda: self.clock.now)
