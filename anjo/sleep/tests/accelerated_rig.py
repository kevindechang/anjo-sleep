# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""An accelerated rig on fakes (Task 12a): the night's fake world, a simulated day that stamps every
line with a fake clock, and an extractor that reads back only the lines inside the night's window.
No model, no GPU, no network. ``python -m anjo.sleep.sim.accelerated --fake`` runs on it."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
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
from anjo.sleep.tests.night_fakes import World
from anjo.sleep.tests.sleep_fakes import FakeContexts, FakeLogprobs, FakeModel, StoreSpy
from anjo.sleep.types import CheckpointRef, Evidence, NightResult, Probe

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


@dataclass
class FakeAccelerated:
    root: Path
    world: World = field(init=False)
    clock: FakeClock = field(default_factory=FakeClock)
    heard: dict[str, list[Heard]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.world = World(self.root)

    # the person's own lines, as the history source reads them from the window
    def extract_day(self, user_id: str, day: date, window: tuple[datetime, datetime]) -> list[Evidence]:
        self.world.events.append(("extract", user_id, day, window))
        if user_id in self.world.extract_fails:
            raise self.world.extract_fails[user_id]
        items = [
            Evidence(id=f"{user_id}-{h.at.isoformat()}", person_id=user_id, day=day, kind="lesson", text=h.text,
                     source=f"message:{h.at.isoformat()}", held_out=False, created_at=h.at, ancestry=h.ancestry)
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
        deps = self.world.deps(extractor=lambda chat: self)
        return Rig(deps_for=lambda night: deps, simulate=self.simulate, sha=sha, score=self.score)
