# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""An accelerated rig on fakes (Task 12a): the night's fake world, a simulated day that stamps every
line with a fake clock, and an extractor that reads back only the lines inside the night's window.
No model, no GPU, no network. ``python -m anjo.sleep.sim.accelerated --fake`` runs on it."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from anjo.sleep.personas.model import Persona
from anjo.sleep.provenance import GLIMMER
from anjo.sleep.sim.accelerated import Rig
from anjo.sleep.sim.day import DayRun, run_day
from anjo.sleep.tests.night_fakes import World
from anjo.sleep.types import Evidence

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

    def rig(self, sha: str = FAKE_SHA) -> Rig:
        deps = self.world.deps(extractor=lambda chat: self)
        return Rig(deps_for=lambda night: deps, simulate=self.simulate, sha=sha)
