# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""Fakes for the night's tests (Task 5): every port of ``NightDeps`` over temp files, with no
model, no GPU and no network. Stores are the real JSONL ones in plain text."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.checkpoints import CheckpointRegistry
from anjo.sleep.context.brief_store import JsonlBriefStore, TrainingBriefs
from anjo.sleep.context.builder import RecallSettings
from anjo.sleep.context.recall_bm25 import Bm25Recall
from anjo.sleep.context.training import TrainingContext
from anjo.sleep.evidence.jsonl import NO_GUARD, PLAIN
from anjo.sleep.evidence.store import JsonlEvidenceStore
from anjo.sleep.evidence.targets import EventLogTargetStore
from anjo.sleep.mind.choose import StateChooser
from anjo.sleep.mind.policy import PUBLIC_DEFAULT
from anjo.sleep.model_api import ChatResult
from anjo.sleep.night.deps import NightDeps, NightSettings, NoGpu, Prices, StudyOutcome
from anjo.sleep.night.meter import WonderMeter
from anjo.sleep.practice.run import MinimalPractice
from anjo.sleep.practice.transcripts import JsonlTranscriptStore
from anjo.sleep.probes.decontam import OverlapDecontaminator
from anjo.sleep.tests.sleep_fakes import WONDER, brief_text
from anjo.sleep.types import CheckpointRef, Evidence, GateResult, Probe, Result, Target

NIGHT = date(2026, 10, 28)
SOURCE_URL = "https://example.org/sorting-study"


def result(text: str) -> ChatResult:
    return ChatResult(text=text, tool_calls=(), model=SPARK_OPEN, input_tokens=100, output_tokens=50,
                      finish_reason="stop")


def window(night: date) -> tuple[datetime, datetime]:
    start = datetime.combine(night, time.min, tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


def own(uid: str, night: date, i: int, text: str | None = None) -> Evidence:
    return Evidence(
        id=f"{uid}-{night.isoformat()}-e{i}", person_id=uid, day=night, kind="lesson",
        text=text or f"Item {i}: I am rebuilding the sorting line and want plain answers.",
        source=f"message:{i}", held_out=False,
        created_at=datetime.combine(night, time(12, i % 60), tzinfo=timezone.utc), ancestry=frozenset({SPARK_OPEN}),
    )


def thread(uid: str, label: str = "the sorting line") -> Target:
    return Target(id=f"{uid}-t1", person_id=uid, kind="thread", label=label, check_after=None, status="open",
                  revision=1, acted_revision=0, confidence=None, last_mentioned=NIGHT)


class Extractor:
    """Adds each (person, night)'s planned items to the store, as the history source does."""

    def __init__(self, world: World) -> None:
        self.world = world

    def extract_day(self, user_id: str, day: date, win: tuple[datetime, datetime],
                    as_of: datetime | None = None) -> list[Evidence]:
        self.world.events.append(("extract", user_id, day, win))
        if user_id in self.world.extract_fails:
            raise self.world.extract_fails[user_id]
        items = self.world.planned.get((user_id, day), [])
        self.world.evidence(user_id).add(items)
        return items


class SparkChat:
    def __init__(self, scenes: int = 8) -> None:
        self.scenes = scenes
        self.prompts: list[str] = []

    def __call__(self, model: Any, messages: Any, tools: Any, timeout_s: float, *, temperature: float | None = None) -> ChatResult:
        content = messages[-1]["content"]
        self.prompts.append(content)
        if content.startswith("# brief_v1"):
            return result(brief_text("brief body"))
        if content.startswith("# scenes_v1"):
            return result(json.dumps({"scenes": [f"New brief {i}: a sorter for film plastics. Where do I start?"
                                                 for i in range(self.scenes)]}))
        raise AssertionError("an unexpected Spark Open call")


class Glimmer:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, adapter: Any, messages: list[dict[str, str]], *, temperature: float, seed: int | None = None) -> ChatResult:
        self.calls += 1
        return result(f"Start with the infeed and keep it simple, step {self.calls}.")


class Registrar:
    def __init__(self, world: World) -> None:
        self.world = world

    def register(self, user_id: str, night: date, store: Any) -> tuple[list[Probe], str]:
        self.world.events.append(("probes", user_id, night))
        return [Probe(f"{user_id}-q1", user_id, "new_work", "How would you price the film-plastics pilot?", None,
                      None, ())], "sealed once"


class Decontam:
    def __init__(self, world: World) -> None:
        self.world = world
        self.inner = OverlapDecontaminator()

    def check(self, texts: Any, sealed: Any) -> tuple[bool, ...]:
        self.world.events.append(("decontam", len(texts), len(sealed)))
        return self.inner.check(texts, sealed)


class Trainer:
    def __init__(self, world: World) -> None:
        self.world = world
        self.calls: list[dict[str, Any]] = []

    def train(self, user_id: str, night: date, prev: CheckpointRef | None, data: Any, name: str):
        self.calls.append({"user": user_id, "night": night, "prev": prev, "examples": len(data.examples), "name": name})
        if user_id in self.world.train_fails:
            raise self.world.train_fails[user_id]
        ckpt = CheckpointRef(name, user_id, night, f"modal-volume://anjo-sleep-adapters/{name}", prev.id if prev else None)
        return ckpt, {"steps": 8, "epochs": 4, "gpu_minutes": 12.0, "train_minutes": 5.0, "examples": len(data.examples)}


class Gates:
    def __init__(self, world: World) -> None:
        self.world = world
        self.checked: list[tuple[str, str]] = []

    def for_person(self, user_id: str):
        failing = self.world.failing_gates.get(user_id, set())

        def gate(name: str):
            def check(ckpt: CheckpointRef) -> GateResult:
                self.checked.append((user_id, name))
                passed = name not in failing
                return GateResult(name, passed, 0.0 if passed else 1.0, 0.0,  # type: ignore[arg-type]
                                  None if passed else "new reply trips on case indices 3")
            return check

        return {name: gate(name) for name in ("general", "collapse", "safety")}


class Notifier:
    def __init__(self, world: World) -> None:
        self.world = world
        self.queued: list[tuple[str, str, str]] = []
        self.pushed: list[str] = []

    def queue(self, user_id: str, message: str, source: str) -> None:
        self.queued.append((user_id, message, source))

    def push(self, user_id: str, message: str) -> bool:
        if user_id in self.world.opted_in:
            self.pushed.append(user_id)
            return True
        return False


class Writer:
    def __init__(self, world: World) -> None:
        self.world = world
        self.calls: list[tuple[str, list[Any]]] = []

    def write(self, user_id: str, topic: str, facts: Any) -> str | None:
        self.calls.append((topic, list(facts)))
        if self.world.writer_declines:
            return None
        return "While you were away I looked into optical sorting for film plastics."


class Homework:
    """A homework port with a verified source (Task 7's shape). ``calls`` is how many Wonder calls a
    study makes; ``failures`` of them fail and are retried, each one still debited."""

    def __init__(self, world: World, *, calls: int = 1, verified: int = 1) -> None:
        self.world = world
        self.available = True
        self.calls = calls
        self.verified = verified

    def study(
        self, user_id: str, target: Target, night: date, meter: WonderMeter, *, action: str = "",
        at: datetime | None = None,
    ) -> StudyOutcome:
        for _ in range(self.calls):
            meter.require()
            meter.debit(0.1)
        item = Evidence(
            id=f"hw-{target.id}-{night.isoformat()}", person_id=user_id, day=night, kind="homework",
            text="Optical sorters miss black film unless it is lit from below.", source=SOURCE_URL,
            held_out=False, created_at=window(night)[1] - timedelta(minutes=30), model=WONDER,
            ancestry=frozenset({WONDER}), target_id=target.id,
        )
        found = Result(id=f"r-{target.id}-{night.isoformat()}", target_id=target.id, homework_id=item.id,
                       verified_sources=self.verified, created_at=window(night)[1] - timedelta(minutes=20),
                       consumed=False)
        return StudyOutcome(result=found, evidence=(item,))


@dataclass
class World:
    root: Path
    events: list[tuple[Any, ...]] = field(default_factory=list)
    planned: dict[tuple[str, date], list[Evidence]] = field(default_factory=dict)
    extract_fails: dict[str, Exception] = field(default_factory=dict)
    train_fails: dict[str, Exception] = field(default_factory=dict)
    failing_gates: dict[str, set[str]] = field(default_factory=dict)
    opted_in: set[str] = field(default_factory=set)
    writer_declines: bool = False
    clock: float = 0.0

    def __post_init__(self) -> None:
        self.extractor = Extractor(self)
        self.spark = SparkChat()
        self.glimmer = Glimmer()
        self.trainer = Trainer(self)
        self.gates = Gates(self)
        self.notifier = Notifier(self)
        self.writer = Writer(self)
        self.gpu = NoGpu()
        self.registry = CheckpointRegistry(self.root / "checkpoints")

    def evidence(self, uid: str) -> JsonlEvidenceStore:
        return JsonlEvidenceStore(self.root / uid / "evidence.jsonl", uid, codec=PLAIN, guard=NO_GUARD)

    def targets(self, uid: str) -> EventLogTargetStore:
        return EventLogTargetStore(self.root / uid / "targets.jsonl", uid, codec=PLAIN, guard=NO_GUARD)

    def transcripts(self, uid: str) -> JsonlTranscriptStore:
        return JsonlTranscriptStore(self.root / uid / "practice.jsonl", uid, codec=PLAIN, guard=NO_GUARD)

    def briefs(self, uid: str) -> JsonlBriefStore:
        return JsonlBriefStore(self.root / uid / "briefs.jsonl", uid, codec=PLAIN, guard=NO_GUARD)

    def plan(self, uid: str, night: date, count: int = 3) -> None:
        self.planned[(uid, night)] = [own(uid, night, i) for i in range(count)]

    def logs(self, uid: str) -> Path:
        return self.root / "logs" / uid

    def tick(self) -> float:
        self.clock += 60.0
        return self.clock

    def practice(self, chat: Any, night: date) -> MinimalPractice:
        recall = Bm25Recall(self.evidence, until=night)
        context = TrainingContext(TrainingBriefs(self.briefs), recall, RecallSettings(k=6, budget_tokens=1200))
        return MinimalPractice(chat=chat, client=self.glimmer, snapshot=context.snapshot, seed=7)

    def deps(self, **changes: Any) -> NightDeps:
        deps = NightDeps(
            extractor=lambda chat: self.extractor, extract_chat=lambda messages: result("{}"),
            evidence=self.evidence, targets=self.targets, transcripts=self.transcripts, briefs=self.briefs,
            spark_chat=self.spark, probes=Registrar(self), chooser=StateChooser(), policy=PUBLIC_DEFAULT,
            practice=self.practice, decontam=Decontam(self), trainer=self.trainer, gates=self.gates,
            registry=self.registry, notifier=self.notifier, note_writer=self.writer, logs=self.logs,
            is_synthetic=lambda uid: True,
            settings=NightSettings(token_cap=16384, min_new_examples=1),
            prices=Prices(spark_in_per_m=0.2, spark_out_per_m=0.6, train_gpu_per_hour=2.5, serve_gpu_per_hour=1.95),
            gpu=self.gpu, timer=self.tick,
        )
        return replace(deps, **changes)


def steps(log_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]


def step(log_path: Path, name: str) -> dict[str, Any]:
    return next(record for record in steps(log_path) if record["step"] == name)


def summary(log_path: Path) -> dict[str, Any]:
    return json.loads((log_path.parent / "summary.json").read_text(encoding="utf-8"))
