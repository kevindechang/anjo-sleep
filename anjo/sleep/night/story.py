"""The night's story: the night as one loop, told from its one log (one system, 2026-10-29).

    sleep → dream → rehearse → learn → check → wake

She lives with you by day, dreams at night about what she's unsure of, rehearses exactly that, and
wakes up a better mind of you. ``story.json`` sits beside ``log.jsonl`` and ``summary.json`` in the
night's folder (``nightlog.STORY``). It is never a second log: ``tell`` reads the night's log records
and nothing else, so the story is what the log says, in the loop's shape. The night writes it when it
finishes (``run.py``), and the morning tells it again once its probe run is logged (``daily.py``).

Every field is its record, ``NotRun`` (why it didn't run) or, where a step can fail inside the loop,
``Failed`` (the step and its error): never missing, never ``None``, never a zero standing in for
"didn't run". A stage that ran but whose part belongs to a branch not merged yet says so in its
``NotRun``. Ids, codes and numbers only: no person's words, no gate's reason (it can name a safety
case). ``STORY.md`` says which branch fills which record.

``Story.record()`` is the JSON record (``version`` first, ``STORY_VERSION``) and ``story_from`` reads
one back, refusing anything the schema doesn't hold (``StoryError`` names where). A change to a
record's fields is a new version. Pure: no clock, file or environment.
"""
from __future__ import annotations

import math
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import date
from typing import Any, Final, Literal, Union, get_args, get_origin, get_type_hints

from anjo.sleep.types import NightStatus

STORY_VERSION: Final = 1
STAGES: Final = ("dream", "rehearse", "learn", "check", "wake")
STAGE_STEPS: Final[Mapping[str, tuple[str, ...]]] = {  # the night log's steps each stage is told from
    "dream": ("notice",), "rehearse": ("practice",), "learn": ("dataset", "train"), "check": ("gates", "promote"),
}
MORNING_FAILED: Final = "morning failed: "  # a failed morning step's ``why`` starts with this (``daily.py``)

# Why a record didn't run, in plain words (the page shows them as they are).
NO_PRESSURE: Final = "no sleep pressure: every night sleeps deep"
LEARNING_TOMORROW: Final = "napped: learning tomorrow"
NOTHING_NEW: Final = "too little new since her last training: nothing to learn tonight"
SCORED_IN_MORNING: Final = "scored in the morning"
UNCOUNTED: Final = "the morning didn't count its replies"
NOT_LOGGED: Final = "its steps aren't in the night's log"
UNLOGGED_STEP: Final = "a step it didn't log"
MIND_OFF: Final = "MIND's inquiry is off"
NO_CRITIC: Final = "this practice ran no critic"
REUSED: Final = "practice reused tonight's earlier run, whose log has the critic's counts"
DREAM_NOT_YET: Final = "MIND's core doesn't dream in the night yet"  # mind-frontier
CURRICULUM_NOT_YET: Final = "practice isn't seeded on the dream's threads yet"  # mind-frontier
ADAPTATION_NOT_YET: Final = "MIND's core doesn't adapt to her in the night yet"  # mind-frontier
SELF_MODEL_NOT_YET: Final = "MIND's self-model isn't scored yet"  # mind-frontier
CHANNEL_NOT_YET: Final = "the thought channel doesn't train in the night yet"  # thought-channel
SURPRISE_NOT_YET: Final = "MIND's surprise doesn't weight the loss yet"  # sleep-objective
SEEDING_NOT_YET: Final = "the night's training has no seeding term yet"  # sleep-objective
FOLD_NOT_YET: Final = "the night's adapter isn't folded yet"  # sleep-objective


class StoryError(ValueError):
    """A story, or a log it is told from, that doesn't hold what the schema needs."""


@dataclass(frozen=True)
class NotRun:
    """This record didn't run tonight, and why."""

    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise StoryError("a not_run says why")


@dataclass(frozen=True)
class Failed:
    """The step that failed here and its error (``state.safe_error``: a Sleep error's message, else its type)."""

    step: str
    error: str


# ── sleep (this branch: ``sleep_pressure.py``) ───────────────────────────────


@dataclass(frozen=True)
class Sleep:
    """A nap or a deep sleep, its pressure against its threshold, and why (a ``sleep_pressure.Reason``)."""

    kind: Literal["nap", "deep"]
    pressure: float
    threshold: float
    reason: str


# ── dream (mind-frontier; ``questions`` from this branch's notice pass) ────────


@dataclass(frozen=True)
class Ignition:
    """A thread the core's dream lit: ``risk``, ``missing`` or ``collision``, citing real event ids only."""

    kind: str
    score: float
    cites: tuple[str, ...]


@dataclass(frozen=True)
class Thread:
    """A thread the dream picked for practice: its slot, its score and the events it rests on."""

    slot: int
    score: float
    cites: tuple[str, ...]


@dataclass(frozen=True)
class Boundary:
    """An imagined state at the edge of a decision: its id in her mind store, its slot, its dream steps."""

    state: str
    slot: int
    steps: int


@dataclass(frozen=True)
class Dream:
    """What she was unsure of: the questions tonight opened, and the core's ignitions, picks and boundaries."""

    questions: int | NotRun
    ignitions: tuple[Ignition, ...] | NotRun
    threads: tuple[Thread, ...] | NotRun
    boundaries: tuple[Boundary, ...] | NotRun


# ── rehearse (this branch; ``seeded`` mind-frontier) ─────────────────────────


@dataclass(frozen=True)
class Seed:
    """Scenes seeded on one picked thread."""

    slot: int
    scenes: int


@dataclass(frozen=True)
class Rehearse:
    """Practice: its scenes, how many were seeded on each picked thread, the scenes where the critic
    marked a weaker reply (a pair kept), and those whose target is the teacher's correction."""

    scenes: int
    seeded: tuple[Seed, ...] | NotRun
    flagged: int | NotRun
    corrected: int | NotRun


# ── learn (``rows`` this branch; the rest as each field says) ────────────────


@dataclass(frozen=True)
class Surprise:
    """Whether the loss applied MIND's surprise weights, how many rows carried one, and their mean."""

    applied: bool
    rows: int
    mean: float


@dataclass(frozen=True)
class Learn:
    """The night's one training job: its rows (this branch), MIND's surprise on them (sleep-objective),
    the thought channel's loss and the projector's stamp (thought-channel), the seeding term's KL and
    the rank the adapter was folded to (sleep-objective), and the core's per-person adaptation loss on
    the outcomes that resolved today (mind-frontier)."""

    rows: int
    surprise: Surprise | NotRun
    channel_loss: float | NotRun
    stamp: str | NotRun
    seeding_kl: float | NotRun
    fold: int | NotRun
    core_adaptation: float | NotRun


# ── check (this branch) ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Gate:
    """One gate: passed, or switched off (``skipped``, never passed), its value against its threshold."""

    name: str
    passed: bool
    skipped: bool
    value: float
    threshold: float


@dataclass(frozen=True)
class Check:
    gates: tuple[Gate, ...]
    promoted: bool


# ── wake (``morning`` this branch; ``self_model`` mind-frontier) ─────────────


@dataclass(frozen=True)
class Morning:
    """The morning's probe run: the replies it scored, and the comparisons the judge gave no verdict on."""

    replies: int | NotRun
    no_verdict: int


@dataclass(frozen=True)
class SelfModel:
    """MIND's predicted error in its own forecasts against the error it made: the gap is tomorrow's surprise."""

    predicted: float
    actual: float


@dataclass(frozen=True)
class Wake:
    morning: Morning | Failed | NotRun
    self_model: SelfModel | NotRun


@dataclass(frozen=True)
class Story:
    night: date
    status: NightStatus
    sleep: Sleep | Failed | NotRun
    dream: Dream | Failed | NotRun
    rehearse: Rehearse | Failed | NotRun
    learn: Learn | Failed | NotRun
    check: Check | Failed | NotRun
    wake: Wake | NotRun

    def record(self) -> dict[str, Any]:
        encoded: dict[str, Any] = _encode(self)
        return {"version": STORY_VERSION, **encoded}


# ── the record and back ──────────────────────────────────────────────────────


def _encode(value: Any) -> Any:
    if isinstance(value, NotRun):
        return {"not_run": value.reason}
    if isinstance(value, Failed):
        return {"failed": {"step": value.step, "error": value.error}}
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if isinstance(value, date):
        return value.isoformat()
    return value


def _marker(raw: Any, key: str) -> bool:
    return isinstance(raw, dict) and set(raw) == {key}


def _union(options: tuple[Any, ...], raw: Any, where: str) -> Any:
    if NotRun in options and _marker(raw, "not_run"):
        return _decode(NotRun, {"reason": raw["not_run"]}, where)
    if Failed in options and _marker(raw, "failed"):
        return _decode(Failed, raw["failed"], where)
    rest = [option for option in options if option not in (NotRun, Failed)]
    if len(rest) != 1:
        raise StoryError(f"{where}: no record reads it")
    return _decode(rest[0], raw, where)


def _scalar(hint: Any, raw: Any, where: str) -> Any:
    number = isinstance(raw, int | float) and not isinstance(raw, bool)
    if hint is bool and isinstance(raw, bool) or hint is str and isinstance(raw, str):
        return raw
    if hint is int and number and isinstance(raw, int):
        return raw
    if hint is float and number and math.isfinite(raw):
        return float(raw)
    if hint is date and isinstance(raw, str):
        try:
            return date.fromisoformat(raw)
        except ValueError:
            pass
    raise StoryError(f"{where}: {raw!r} is not a {getattr(hint, '__name__', hint)}")


def _decode(hint: Any, raw: Any, where: str) -> Any:
    origin = get_origin(hint)
    if origin in (Union, types.UnionType):
        return _union(get_args(hint), raw, where)
    if origin is Literal:
        if raw not in get_args(hint):
            raise StoryError(f"{where}: {raw!r} is not one of {list(get_args(hint))}")
        return raw
    if origin is tuple:
        if not isinstance(raw, list):
            raise StoryError(f"{where}: not a list")
        return tuple(_decode(get_args(hint)[0], item, f"{where}[{n}]") for n, item in enumerate(raw))
    if isinstance(hint, type) and is_dataclass(hint):
        return _record(hint, raw, where)
    return _scalar(hint, raw, where)


def _record(cls: type, raw: Any, where: str) -> Any:
    names, at = [f.name for f in fields(cls)], where or "story"
    if not isinstance(raw, dict):
        raise StoryError(f"{at}: not a record")
    if missing := [name for name in names if name not in raw]:
        raise StoryError(f"{at}: no {', '.join(missing)}")
    if extra := sorted(set(raw) - set(names)):
        raise StoryError(f"{at}: no field {', '.join(extra)}")
    hints = get_type_hints(cls)
    values = {name: _decode(hints[name], raw[name], f"{where}.{name}" if where else name) for name in names}
    try:
        return cls(**values)
    except StoryError as error:  # a record's own check (a not_run with no reason)
        raise StoryError(f"{at}: {error}") from None


def story_from(raw: Any) -> Story:
    """A story from its record; ``StoryError`` names the field that breaks the schema."""
    if not isinstance(raw, dict) or raw.get("version") != STORY_VERSION:
        version = raw.get("version") if isinstance(raw, dict) else None
        raise StoryError(f"story version {version!r}: this reads version {STORY_VERSION}")
    story: Story = _record(Story, {k: v for k, v in raw.items() if k != "version"}, "")
    return story


# ── telling it from the log ──────────────────────────────────────────────────

Steps = Mapping[str, Mapping[str, Any]]


def _need(record: Mapping[str, Any], key: str, kind: Literal["int", "number", "text", "record"]) -> Any:
    value = record.get(key)
    ok = {"int": isinstance(value, int), "number": isinstance(value, int | float) and math.isfinite(value),
          "text": isinstance(value, str), "record": isinstance(value, dict)}[kind]
    if not ok or isinstance(value, bool):
        raise StoryError(f"the {record.get('step')} step's record has no {key} ({kind})")
    return value


def _failed(record: Mapping[str, Any]) -> Failed:
    return Failed(str(record["step"]), str(record.get("error") or record.get("why") or "no error logged"))


def _sleep(by: Steps) -> Sleep | Failed | NotRun:
    record = by.get("sleep")
    if record is None:
        return NotRun(NO_PRESSURE)
    if record.get("status") == "failed":
        return _failed(record)
    kind = _need(record, "kind", "text")
    if kind not in ("nap", "deep"):
        raise StoryError(f"the sleep step's kind {kind!r} is neither nap nor deep")
    return Sleep(kind, float(_need(record, "pressure", "number")), float(_need(record, "threshold", "number")),
                 _need(record, "reason", "text"))


def _dream(by: Steps) -> Dream:
    notice = by["notice"]
    questions: int | NotRun = (NotRun(MIND_OFF) if notice.get("noop")
                               else NotRun(f"the notice pass stopped: {notice['stopped']}") if notice.get("stopped")
                               else _need(notice, "questions_opened", "int"))
    return Dream(questions, NotRun(DREAM_NOT_YET), NotRun(DREAM_NOT_YET), NotRun(DREAM_NOT_YET))


def _critic_ran(practice: Mapping[str, Any]) -> bool:
    spark = practice.get("spark")
    critic = spark.get("critic") if isinstance(spark, Mapping) else None
    return isinstance(critic, Mapping) and isinstance(critic.get("calls"), int) and critic["calls"] > 0


def _rehearse(by: Steps) -> Rehearse:
    practice = by["practice"]
    scenes = _need(practice, "scenes", "int")
    if practice.get("reused") or not _critic_ran(practice):
        why = NotRun(REUSED if practice.get("reused") else NO_CRITIC)
        return Rehearse(scenes, NotRun(CURRICULUM_NOT_YET), why, why)
    return Rehearse(scenes, NotRun(CURRICULUM_NOT_YET), _need(practice, "pairs", "int"),
                    _need(practice, "teacher_targets", "int"))


def _learn(by: Steps) -> Learn:
    return Learn(_need(by["dataset"], "examples", "int"), NotRun(SURPRISE_NOT_YET), NotRun(CHANNEL_NOT_YET),
                 NotRun(CHANNEL_NOT_YET), NotRun(SEEDING_NOT_YET), NotRun(FOLD_NOT_YET), NotRun(ADAPTATION_NOT_YET))


def _check(by: Steps) -> Check:
    gates = _need(by["gates"], "gates", "record")
    try:
        told = tuple(Gate(str(name), bool(g["passed"]), bool(g.get("skipped", False)), float(g["value"]), float(g["threshold"]))
                     for name, g in sorted(gates.items()))
    except (KeyError, TypeError, ValueError) as error:
        raise StoryError(f"the gates step's record has a gate without {error}") from None
    return Check(told, by["promote"].get("promoted") is not None)


def _morning(record: Mapping[str, Any] | None) -> Morning | Failed | NotRun:
    if record is None:
        return NotRun(SCORED_IN_MORNING)
    if record.get("status") == "failed":
        return Failed("morning", str(record.get("why") or "no error logged").removeprefix(MORNING_FAILED))
    if record.get("status") == "skipped":
        return NotRun(str(record.get("why") or "skipped"))
    replies = record.get("replies")
    return Morning(replies if isinstance(replies, int) and not isinstance(replies, bool) else NotRun(UNCOUNTED),
                   int(record.get("no_verdict") or 0))


def _stopped(step: str | None) -> NotRun:
    return NotRun(f"the night stopped at {step or UNLOGGED_STEP}")


def _stage(name: str, by: Steps, status: str, stopped: str | None, build: Callable[[Steps], Any]) -> Any:
    told = [by[step] for step in STAGE_STEPS[name] if step in by]
    if failed := next((r for r in told if r.get("status") == "failed"), None):
        return _failed(failed)
    if status == "napped" and name != "dream":
        return NotRun(LEARNING_TOMORROW)
    if len(told) < len(STAGE_STEPS[name]):
        return _stopped(stopped) if stopped or status == "failed" else NotRun(NOT_LOGGED)
    if skipped := next((r for r in told if r.get("status") == "skipped"), None):
        return NotRun(NOTHING_NEW if status == "nothing_new" else str(skipped.get("why") or "skipped"))
    return build(by)


def _wake(by: Steps, status: str, stopped: str | None) -> Wake | NotRun:
    if status == "napped":
        return NotRun(LEARNING_TOMORROW)
    if stopped is not None or status == "failed":
        return _stopped(stopped)
    return Wake(_morning(by.get("morning")), NotRun(SELF_MODEL_NOT_YET))


def tell(night: date, status: NightStatus, steps: Sequence[Mapping[str, Any]]) -> Story:
    """The night's story from its log records (``NightLog.steps()``), the latest record of each step."""
    by = {str(record.get("step")): record for record in steps}
    stopped = next((str(r["step"]) for r in steps if r.get("status") == "failed" and r.get("step") != "morning"), None)
    builders: Mapping[str, Callable[[Steps], Any]] = {
        "dream": _dream, "rehearse": _rehearse, "learn": _learn, "check": _check}
    stages = {name: _stage(name, by, status, stopped, build) for name, build in builders.items()}
    return Story(night=night, status=status, sleep=_sleep(by), wake=_wake(by, status, stopped), **stages)
