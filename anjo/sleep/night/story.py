"""The night's story: the night as one loop, told from its one log (one system, 2026-10-29).

    sleep → dream → rehearse → learn → check → wake

She lives with you by day, dreams at night about what she's unsure of, rehearses exactly that, and
wakes up a better mind of you. ``story.json`` sits beside ``log.jsonl`` and ``summary.json`` in the
night's folder (``nightlog.STORY``). It is never a second log: ``tell`` reads the night's log records
and nothing else, so the story is what the log says, in the loop's shape. The night writes it when it
finishes (``run.py``), and the morning tells it again once its probe run is logged (``daily.py``).

Every field is its record, ``NotRun`` (why it didn't run) or, where a step can fail inside the loop,
``Failed`` (the step and its error): never missing, never ``None``, never a zero standing in for
"didn't run". A record no step logs yet says so in its ``NotRun``, in plain words; so does one
that's off tonight. Ids, codes and numbers only: no person's words, no gate's reason (it can name a safety
case). ``STORY.md`` says which branch fills which record.

``Story.record()`` is the JSON record (``version`` first, ``STORY_VERSION``) and ``story_from`` reads
one back, refusing anything the schema doesn't hold (``StoryError`` names where), an older version
too: the page then tells that night again from its log. A change to a record's fields is a new
version (2: practice reused from an earlier run is its own record, ``Reused``; 3: the brief, what it
folded or, when it couldn't be rewritten, the night whose brief was kept, ``Kept``; 4: what the core's
night and the sleep objective log, Glimmer's ``Dreams`` and their keep rate, practice ``Seeded`` on the
dream's picks, and the training terms ``Seeding``, ``Imitation``, the fold and ``Protect``). Pure: no
clock, file or environment.

A kept brief is a ``failed`` brief step carrying ``KEPT``: the night went on with the briefs it had,
so it never reads as where the night stopped.
"""
from __future__ import annotations

import math
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import MISSING, dataclass, fields, is_dataclass, replace
from datetime import date
from pathlib import PurePosixPath
from typing import Any, Final, Literal, Union, cast, get_args, get_origin, get_type_hints

from anjo.sleep.types import NightStatus

STORY_VERSION: Final = 4
STATUSES: Final = frozenset(get_args(NightStatus))
STAGES: Final = ("dream", "rehearse", "learn", "check", "wake")
STAGE_STEPS: Final[Mapping[str, tuple[str, ...]]] = {  # the night log's steps each stage is told from
    "dream": ("notice",), "rehearse": ("practice",), "learn": ("dataset", "train"), "check": ("gates", "promote"),
}
CORE_DREAM: Final = "dream"  # MIND's core's dream step (mind-frontier), logged after ``notice`` when the core serves her
STORY_STEP: Final = "story"  # a story that couldn't be told or written, logged as this failed step (``nightlog``)
# Steps logged once the night ended (the morning's runners, ``daily`` and ``accelerated``, and the story):
# their failure never stopped the night, so it never reads as where it stopped.
AFTER_THE_NIGHT: Final = ("morning", "dailies", STORY_STEP)
KEPT: Final = "kept"  # a failed brief step's night whose briefs were kept (``run.py``): the night went on
EMPTY: Final = "empty"  # what a first night keeps: no brief was written before it
MORNING_FAILED: Final = "morning failed: "  # a failed morning step's ``why`` starts with this (``daily.py``)
STAMP_CHARS: Final = 12  # a projector's stamp: the first characters of its id (``projector_math.params_sha``)
NOT_WIRED: Final = "not wired"  # the dataset's dreams while Glimmer's aren't wired (``train.terms.DREAMS_NOT_WIRED``)
APPLIED: Final = "applied"  # the training step's surprise_weights when the loss took them (``colab_train``)

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
NO_CURRICULUM: Final = "the dream's curriculum is off: it picked no threads"
CHANNEL_OFF: Final = "the thought channel was off tonight"
NO_CHANNEL_LOSS: Final = "the channel's training reported no loss"
NO_STAMP: Final = "the channel's training reported no projector"
CORE_OFF: Final = "MIND's core didn't dream for her tonight"  # no dream step: the core is off, or not hers
FELL_BACK: Final = "none of the dream's picks could train, so practice ran on the newest evidence"
SEEDING_UNLOGGED: Final = "practice didn't log what it was seeded on"
TERMS_OFF: Final = "the night trained without the sleep objective's terms"
DREAMS_NOT_WIRED: Final = "Glimmer's dreams aren't wired"
DREAMS_OFF: Final = "the dreams term is off tonight"
NO_DREAM_SEEDS: Final = "her mind gave Glimmer no dream seeds tonight"
NO_SURPRISE: Final = "no row carried MIND's surprise weight"
NO_SEEDS: Final = "no seeding row survived tonight"
NO_IMITATION: Final = "no imitation row tonight"
FOLD_OFF: Final = "the adapter wasn't folded tonight"
PROTECT_OFF: Final = "no protected slot tonight"
NOT_REPORTED: Final = "the training job didn't report it"
ADAPTATION_NOT_YET: Final = "MIND's core doesn't adapt to her in the night yet"  # mind-frontier: no step logs it
SELF_MODEL_NOT_YET: Final = "MIND's self-model isn't scored yet"  # mind-frontier: no step logs it


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
    """The step that failed here and its error (``nightlog.safe_error``: a Sleep error's message, else its type)."""

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
class Dreams:
    """Glimmer's dreams from her mind's dream seeds (the sleep objective's ``dreams`` term, in the dataset
    step): the candidates she wrote, those the dataset's filter kept, and its keep rate."""

    candidates: int
    kept: int
    keep_rate: float


@dataclass(frozen=True)
class Dream:
    """What she was unsure of: the questions tonight opened, the core's ignitions, picks and boundaries
    (a nap's CPU dream too), and Glimmer's dreams from them."""

    questions: int | NotRun
    ignitions: tuple[Ignition, ...] | NotRun
    threads: tuple[Thread, ...] | NotRun
    boundaries: tuple[Boundary, ...] | NotRun
    dreams: Dreams | NotRun


# ── rehearse (this branch; ``seeded`` the core's) ────────────────────────────


@dataclass(frozen=True)
class Seeded:
    """Practice seeded on the dream's picks: the threads they were picked on, and how many it used."""

    threads: tuple[int, ...]
    picked: int


@dataclass(frozen=True)
class Rehearse:
    """Practice: its scenes, what of the dream it was seeded on, the scenes where the critic marked a
    weaker reply (a pair kept), and those whose target is the teacher's correction."""

    scenes: int
    seeded: Seeded | NotRun
    flagged: int | NotRun
    corrected: int | NotRun


@dataclass(frozen=True)
class Reused:
    """Practice that didn't run again: a rerun of ``night`` reused the scenes its earlier run made
    and paid for."""

    night: date
    scenes: int


# ── learn (``rows`` this branch; the rest as each field says) ────────────────


@dataclass(frozen=True)
class Surprise:
    """Whether the loss applied MIND's surprise weights, and how many rows carried one."""

    applied: bool
    rows: int


@dataclass(frozen=True)
class Seeding:
    """Knowledge seeding: its rows, the seeding KL on its probe row before and after training, its minutes."""

    rows: int
    kl_before: float | NotRun
    kl_after: float | NotRun
    minutes: float | NotRun


@dataclass(frozen=True)
class Imitation:
    """Imitation on her own samples: its rows, the mean reward, its minutes."""

    rows: int
    reward: float | NotRun
    minutes: float | NotRun


@dataclass(frozen=True)
class Protect:
    """The protected slot: the basis tonight's delta trained off (``anchors``, or ``random`` for its twin),
    its rank, and the minutes the basis took."""

    basis: str
    rank: int
    minutes: float


@dataclass(frozen=True)
class Learn:
    """The night's one training job: its rows, MIND's surprise on them, the thought channel's loss and
    the projector's stamp (thought-channel), the sleep objective's seeding and imitation terms, the rank
    the adapter was folded to and the protected slot, and the core's per-person adaptation loss on the
    outcomes that resolved today (mind-frontier: no step logs it yet)."""

    rows: int
    surprise: Surprise | NotRun
    channel_loss: float | NotRun
    stamp: str | NotRun
    seeding: Seeding | NotRun
    imitation: Imitation | NotRun
    fold: int | NotRun
    protect: Protect | NotRun
    core_adaptation: float | NotRun
    # One strategy line the harness promoted tonight, in its words; empty on every other night, and
    # then the story has no such key and is byte-identical to today's.
    learned: tuple[str, ...] = ()


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
class Brief:
    folded: int  # the items tonight folded into both briefs


@dataclass(frozen=True)
class Kept:
    """The brief couldn't be rewritten tonight (it broke a rule twice), so the briefs she had were kept
    and the night went on with them: ``kept_from`` is their night (ISO), or ``EMPTY`` on a first night."""

    kept_from: str
    error: str


@dataclass(frozen=True)
class Story:
    night: date
    status: NightStatus
    sleep: Sleep | Failed | NotRun
    brief: Brief | Kept | Failed | NotRun
    dream: Dream | Failed | NotRun
    rehearse: Rehearse | Reused | Failed | NotRun
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
        return {f.name: _encode(getattr(value, f.name)) for f in fields(value)
                if not (f.name == "learned" and not getattr(value, f.name))}
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
    if len(rest) > 1 and isinstance(raw, dict):  # two records (practice, or a reused one): its fields say which
        rest = [option for option in rest if is_dataclass(option) and {f.name for f in fields(option)} == set(raw)]
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
    optional = {f.name for f in fields(cls) if f.default is not MISSING or f.default_factory is not MISSING}
    if missing := [name for name in names if name not in raw and name not in optional]:
        raise StoryError(f"{at}: no {', '.join(missing)}")
    if extra := sorted(set(raw) - set(names)):
        raise StoryError(f"{at}: no field {', '.join(extra)}")
    hints = get_type_hints(cls)
    values = {name: _decode(hints[name], raw[name], f"{where}.{name}" if where else name)
              for name in names if name in raw}
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


def _need(record: Mapping[str, Any], key: str, kind: Literal["int", "number", "text", "record"],
          step: str | None = None) -> Any:
    value = record.get(key)
    ok = {"int": isinstance(value, int), "number": isinstance(value, int | float) and math.isfinite(value),
          "text": isinstance(value, str), "record": isinstance(value, dict)}[kind]
    if not ok or isinstance(value, bool):
        raise StoryError(f"the {step or record.get('step')} step's record has no {key} ({kind})")
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


def _briefed(by: Steps, status: str, stopped: str | None) -> Brief | Kept | Failed | NotRun:
    record = by.get("brief")
    if record is None:
        return _stopped(stopped) if stopped or status == "failed" else NotRun(NOT_LOGGED)
    if record.get("status") == "failed":
        return Kept(_need(record, KEPT, "text"), _failed(record).error) if KEPT in record else _failed(record)
    if record.get("status") == "skipped":
        return NotRun(str(record.get("why") or "skipped"))
    folded = _need(record, "folded", "int")
    return Brief(folded) if folded else NotRun(str(record.get("why") or "nothing to fold"))


def _dream(by: Steps) -> Dream:
    """The questions tonight's notice pass opened; the core's parts and Glimmer's dreams are ``_dreamt``'s."""
    notice = by["notice"]
    questions: int | NotRun = (NotRun(MIND_OFF) if notice.get("noop")
                               else NotRun(f"the notice pass stopped: {notice['stopped']}") if notice.get("stopped")
                               else _need(notice, "questions_opened", "int"))
    return Dream(questions, NotRun(CORE_OFF), NotRun(CORE_OFF), NotRun(CORE_OFF), NotRun(NOT_LOGGED))


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{value!r} is not a number")
    return float(value)


def _cites(record: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(cite) for cite in record["cites"])


def _core(record: Mapping[str, Any]) -> tuple[tuple[Ignition, ...], tuple[Thread, ...] | NotRun, tuple[Boundary, ...]]:
    """The core's dream from its step's record (mind-frontier's ``mind.core.night`` receipt): its
    ignitions, the threads its curriculum picked (``picks``: None with the curriculum off) and its
    boundaries."""
    try:
        ignitions = tuple(Ignition(str(i["kind"]), _number(i["score"]), _cites(i)) for i in record["ignitions"])
        picks = record["picks"]
        threads = (NotRun(NO_CURRICULUM) if picks is None else
                   tuple(Thread(int(t["slot"]), _number(t["score"]), _cites(t)) for t in picks["threads"]))
        boundaries = tuple(Boundary(str(b["state"]), int(b["slot"]), int(b["steps"])) for b in record["boundaries"])
    except (KeyError, TypeError, ValueError) as error:
        raise StoryError(f"the dream step's record has no readable {error}") from None
    return ignitions, threads, boundaries


def _glimmer_dreams(by: Steps, status: str, stopped: str | None) -> Dreams | NotRun:
    """Glimmer's dreams, from the dataset step's ``terms`` (``train.terms``): a GPU step, so a nap's wait."""
    dataset = by.get("dataset")
    if status == "napped":
        return NotRun(LEARNING_TOMORROW)
    if dataset is None or dataset.get("status") == "failed":
        return _stopped(stopped) if stopped or status == "failed" else NotRun(NOT_LOGGED)
    if dataset.get("status") == "skipped":
        return NotRun(NOTHING_NEW if status == "nothing_new" else str(dataset.get("why") or "skipped"))
    terms = dataset.get("terms")
    if not isinstance(terms, Mapping):
        return NotRun(TERMS_OFF)
    if terms.get("dreams") == NOT_WIRED:
        return NotRun(DREAMS_NOT_WIRED)
    if "dream_candidates" not in terms:
        return NotRun(DREAMS_OFF)
    candidates = _need(terms, "dream_candidates", "int", "dataset")
    if not candidates:
        return NotRun(NO_DREAM_SEEDS)
    return Dreams(candidates, _need(terms, "dreams", "int", "dataset"),
                  float(_need(terms, "dreams_keep_rate", "number", "dataset")))


def _dreamt(by: Steps, status: str, stopped: str | None) -> Dream | Failed | NotRun:
    """The dream stage: the questions tonight opened, the core's dream when it dreamt (a nap's CPU dream
    too; one that refused her says why), and Glimmer's dreams from it."""
    told: Dream | Failed | NotRun = _stage("dream", by, status, stopped, _dream)
    core = by.get(CORE_DREAM)
    if isinstance(told, Failed):
        return told
    if core is not None and core.get("status") == "failed":
        return _failed(core)
    dreams = _glimmer_dreams(by, status, stopped)
    if core is None or core.get("status") == "skipped":
        return replace(told, dreams=dreams) if isinstance(told, Dream) else told
    questions = told.questions if isinstance(told, Dream) else told
    if core.get("noop"):  # the core refused her: its reason in the core's places
        why = NotRun(str(core.get("why") or CORE_OFF))
        return Dream(questions, why, why, why, dreams)
    return Dream(questions, *_core(core), dreams)


def _critic_ran(practice: Mapping[str, Any]) -> bool:
    spark = practice.get("spark")
    critic = spark.get("critic") if isinstance(spark, Mapping) else None
    return isinstance(critic, Mapping) and isinstance(critic.get("calls"), int) and critic["calls"] > 0


def _seeded(by: Steps) -> Seeded | NotRun:
    """What practice was seeded on: the dream's picks, as its record says (``core_steps._seed_record``)."""
    core = by.get(CORE_DREAM)
    if core is None or core.get("status") != "ok":
        return NotRun(CORE_OFF)
    if core.get("noop"):
        return NotRun(str(core.get("why") or CORE_OFF))
    if core.get("picks") is None:
        return NotRun(NO_CURRICULUM)
    said = by["practice"].get("seeded")
    if not isinstance(said, Mapping):
        return NotRun(SEEDING_UNLOGGED)
    picked = _need(said, "picked", "int", "practice")
    try:
        threads = tuple(int(slot) for slot in said["threads"])
    except (KeyError, TypeError, ValueError):
        raise StoryError("the practice step's seeded record has no readable threads") from None
    return Seeded(threads, picked) if picked else NotRun(FELL_BACK)


def _rehearse(by: Steps, night: date) -> Rehearse | Reused:
    practice = by["practice"]
    scenes = _need(practice, "scenes", "int")
    if practice.get("reused"):  # whose practice it was; a log from before it said so reused tonight's own
        source = practice.get("reused_from")
        return Reused(night if source is None else _scalar(date, source, "the practice step's reused_from"), scenes)
    if not _critic_ran(practice):
        why = NotRun(NO_CRITIC)
        return Rehearse(scenes, _seeded(by), why, why)
    return Rehearse(scenes, _seeded(by), _need(practice, "pairs", "int"), _need(practice, "teacher_targets", "int"))


def _channel(train: Mapping[str, Any]) -> tuple[float | NotRun, str | NotRun]:
    """The thought channel's loss and its projector's stamp, from the training step's receipt
    (``channel_night.train``): the projector file it kept and the job's ``channel`` record. The stamp
    is the projector's id, else its file's name, never its folder."""
    said = train.get("receipt")
    receipt: Mapping[str, Any] = said if isinstance(said, Mapping) else {}
    kept, channel = receipt.get("projector"), receipt.get("channel")
    channel = channel if isinstance(channel, Mapping) else {}
    if not (isinstance(kept, str) and kept) and not channel:
        return NotRun(CHANNEL_OFF), NotRun(CHANNEL_OFF)
    try:
        loss: float | NotRun = _number(channel.get("loss_last"))
    except ValueError:
        loss = NotRun(NO_CHANNEL_LOSS)
    sha = channel.get("projector")
    stamp = (sha[:STAMP_CHARS] if isinstance(sha, str) and sha else
             PurePosixPath(kept).name.split(".")[0] if isinstance(kept, str) and kept else NotRun(NO_STAMP))
    return loss, stamp


def _surprise(by: Steps) -> Surprise | NotRun:
    """Whether the loss took MIND's surprise weights (the training step's ``surprise_weights``, logged only
    when a row carried one) and how many rows did (the dataset step's ``mind_weighted``)."""
    said = by["train"].get("surprise_weights")
    if not isinstance(said, str):
        return NotRun(NO_SURPRISE)
    return Surprise(said == APPLIED, _need(by["dataset"], "mind_weighted", "int"))


def _reported(value: Any) -> float | NotRun:
    try:
        return _number(value)
    except ValueError:
        return NotRun(NOT_REPORTED)


def _rows(rows: Mapping[str, Any], term: str) -> int:
    value = rows.get(term)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _protect(raw: Any) -> Protect | NotRun:
    if not isinstance(raw, Mapping):
        return NotRun(PROTECT_OFF)
    try:
        return Protect(str(raw["basis"]), int(raw["rank"]), _number(raw["minutes"]))
    except (KeyError, TypeError, ValueError) as error:
        raise StoryError(f"the train step's protect record has no readable {error}") from None


def _terms(terms: Mapping[str, Any]) -> tuple[Seeding | NotRun, Imitation | NotRun, int | NotRun, Protect | NotRun]:
    """The sleep objective's terms from the training job's receipt (``sleep_loss.train_terms``, with the
    fold's and the protected slot's): rows and minutes by term, the seeding KL, the imitation reward."""
    rows, minutes = terms.get("rows"), terms.get("minutes")
    rows, minutes = (rows if isinstance(rows, Mapping) else {}), (minutes if isinstance(minutes, Mapping) else {})
    seeds, imitating, folded = _rows(rows, "seed"), _rows(rows, "imitate"), terms.get("folded_to")
    seeding: Seeding | NotRun = (Seeding(seeds, _reported(terms.get("seed_kl_before")),
                                         _reported(terms.get("seed_kl_after")), _reported(minutes.get("seed")))
                                 if seeds else NotRun(NO_SEEDS))
    imitation: Imitation | NotRun = (Imitation(imitating, _reported(terms.get("imitation_reward")),
                                               _reported(minutes.get("imitate"))) if imitating else NotRun(NO_IMITATION))
    fold: int | NotRun = folded if isinstance(folded, int) and not isinstance(folded, bool) else NotRun(FOLD_OFF)
    return seeding, imitation, fold, _protect(terms.get("protect"))


def _learn(by: Steps) -> Learn:
    train = by["train"]
    channel_loss, stamp = _channel(train)
    said = train.get("receipt")
    terms = said.get("terms") if isinstance(said, Mapping) else None
    off = NotRun(TERMS_OFF)
    seeding, imitation, fold, protect = _terms(terms) if isinstance(terms, Mapping) else (off, off, off, off)
    harness = by.get("harness")
    learned = (harness["learned"],) if (
        isinstance(harness, Mapping) and harness.get("status") == "ok"
        and isinstance(harness.get("learned"), str) and harness["learned"]
    ) else ()
    return Learn(_need(by["dataset"], "examples", "int"), _surprise(by), channel_loss, stamp, seeding, imitation, fold,
                 protect, NotRun(ADAPTATION_NOT_YET), learned=learned)


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


def _stops(record: Mapping[str, Any]) -> bool:
    """A failed step stopped the night, unless it was logged once the night ended or kept what it had."""
    return record.get("status") == "failed" and record.get("step") not in AFTER_THE_NIGHT and KEPT not in record


def tell(night: date, status: str, steps: Sequence[Mapping[str, Any]]) -> Story:
    """The night's story from its log records (``NightLog.steps()``), the latest record of each step."""
    if status not in STATUSES:
        raise StoryError(f"no night ends {status!r}")
    by = {str(record.get("step")): record for record in steps}
    stopped = next((str(r["step"]) for r in steps if _stops(r)), None)
    builders: Mapping[str, Callable[[Steps], Any]] = {
        "rehearse": lambda told: _rehearse(told, night), "learn": _learn, "check": _check}
    stages = {name: _stage(name, by, status, stopped, build) for name, build in builders.items()}
    return Story(night=night, status=cast(NightStatus, status), sleep=_sleep(by), brief=_briefed(by, status, stopped),
                 dream=_dreamt(by, status, stopped), wake=_wake(by, status, stopped), **stages)
