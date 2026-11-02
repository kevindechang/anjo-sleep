"""The run loop of MIND's cognition sprint: one life, one predictor, maybe one inquirer.

Each day ``d``: the predictor observes the day's events; the inquirer (if any) may ask up to
``max_asks_per_day`` of the day's questions, seeing only today's queries as ``upcoming`` (asking
for tomorrow's would read the future), and each answer is observed on day ``d``; the predictor
predicts the day's queries (only on ``query_days`` when given, for the costly Gemini baseline);
then its night runs (``end_day``). Every prediction is stored next to its resolution.

An inquirer may optionally have ``begin(meta)`` and ``observe(day, events)``; the loop calls them
like a predictor's, so an inquirer with its own view of the life (Gemini's) can keep one. A
predictor or inquirer with a ``usage`` mapping (prompt characters, calls) has it copied into the
record, the inquirer's under an ``inquirer.`` prefix.

Records are frozen and JSON round-trips them exactly. ``run_many`` runs lives in a process pool;
each life runs alone with a fresh predictor, so the records do not depend on the worker count.
"""
from __future__ import annotations

import json
import multiprocessing
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from anjo.sleep.mind.cog.types import Event, Inquirer, Life, Predictor, Query, Question, Truth

PredictorFactory = Callable[[], Predictor]
InquirerFactory = Callable[[Predictor], Inquirer]
LifeMaker = Callable[[int], Life]


@dataclass(frozen=True)
class QueryRecord:
    qid: str
    subject: str
    family: str
    asked_day: int
    target_day: int
    p: float
    u: float | None
    outcome: bool
    oracle_p: float
    state_p: float


@dataclass(frozen=True)
class AskRecord:
    day: int
    question_id: str
    subject: str
    cost: float
    is_distractor: bool  # the question's subject is one of the life's topics
    answer_tokens: tuple[str, ...]


@dataclass(frozen=True)
class LifeRecord:
    run: str  # the name the report groups by: "<predictor>" or "<predictor>+<inquirer>"
    predictor: str
    inquirer: str | None
    seed: int | None
    person_id: str
    n_days: int
    start_weekday: int
    topics: tuple[str, ...]
    shift_days: tuple[int, ...]
    shift_subjects: tuple[str, ...] | None  # subjects a shift changed, when the world says
    queries: tuple[QueryRecord, ...]
    asks: tuple[AskRecord, ...]
    usage: Mapping[str, float] = field(default_factory=dict)

    @property
    def key(self) -> tuple[int | None, str]:
        """What pairs two runs over the same life."""
        return (self.seed, self.person_id)


def run_life(life: Life, predictor: Predictor, inquirer: Inquirer | None = None, *,
             max_asks_per_day: int = 1, query_days: Collection[int] | None = None,
             seed: int | None = None, run: str | None = None) -> LifeRecord:
    meta = life.meta
    predictor.begin(meta)
    _hook(inquirer, "begin", meta)
    keep = None if query_days is None else frozenset(query_days)
    queries: list[QueryRecord] = []
    asks: list[AskRecord] = []
    for day in range(meta.n_days):
        events = tuple(life.events(day))
        predictor.observe(day, events)
        _hook(inquirer, "observe", day, events)
        todays = tuple(life.queries(day))
        if inquirer is not None:
            asks.extend(_inquire(life, predictor, inquirer, day, todays, max_asks_per_day))
        if todays and (keep is None or day in keep):
            queries.extend(_predict(life, predictor, todays))
        predictor.end_day(day)
    truth = life.truth()
    inquirer_name = None if inquirer is None else inquirer.name
    return LifeRecord(
        run=run or (predictor.name if inquirer is None else f"{predictor.name}+{inquirer.name}"),
        predictor=predictor.name, inquirer=inquirer_name, seed=seed, person_id=meta.person_id,
        n_days=meta.n_days, start_weekday=meta.start_weekday, topics=tuple(meta.topics),
        shift_days=tuple(int(d) for d in truth.shift_days), shift_subjects=shift_subjects(truth),
        queries=tuple(queries), asks=tuple(asks), usage=_usage(predictor, inquirer))


def _hook(obj: object, method: str, *args: object) -> None:
    fn = getattr(obj, method, None) if obj is not None else None
    if callable(fn):
        fn(*args)


def _inquire(life: Life, predictor: Predictor, inquirer: Inquirer, day: int,
             todays: tuple[Query, ...], max_asks: int) -> list[AskRecord]:
    open_questions: tuple[Question, ...] = tuple(life.questions(day))
    out: list[AskRecord] = []
    topics = frozenset(life.meta.topics)
    while open_questions and len(out) < max_asks:
        chosen = inquirer.choose(day, open_questions, todays)
        if chosen is None:
            break
        question = next((q for q in open_questions if q.question_id == chosen), None)
        if question is None:
            raise ValueError(f"{inquirer.name} asked {chosen!r}, not among day {day}'s questions")
        answer: Event = life.answer(chosen)
        predictor.observe(day, (answer,))
        _hook(inquirer, "observe", day, (answer,))
        out.append(AskRecord(day=day, question_id=chosen, subject=question.subject,
                             cost=float(question.cost), is_distractor=question.subject in topics,
                             answer_tokens=tuple(answer.tokens)))
        open_questions = tuple(q for q in open_questions if q.question_id != chosen)
    return out


def _predict(life: Life, predictor: Predictor, todays: tuple[Query, ...]) -> list[QueryRecord]:
    by_qid = {pred.qid: pred for pred in predictor.predict(todays)}
    out = []
    for q in todays:
        pred = by_qid.get(q.qid)
        if pred is None:
            raise ValueError(f"{predictor.name} gave no prediction for {q.qid}")
        p = float(pred.p)
        if not 0.0 < p < 1.0:
            raise ValueError(f"{predictor.name} gave {q.qid} p={p}, not a probability in (0, 1)")
        res = life.resolve(q.qid)
        out.append(QueryRecord(
            qid=q.qid, subject=q.subject, family=q.family, asked_day=q.asked_day,
            target_day=q.target_day, p=p, u=None if pred.u is None else float(pred.u),
            outcome=bool(res.outcome), oracle_p=float(res.oracle_p), state_p=float(res.state_p)))
    return out


def shift_subjects(truth: Truth) -> tuple[str, ...] | None:
    """The subjects the life's shifts changed, if ``Truth.params`` names them.

    Read from ``params["shift_subjects"]`` (strings) or ``params["shifts"]`` (mappings with a
    ``subject``, ``subjects`` or ``activity``). None when the world does not say, and then T4's
    "untouched subjects" loss is not reported.
    """
    if not truth.shift_days:
        return None
    params = truth.params or {}
    named = params.get("shift_subjects")
    if isinstance(named, (list, tuple, frozenset, set)):
        return tuple(sorted(str(s) for s in named))
    shifts = params.get("shifts")
    if isinstance(shifts, (list, tuple)):
        found: set[str] = set()
        for shift in shifts:
            if not isinstance(shift, Mapping):
                continue
            for key in ("subject", "activity"):
                if isinstance(shift.get(key), str):
                    found.add(shift[key])
            if isinstance(shift.get("subjects"), (list, tuple)):
                found.update(str(s) for s in shift["subjects"])
        return tuple(sorted(found)) if found else None
    return None


def _usage(predictor: Predictor, inquirer: Inquirer | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for prefix, obj in (("", predictor), ("inquirer.", inquirer)):
        usage = getattr(obj, "usage", None) if obj is not None else None
        if isinstance(usage, Mapping):
            out.update({f"{prefix}{k}": float(v) for k, v in usage.items()})
    return out


# -- many lives --
@dataclass(frozen=True)
class _Job:
    item: Any  # a seed or a Life
    factory: PredictorFactory
    inquirer_factory: InquirerFactory | None
    make_life: LifeMaker | None
    max_asks_per_day: int
    query_days: frozenset[int] | None
    run: str | None


def _run_job(job: _Job) -> LifeRecord:
    if isinstance(job.item, int):
        assert job.make_life is not None
        life, seed = job.make_life(job.item), job.item
    else:
        life, seed = job.item, None
    predictor = job.factory()
    inquirer = None if job.inquirer_factory is None else job.inquirer_factory(predictor)
    return run_life(life, predictor, inquirer, max_asks_per_day=job.max_asks_per_day,
                    query_days=job.query_days, seed=seed, run=job.run)


def run_many(lives_or_seeds: Iterable[int | Life], factory: PredictorFactory, *,
             inquirer_factory: InquirerFactory | None = None, make_life: LifeMaker | None = None,
             workers: int = 1, max_asks_per_day: int = 1,
             query_days: Collection[int] | None = None, run: str | None = None) -> tuple[LifeRecord, ...]:
    """Run each life with a fresh predictor (and inquirer), in input order.

    With ``workers > 1`` the lives run in a spawned process pool, so ``factory``,
    ``inquirer_factory``, ``make_life`` and any ``Life`` given must pickle (module-level
    functions, classes, ``functools.partial`` of them, or frozen dataclasses).
    """
    items = list(lives_or_seeds)
    if any(isinstance(i, int) for i in items) and make_life is None:
        raise ValueError("seeds need a make_life(seed) -> Life")
    days = None if query_days is None else frozenset(query_days)
    jobs = [_Job(i, factory, inquirer_factory, make_life, max_asks_per_day, days, run) for i in items]
    if workers <= 1 or len(jobs) <= 1:
        return tuple(_run_job(job) for job in jobs)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=context) as pool:
        return tuple(pool.map(_run_job, jobs, chunksize=1))


# -- records on disk --
def record_to_dict(record: LifeRecord) -> dict[str, Any]:
    out = asdict(record)
    out["usage"] = dict(record.usage)
    return out


def record_from_dict(raw: Mapping[str, Any]) -> LifeRecord:
    subjects = raw.get("shift_subjects")
    return LifeRecord(
        run=str(raw["run"]), predictor=str(raw["predictor"]), inquirer=raw.get("inquirer"),
        seed=raw.get("seed"), person_id=str(raw["person_id"]), n_days=int(raw["n_days"]),
        start_weekday=int(raw["start_weekday"]), topics=tuple(raw["topics"]),
        shift_days=tuple(int(d) for d in raw["shift_days"]),
        shift_subjects=None if subjects is None else tuple(subjects),
        queries=tuple(QueryRecord(**q) for q in raw["queries"]),
        asks=tuple(AskRecord(**{**a, "answer_tokens": tuple(a["answer_tokens"])}) for a in raw["asks"]),
        usage={str(k): float(v) for k, v in (raw.get("usage") or {}).items()})


def save_records(path: Path, records: Sequence[LifeRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record_to_dict(record), allow_nan=False) + "\n")


def load_records(path: Path) -> tuple[LifeRecord, ...]:
    with Path(path).open(encoding="utf-8") as fh:
        return tuple(record_from_dict(json.loads(line)) for line in fh if line.strip())
