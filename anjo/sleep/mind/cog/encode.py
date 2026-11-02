"""Tokens to ids and lives to padded arrays: what the cog core reads (``core``) and trains on.

A ``Vocab`` is built over train lives (every token of their events, queries, questions, options and
names); a token it never saw maps to the shared ``UNK`` id. An event is its tokens, its kind and
three value features: whether it has a value, the value itself (scaled) for kinds whose value is a
level, and how many days ahead it points for kinds whose value is a day (``plan``, ``cancel``).
An event also points at a day (``plan``): a plan's or cancel's value, or else a token ``in=k`` (the
day ``k`` after the event, as an answer "exam in=2" says), and may cover a span (``span``): a token
``left=k`` covers the event's day through ``k`` days after it (a visitor here, an injury).

The outcome of an activity on a day is read from that day's events the same way in training and at
test time: a ``did`` event for it means it happened; ``skipped`` or no event means it didn't.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import torch
from torch import Tensor

from anjo.sleep.mind.cog.types import EVENT_KINDS, Event, Life, Query, Question

PAD: Final = 0
UNK: Final = 1
KIND_INDEX: Final = {k: i for i, k in enumerate(EVENT_KINDS)}
DAY_KINDS: Final = frozenset({"plan", "cancel"})  # kinds whose value is a day
MAX_TOKENS: Final = 12  # tokens kept per event, query, question or option
VALUE_SCALE: Final = 5.0
NO_PLAN: Final = -1.0
_IN = re.compile(r"^in=(\d+)$")
_LEFT = re.compile(r"^left=(\d+)$")


@dataclass(frozen=True)
class Vocab:
    tokens: tuple[str, ...]  # index 0 is PAD, 1 is UNK
    index: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.tokens[:2] != ("<pad>", "<unk>"):
            raise ValueError("a vocabulary starts with <pad> and <unk>")
        object.__setattr__(self, "index", {t: i for i, t in enumerate(self.tokens)})

    def __len__(self) -> int:
        return len(self.tokens)

    def id(self, token: str) -> int:
        return self.index.get(token, UNK)

    def ids(self, tokens: Sequence[str]) -> list[int]:
        return [self.id(t) for t in tokens[:MAX_TOKENS]]

    def to_json(self) -> str:
        return json.dumps(list(self.tokens))

    @staticmethod
    def from_json(text: str) -> Vocab:
        return Vocab(tuple(json.loads(text)))


def _life_tokens(life: Life) -> Iterable[str]:
    meta = life.meta
    yield from (*meta.activities, *meta.people, *meta.topics)
    for d in range(meta.n_days):
        for e in life.events(d):
            yield e.subject
            yield from e.tokens
        for q in life.queries(d):
            yield q.subject
            yield from q.tokens
        for qu in life.questions(d):
            yield qu.subject
            yield from qu.tokens
            for opt in qu.options:
                yield from opt


def build_vocab(lives: Iterable[Life]) -> Vocab:
    seen: dict[str, None] = {}
    for life in lives:
        for t in _life_tokens(life):
            seen.setdefault(t, None)
    return Vocab(("<pad>", "<unk>", *sorted(seen)))


# ── one day, one batch of queries, one question (numpy) ───────────────────────


@dataclass(frozen=True)
class DayArrays:
    tok: np.ndarray  # (E, T) int64
    kind: np.ndarray  # (E,) int64
    val: np.ndarray  # (E, 3) float32
    subj: np.ndarray  # (E,) int64
    plan: np.ndarray  # (E,) float32, the day the event points at, NO_PLAN otherwise
    span: np.ndarray  # (E,) float32, the last day a ``left=k`` event covers, NO_PLAN otherwise
    answer: np.ndarray  # (E,) bool, an answer to a question asked today


def _pad_tokens(rows: Sequence[Sequence[int]], width: int | None = None) -> np.ndarray:
    width = max([1, *(len(r) for r in rows)]) if width is None else width
    out = np.zeros((len(rows), width), dtype=np.int64)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r[:width]
    return out


def _token_days(e: Event) -> tuple[float, float]:
    """The day an ``in=k`` token points at and the last day a ``left=k`` token covers."""
    point = span = NO_PLAN
    for t in e.tokens:
        m = _IN.match(t)
        if m:
            point = float(e.day + int(m.group(1)))
        m = _LEFT.match(t)
        if m:
            span = float(e.day + int(m.group(1)))
    return point, span


def _value_features(e: Event) -> tuple[float, float, float, float]:
    point, _ = _token_days(e)
    if e.value is None:
        return 0.0, 0.0, 0.0 if point == NO_PLAN else (point - e.day) / 7.0, point
    v = float(e.value)
    if e.kind in DAY_KINDS:
        return 1.0, 0.0, (v - e.day) / 7.0, float(round(v))
    return 1.0, float(np.clip(v / VALUE_SCALE, -2.0, 2.0)), 0.0, point


def encode_day(vocab: Vocab, events: Sequence[Event]) -> DayArrays:
    vals = [_value_features(e) for e in events]
    return DayArrays(
        tok=_pad_tokens([vocab.ids(e.tokens) for e in events]),
        kind=np.array([KIND_INDEX[e.kind] for e in events], dtype=np.int64),
        val=np.array([v[:3] for v in vals], dtype=np.float32).reshape(len(events), 3),
        subj=np.array([vocab.id(e.subject) for e in events], dtype=np.int64),
        plan=np.array([v[3] for v in vals], dtype=np.float32),
        span=np.array([_token_days(e)[1] for e in events], dtype=np.float32),
        answer=np.array([e.kind == "answer" for e in events], dtype=bool),
    )


def outcomes(events: Sequence[Event], subjects: Sequence[str]) -> np.ndarray:
    """1 where a ``did`` event for the subject is among the events, else 0 (skipped or absent)."""
    did = {e.subject for e in events if e.kind == "did"}
    return np.array([1.0 if s in did else 0.0 for s in subjects], dtype=np.float32)


@dataclass(frozen=True)
class QueryArrays:
    tok: np.ndarray  # (Q, T)
    subj: np.ndarray  # (Q,)
    target: np.ndarray  # (Q,) the target day
    asked: np.ndarray  # (Q,) the asked day


def encode_queries(vocab: Vocab, queries: Sequence[Query]) -> QueryArrays:
    return QueryArrays(
        tok=_pad_tokens([vocab.ids(q.tokens) for q in queries]),
        subj=np.array([vocab.id(q.subject) for q in queries], dtype=np.int64),
        target=np.array([q.target_day for q in queries], dtype=np.int64),
        asked=np.array([q.asked_day for q in queries], dtype=np.int64),
    )


@dataclass(frozen=True)
class QuestionArrays:
    tok: np.ndarray  # (T,)
    subj: int
    options: np.ndarray  # (O, T)


def encode_question(vocab: Vocab, question: Question) -> QuestionArrays:
    return QuestionArrays(
        tok=_pad_tokens([vocab.ids(question.tokens)])[0], subj=vocab.id(question.subject),
        options=_pad_tokens([vocab.ids(o) for o in question.options]),
    )


# ── a whole life for training ─────────────────────────────────────────────────


@dataclass(frozen=True)
class LifeArrays:
    n_days: int
    start_weekday: int
    days: tuple[DayArrays, ...]  # each day's events, the answer to a random ask included
    act: np.ndarray  # (A,) the activities' ids
    probe_y: np.ndarray  # (D, A) each activity's outcome on each day
    queries: tuple[QueryArrays, ...]
    y: tuple[np.ndarray, ...]  # per day, (Q,) outcomes
    oracle: tuple[np.ndarray, ...]  # per day, (Q,) oracle_p
    asks: tuple[tuple[QuestionArrays, int] | None, ...]  # per day, the question asked and the answer's index
    asking: bool = True  # whether this life's days could have asks at all


def encode_life(
    vocab: Vocab, life: Life, rng: np.random.Generator, ask_rate: float = 0.2, asking: bool = True,
) -> LifeArrays:
    """The life as arrays, with a random question asked on about ``ask_rate`` of days (its answer is
    the last event of that day) when ``asking``."""
    meta = life.meta
    if not asking:
        ask_rate = 0.0
    days, probe, qs, ys, ors, asks = [], [], [], [], [], []
    for d in range(meta.n_days):
        events = list(life.events(d))
        ask: tuple[QuestionArrays, int] | None = None
        offered = [q for q in life.questions(d) if q.options]
        if offered and rng.random() < ask_rate:
            question = offered[int(rng.integers(len(offered)))]
            reply = life.answer(question.question_id)
            index = question.options.index(tuple(reply.tokens))
            events.append(reply)
            ask = (encode_question(vocab, question), index)
        days.append(encode_day(vocab, events))
        probe.append(outcomes(events, meta.activities))
        queries = life.queries(d)
        qs.append(encode_queries(vocab, queries))
        res = [life.resolve(q.qid) for q in queries]
        ys.append(np.array([r.outcome for r in res], dtype=np.float32))
        ors.append(np.array([r.oracle_p for r in res], dtype=np.float32))
        asks.append(ask)
    return LifeArrays(meta.n_days, meta.start_weekday, tuple(days),
                      np.array([vocab.id(a) for a in meta.activities], dtype=np.int64),
                      np.stack(probe), tuple(qs), tuple(ys), tuple(ors), tuple(asks), asking)


@dataclass(frozen=True)
class Batch:
    """Lives padded together; every tensor leads with B, then the day."""

    n_days: Tensor  # (B,)
    start_weekday: Tensor  # (B,)
    ev_tok: Tensor  # (B, D, E, T)
    ev_kind: Tensor  # (B, D, E)
    ev_val: Tensor  # (B, D, E, 3)
    ev_subj: Tensor  # (B, D, E)
    ev_plan: Tensor  # (B, D, E)
    ev_span: Tensor  # (B, D, E)
    ev_answer: Tensor  # (B, D, E) bool
    ev_mask: Tensor  # (B, D, E) bool
    act: Tensor  # (B, A)
    act_mask: Tensor  # (B, A) bool
    probe_y: Tensor  # (B, D, A)
    q_tok: Tensor  # (B, D, Q, T)
    q_subj: Tensor  # (B, D, Q)
    q_target: Tensor  # (B, D, Q)
    q_mask: Tensor  # (B, D, Q) bool
    q_y: Tensor  # (B, D, Q)
    q_oracle: Tensor  # (B, D, Q)
    ask_mask: Tensor  # (B, D) bool
    ask_tok: Tensor  # (B, D, T)
    ask_subj: Tensor  # (B, D)
    ask_opt: Tensor  # (B, D, O, T)
    ask_opt_mask: Tensor  # (B, D, O) bool
    ask_answer: Tensor  # (B, D)
    asking: Tensor  # (B,) bool, lives whose days could have asks


def collate(lives: Sequence[LifeArrays]) -> Batch:
    b = len(lives)
    dmax = max(x.n_days for x in lives)
    emax = max([1, *(len(day.kind) for x in lives for day in x.days)])
    tmax = max([1, *(day.tok.shape[1] for x in lives for day in x.days)])
    amax = max(len(x.act) for x in lives)
    qmax = max([1, *(len(q.subj) for x in lives for q in x.queries)])
    qtmax = max([1, *(q.tok.shape[1] for x in lives for q in x.queries)])
    asks = [a for x in lives for a in x.asks if a is not None]
    omax = max([1, *(a[0].options.shape[0] for a in asks)])
    atmax = max([1, *(max(a[0].tok.shape[0], a[0].options.shape[1]) for a in asks)])
    z = np.zeros
    ev_tok, ev_kind = z((b, dmax, emax, tmax), np.int64), z((b, dmax, emax), np.int64)
    ev_val, ev_subj = z((b, dmax, emax, 3), np.float32), z((b, dmax, emax), np.int64)
    ev_plan = np.full((b, dmax, emax), NO_PLAN, np.float32)
    ev_span = np.full((b, dmax, emax), NO_PLAN, np.float32)
    ev_answer, ev_mask = z((b, dmax, emax), bool), z((b, dmax, emax), bool)
    act, act_mask, probe_y = z((b, amax), np.int64), z((b, amax), bool), z((b, dmax, amax), np.float32)
    q_tok, q_subj = z((b, dmax, qmax, qtmax), np.int64), z((b, dmax, qmax), np.int64)
    q_target, q_mask = z((b, dmax, qmax), np.int64), z((b, dmax, qmax), bool)
    q_y, q_oracle = z((b, dmax, qmax), np.float32), np.full((b, dmax, qmax), 0.5, np.float32)
    ask_mask, ask_tok, ask_subj = z((b, dmax), bool), z((b, dmax, atmax), np.int64), z((b, dmax), np.int64)
    ask_opt, ask_opt_mask = z((b, dmax, omax, atmax), np.int64), z((b, dmax, omax), bool)
    ask_answer = z((b, dmax), np.int64)
    for i, x in enumerate(lives):
        na = len(x.act)
        act[i, :na], act_mask[i, :na] = x.act, True
        probe_y[i, : x.n_days, :na] = x.probe_y
        for d, day in enumerate(x.days):
            ne, nt = len(day.kind), day.tok.shape[1]
            ev_tok[i, d, :ne, :nt] = day.tok
            ev_kind[i, d, :ne], ev_val[i, d, :ne] = day.kind, day.val
            ev_subj[i, d, :ne], ev_plan[i, d, :ne], ev_span[i, d, :ne] = day.subj, day.plan, day.span
            ev_answer[i, d, :ne], ev_mask[i, d, :ne] = day.answer, True
            q = x.queries[d]
            nq, nqt = len(q.subj), q.tok.shape[1]
            q_tok[i, d, :nq, :nqt], q_subj[i, d, :nq] = q.tok, q.subj
            q_target[i, d, :nq], q_mask[i, d, :nq] = q.target, True
            q_y[i, d, :nq], q_oracle[i, d, :nq] = x.y[d], x.oracle[d]
            ask = x.asks[d]
            if ask is not None:
                qa, index = ask
                no, nto = qa.options.shape
                ask_mask[i, d] = True
                ask_tok[i, d, : len(qa.tok)], ask_subj[i, d] = qa.tok, qa.subj
                ask_opt[i, d, :no, :nto], ask_opt_mask[i, d, :no] = qa.options, True
                ask_answer[i, d] = index
    t = torch.from_numpy
    return Batch(
        n_days=torch.tensor([x.n_days for x in lives]), start_weekday=torch.tensor([x.start_weekday for x in lives]),
        ev_tok=t(ev_tok), ev_kind=t(ev_kind), ev_val=t(ev_val), ev_subj=t(ev_subj), ev_plan=t(ev_plan), ev_span=t(ev_span),
        ev_answer=t(ev_answer), ev_mask=t(ev_mask), act=t(act), act_mask=t(act_mask), probe_y=t(probe_y),
        q_tok=t(q_tok), q_subj=t(q_subj), q_target=t(q_target), q_mask=t(q_mask), q_y=t(q_y),
        q_oracle=t(q_oracle), ask_mask=t(ask_mask), ask_tok=t(ask_tok), ask_subj=t(ask_subj),
        ask_opt=t(ask_opt), ask_opt_mask=t(ask_opt_mask), ask_answer=t(ask_answer),
        asking=torch.tensor([x.asking for x in lives], dtype=torch.bool),
    )
