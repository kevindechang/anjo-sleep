"""The Gemini baseline of MIND's cognition sprint, built against an injected client.

``complete`` is any ``Callable[[str], str]`` (a prompt in, the model's text out). Nothing here
reaches a network: the caller supplies the client, and the tests supply fakes.

- ``GeminiPredictor`` builds ONE prompt per (life, day): the person's whole history so far as dated
  lines of their own words ("Day 12 (Wed): <event text>", text only, never tokens), then the day's
  queries with their ids, asking for a JSON object {qid: probability}. With ``stats=True`` the
  prompt also carries a table counted from the same events up to the asked day (done / skipped or
  cancelled per activity and weekday, each activity's last 7 outcomes, the plan kept-rate, late
  messages and mean mood over 3 and 7 days), so a loss cannot be blamed on the model not counting.
  Answers are read leniently (fenced or bare JSON, percentages), clamped to [0.01, 0.99]; an id
  left out gets 0.5 and is counted. ``platt=(a, b)`` maps p to sigmoid(a * logit(p) + b), the
  recalibrated Gemini of DESIGN.md, fitted on dev records by ``fit_platt``.
- ``GeminiInquirer`` shows the history, the day's questions (their costs and possible answers)
  and the day's queries, and asks which one question, if any, would most improve its predictions
  of the next week: {"ask": "<id>"} or {"ask": null}.

Responses are cached in ``cache_dir/<model_label>/<sha256 of the prompt>.json``, so a rerun (and
the Platt-recalibrated rerun, whose prompts are the same) is free. ``usage`` counts the calls, the
cache hits and the characters sent and received; ``CHARS_PER_TOKEN`` turns them into a rough
token estimate.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from anjo.sleep.mind.cog.rivals import PersonHistory
from anjo.sleep.mind.cog.run import LifeRecord
from anjo.sleep.mind.cog.types import Event, LifeMeta, Prediction, Query, Question

Complete = Callable[[str], str]
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
P_LOW, P_HIGH = 0.01, 0.99
MISSING_P = 0.5
CHARS_PER_TOKEN = 4.0  # a rough rule for English prose; good enough to budget calls
LAST_OUTCOMES = 7

PREDICT_PROMPT = """You are forecasting what one person will do in the coming days, from what they have told you.

Everything they have said so far, oldest first, one line per message:
{history}
{stats}
Today is Day {day} ({weekday}). For each question below, give the probability, from 0 to 1, that the answer is yes.
{queries}

Answer with only a JSON object that maps every question id to its probability, like {{"{example}": 0.5}}.
"""

STATS_BLOCK = """
The same history, counted for you (only what they said up to today):
{table}
"""

INQUIRE_PROMPT = """You are getting to know one person so that you can predict what they will do in the coming days.

Everything they have said so far, oldest first, one line per message:
{history}

Today is Day {day} ({weekday}). You may ask them at most one question today. Each question has a cost: how much it interrupts them. Ask only if the answer would improve your predictions for the next week by more than its cost.

Questions you could ask:
{questions}

Questions you will be asked to predict today:
{queries}

Answer with only a JSON object: {{"ask": "<question id>"}} to ask one, or {{"ask": null}} to stay silent.
"""


# -- reading the model's answer --
def extract_json(text: str) -> Any | None:
    """The first JSON object in ``text``: in a fenced block, the whole text, or a {...} span."""
    candidates = [m.group(1) for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)]
    candidates.append(text.strip())
    if "{" in text and "}" in text:
        candidates.append(text[text.index("{"): text.rindex("}") + 1])
    for start in (i for i, c in enumerate(text) if c == "{"):
        depth = 0
        for end in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[end], 0)
            if depth == 0:
                candidates.append(text[start: end + 1])
                break
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _as_probability(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    percent = False
    if isinstance(value, str):
        raw = value.strip()
        percent = raw.endswith("%")
        try:
            value = float(raw.rstrip("%").strip())
        except ValueError:
            return None
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    p = float(value) / 100.0 if percent or 1.0 < value <= 100.0 else float(value)
    return min(max(p, P_LOW), P_HIGH) if 0.0 <= p <= 1.0 else None


def parse_probabilities(text: str, qids: Sequence[str]) -> tuple[dict[str, float], int, bool]:
    """({qid: p}, how many ids got the 0.5 fallback, whether the reply had no JSON object)."""
    obj = extract_json(text)
    if not isinstance(obj, dict):
        return {q: MISSING_P for q in qids}, len(qids), True
    out, missing = {}, 0
    for q in qids:
        p = _as_probability(obj.get(q))
        if p is None:
            p, missing = MISSING_P, missing + 1
        out[q] = p
    return out, missing, False


# -- Platt scaling --
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    out: np.ndarray = np.log(p / (1 - p))
    return out


def apply_platt(p: float, platt: tuple[float, float]) -> float:
    a, b = platt
    x = a * float(_logit(np.array([p]))[0]) + b
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def fit_platt(records: Sequence[LifeRecord], *, l2: float = 1e-4, iters: int = 100) -> tuple[float, float]:
    """(a, b) maximizing the likelihood of outcome ~ sigmoid(a * logit(p) + b) over the records'
    queries (fit on dev records only)."""
    p = np.array([q.p for r in records for q in r.queries], dtype=float)
    y = np.array([q.outcome for r in records for q in r.queries], dtype=float)
    if len(p) == 0:
        raise ValueError("fit_platt needs records with queries")
    x = np.column_stack([_logit(p), np.ones(len(p))])
    w = np.array([1.0, 0.0])
    for _ in range(iters):
        mu = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))
        grad = x.T @ (mu - y) + l2 * w
        hess = (x * (mu * (1 - mu))[:, None]).T @ x + l2 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        w = w - step
        if np.max(np.abs(step)) < 1e-9:
            break
    return float(w[0]), float(w[1])


# -- the history as the person said it --
class Transcript:
    """The person's own words so far, as dated lines, and the same events counted."""

    def __init__(self, meta: LifeMeta) -> None:
        self.meta = meta
        self.lines: list[tuple[int, str]] = []
        self.hist = PersonHistory(meta)

    def weekday(self, day: int) -> str:
        return WEEKDAYS[(self.meta.start_weekday + day) % 7]

    def observe(self, events: Sequence[Event]) -> None:
        self.hist.add(events)
        self.lines.extend((e.day, f"Day {e.day} ({self.weekday(e.day)}): {e.text}") for e in events)

    def history(self, as_of: int) -> str:
        lines = [line for day, line in self.lines if day <= as_of]
        return "\n".join(lines) if lines else "(nothing yet)"

    def stats(self, as_of: int) -> str:
        return stats_table(self.hist, as_of)


def stats_table(hist: PersonHistory, as_of: int) -> str:
    """A compact table counted from ``hist``'s events up to ``as_of`` (inclusive)."""
    start = hist.meta.start_weekday
    width = max([len(a) for a in hist.activities] + [8])
    rows = [f"Done / skipped or cancelled, by weekday (Day 0 to Day {as_of}):",
            " " * width + "  " + "  ".join(f"{d:>5}" for d in WEEKDAYS)]
    for a in hist.activities:
        cells = []
        for wd in range(7):
            days = range((wd - start) % 7, as_of + 1, 7)
            done = sum(a in hist.did.get(j, ()) for j in days)
            missed = sum(a not in hist.did.get(j, ()) and hist.outcome(a, j, absent_is_negative=False) == 0
                         for j in days)
            cells.append(f"{done}/{missed}")
        rows.append(f"{a:<{width}}  " + "  ".join(f"{c:>5}" for c in cells))
    rows.append(f"Last {LAST_OUTCOMES} days with news of each activity, oldest first (+ done, - not):")
    for a in hist.activities:
        outcomes = hist.last_outcomes(a, as_of, LAST_OUTCOMES)
        rows.append(f"{a:<{width}}  " + (" ".join("+" if y else "-" for y in outcomes) or "(none)"))
    kept, due = hist.kept(as_of)
    rows.append(f"Plans kept: {kept} of {due} that came due.")
    rows.append(f"Late-night messages: {hist.total(hist.late, as_of, 3)} in the last 3 days, "
                f"{hist.total(hist.late, as_of, 7)} in the last 7.")
    m3, m7 = hist.mood_mean(as_of, 3), hist.mood_mean(as_of, 7)
    rows.append("Mean mood level: " + ", ".join(
        f"{'none' if m is None else f'{m:.1f}'} over the last {k} days" for m, k in ((m3, 3), (m7, 7))) + ".")
    return "\n".join(rows)


# -- the cached client --
class CachedClient:
    """``complete`` behind a prompt-hash cache, counting what it costs."""

    def __init__(self, complete: Complete, cache_dir: Path | None, model_label: str) -> None:
        self.complete, self.model_label = complete, model_label
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model_label) or "model"
        self.dir = None if cache_dir is None else Path(cache_dir) / safe
        self.memory: dict[str, str] = {}
        self.usage: dict[str, float] = {"calls": 0.0, "cache_hits": 0.0, "prompt_chars": 0.0,
                                        "response_chars": 0.0, "prompt_chars_total": 0.0}

    def __call__(self, prompt: str) -> str:
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        self.usage["prompt_chars_total"] += len(prompt)
        cached = self._cached(key)
        if cached is not None:
            self.usage["cache_hits"] += 1
            return cached
        response = self.complete(prompt)
        if not isinstance(response, str):
            raise TypeError(f"the client returned {type(response).__name__}, not text")
        self.usage["calls"] += 1
        self.usage["prompt_chars"] += len(prompt)
        self.usage["response_chars"] += len(response)
        self._store(key, response)
        return response

    def _cached(self, key: str) -> str | None:
        if key in self.memory:
            return self.memory[key]
        if self.dir is None:
            return None
        path = self.dir / f"{key}.json"
        if not path.exists():
            return None
        response = json.loads(path.read_text(encoding="utf-8")).get("response")
        if not isinstance(response, str):
            raise ValueError(f"cache file {path} holds no text response; delete it to re-ask")
        self.memory[key] = response
        return response

    def _store(self, key: str, response: str) -> None:
        self.memory[key] = response
        if self.dir is None:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f".{key}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps({"model": self.model_label, "prompt_sha256": key, "response": response}),
                       encoding="utf-8")
        tmp.replace(self.dir / f"{key}.json")


def estimated_tokens(chars: float) -> float:
    return chars / CHARS_PER_TOKEN


# -- the predictor and the inquirer --
class GeminiPredictor:
    def __init__(self, complete: Complete, *, cache_dir: Path | None, model_label: str,
                 stats: bool = False, platt: tuple[float, float] | None = None,
                 name: str | None = None) -> None:
        self.client = CachedClient(complete, cache_dir, model_label)
        self.stats, self.platt = stats, platt
        self.name = name or "gemini" + ("_stats" if stats else "") + ("_platt" if platt else "")
        self.usage: dict[str, float] = dict(self.client.usage)

    def begin(self, meta: LifeMeta) -> None:
        self.transcript = Transcript(meta)
        self.client.usage = {k: 0.0 for k in self.client.usage}
        self._counts = {"queries": 0.0, "missing_ids": 0.0, "parse_failures": 0.0}
        self._sync()

    def observe(self, day: int, events: Sequence[Event]) -> None:
        self.transcript.observe(events)

    def prompt(self, queries: Sequence[Query]) -> str:
        day = max(q.asked_day for q in queries)
        t = self.transcript
        lines = "\n".join(f"- {q.qid} (about Day {q.target_day}, {t.weekday(q.target_day)}): {q.text}"
                          for q in queries)
        stats = STATS_BLOCK.format(table=t.stats(day)) if self.stats else ""
        return PREDICT_PROMPT.format(history=t.history(day), stats=stats, day=day,
                                     weekday=t.weekday(day), queries=lines, example=queries[0].qid)

    def predict(self, queries: Sequence[Query]) -> list[Prediction]:
        if not queries:
            return []
        qids = [q.qid for q in queries]
        probs, missing, failed = parse_probabilities(self.client(self.prompt(queries)), qids)
        self._counts["queries"] += len(qids)
        self._counts["missing_ids"] += missing
        self._counts["parse_failures"] += int(failed)
        self._sync()
        return [Prediction(q, apply_platt(probs[q], self.platt) if self.platt else probs[q]) for q in qids]

    def end_day(self, day: int) -> None: ...

    def _sync(self) -> None:
        self.usage = {**self.client.usage, **self._counts}


class GeminiInquirer:
    name = "gemini"

    def __init__(self, complete: Complete, *, cache_dir: Path | None, model_label: str,
                 name: str | None = None) -> None:
        self.client = CachedClient(complete, cache_dir, model_label)
        if name is not None:
            self.name = name
        self.usage: dict[str, float] = dict(self.client.usage)

    def begin(self, meta: LifeMeta) -> None:
        self.transcript = Transcript(meta)
        self.client.usage = {k: 0.0 for k in self.client.usage}
        self._counts = {"asks": 0.0, "silent": 0.0, "invalid": 0.0}
        self._sync()

    def observe(self, day: int, events: Sequence[Event]) -> None:
        self.transcript.observe(events)

    def prompt(self, day: int, questions: Sequence[Question], upcoming: Sequence[Query]) -> str:
        t = self.transcript
        qs = "\n".join(
            f"- {q.question_id} (cost {q.cost:.2f}): {q.text}"
            + (f" Possible answers: {' / '.join(' '.join(opt) for opt in q.options)}" if q.options else "")
            for q in questions)
        ups = "\n".join(f"- {q.qid} (about Day {q.target_day}, {t.weekday(q.target_day)}): {q.text}"
                        for q in upcoming) or "(none)"
        return INQUIRE_PROMPT.format(history=t.history(day), day=day, weekday=t.weekday(day),
                                     questions=qs, queries=ups)

    def choose(self, day: int, questions: Sequence[Question], upcoming: Sequence[Query]) -> str | None:
        if not questions:
            return None
        obj = extract_json(self.client(self.prompt(day, questions, upcoming)))
        ask = obj.get("ask") if isinstance(obj, dict) else None
        ids = {q.question_id for q in questions}
        if isinstance(ask, str) and ask in ids:
            self._counts["asks"] += 1
            chosen: str | None = str(ask)
        else:
            self._counts["silent" if ask is None and isinstance(obj, dict) else "invalid"] += 1
            chosen = None
        self._sync()
        return chosen

    def _sync(self) -> None:
        self.usage = {**self.client.usage, **self._counts}
