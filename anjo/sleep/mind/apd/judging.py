"""Judging the APD runs (Task 17): Muse Spark on ``judge_apd_v1.md``, blind, Gemini's rescoring,
and Claude's blind audit where the two differ.

- **Blind by construction:** every judge reads ``score.judge_prompt`` alone: the finding, the
  planted item (or none) and the evidence the arm had when it reported (``score._Rulings``). No
  arm and no life id reach a judge, a cache line or the audit file.
- **Each prompt asked once:** each judge's replies are kept in ``judging/<judge>.jsonl`` by the
  hash of the prompt, so judging again, or after a crash, asks only what it hasn't.
- **The audit:** where Muse Spark and Gemini differ, the prompt goes to ``audit/pending.jsonl``
  (its hash and the prompt, in hash order, so the order says nothing of the arm), for Claude to
  answer by hand in ``audit/answers.jsonl`` as ``{"id", "same_problem", "holds"}``. The audited
  view is never reported while one is unanswered (``Judged.pending``).
- **Spend:** each call is held before it runs, Muse Spark's on ``meta_usd`` and Gemini's on
  ``vertex_usd``, at the most one call reads and writes, every billed try it may make, and
  settled at what it used.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from anjo.sleep.budget import Ledger, Pool
from anjo.sleep.mind.apd import retry
from anjo.sleep.mind.apd.ceiling import Ceiling
from anjo.sleep.mind.apd.score import (
    ApdJudge,
    AuditedJudge,
    PromptJudge,
    Ruling,
    ScoreSheet,
    judge_prompt,
    parse_ruling,
    score_life,
    score_sheet,
)
from anjo.sleep.mind.apd.store import DataDir, write_text
from anjo.sleep.mind.apd.types import ArmRun, FindingReport, Life, LifeScore, PlantedItem
from anjo.sleep.model_api import ModelApiError
from anjo.sleep.night.holds import SPARK_MAX_IN_TOKENS, SPARK_MAX_OUT_TOKENS, held, usd
from anjo.sleep.types import Evidence

TextModel = Callable[[str], tuple[str, int, int]]  # a prompt -> the reply, its tokens in and out
MUSE: Final = "muse-spark"
GEMINI: Final = "gemini"
POOLS: Final[Mapping[str, Pool]] = {MUSE: "meta_usd", GEMINI: "vertex_usd"}
MAX_IN_TOKENS: Final = SPARK_MAX_IN_TOKENS
MAX_OUT_TOKENS: Final = SPARK_MAX_OUT_TOKENS


def prompt_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class Reply(tuple[str, int, int]):
    """A judge's reply and its tokens; ``failed`` counts the tries before it that failed on
    something other than a 429, each possibly billed at a call's worst case (``_held``)."""

    failed: int

    def __new__(cls, text: str, tokens_in: int, tokens_out: int, failed: int = 0) -> Reply:
        reply = super().__new__(cls, (text, tokens_in, tokens_out))
        reply.failed = failed
        return reply


def _failed_tries(error: ModelApiError) -> int:
    """The tries before a judge's failure that weren't 429s, each possibly billed
    (``backend._gemini`` sets ``failed``)."""
    return int(getattr(error, "failed", 0))


def _rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass(frozen=True)
class Rate:
    """A judge's list prices, dollars per million tokens in and out."""

    per_m_in: float
    per_m_out: float

    def usd(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.per_m_in + tokens_out * self.per_m_out) / 1e6

    @property
    def worst(self) -> float:
        """The most one call can cost."""
        return self.usd(MAX_IN_TOKENS, MAX_OUT_TOKENS)


@dataclass
class CachedJudge:
    """One judge's text model behind its reply cache, each new call held first on its pool. Only a
    reply that holds a ruling is cached: an unparsable one (cut at its cap, say) raises
    ``score.JudgeError`` after its spend is settled, so a rerun asks again rather than reading it
    back forever."""

    name: str
    model: TextModel
    path: Path
    ledger: Ledger | None = None
    rate: Rate | None = None
    ceiling: Ceiling | None = field(default=None, repr=False)  # the command's --max-usd, checked first
    calls: int = 0
    _cache: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._cache = {str(row["id"]): str(row["reply"]) for row in _rows(self.path)}
        if self.ledger is not None and self.rate is None:
            raise ValueError("a held judge is priced")

    def complete(self, prompt: str) -> str:
        key = prompt_id(prompt)
        if key in self._cache:
            return self._cache[key]
        if self.ceiling is not None:
            self.ceiling.check()  # before a new call's hold
        if self.ledger is None or self.rate is None:
            reply, _, _ = self.model(prompt)
        else:
            reply = self._held(prompt, key, self.ledger, self.rate)
        self.calls += 1
        if self.ceiling is not None:
            self.ceiling.counted()
        parse_ruling(reply)
        self._cache[key] = reply
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as cache:
            cache.write(json.dumps({"id": key, "reply": reply}) + "\n")
        return reply

    def _held(self, prompt: str, key: str, ledger: Ledger, rate: Rate) -> str:
        spent = [0.0]
        bound = usd(rate.worst * retry.ATTEMPTS)  # every billed try the call's policy allows
        with held(ledger, f"apd:judge:{self.name}:{key[:16]}", (POOLS[self.name],), bound, lambda: spent[0]):
            try:
                answer = self.model(prompt)
            except ModelApiError as failure:
                spent[0] = rate.worst * _failed_tries(failure)
                raise
            reply, tokens_in, tokens_out = answer
            failed = answer.failed if isinstance(answer, Reply) else 0
            spent[0] = rate.worst * failed + rate.usd(tokens_in, tokens_out)
        return reply

    @property
    def judge(self) -> ApdJudge:
        return PromptJudge(self.complete)


def read_answers(path: Path) -> dict[str, Ruling]:
    """Claude's audit answers, by prompt hash."""
    answers: dict[str, Ruling] = {}
    for row in _rows(path):
        same, holds = row.get("same_problem"), row.get("holds")
        if not (isinstance(same, bool) and isinstance(holds, bool)):
            raise ValueError("an audit answer is {\"id\", \"same_problem\": bool, \"holds\": bool}")
        answers[str(row["id"])] = Ruling(same_problem=same, holds=holds)
    return answers


@dataclass
class BlindAudit:
    """Claude's audit where the judges differ: the answer on file, or the prompt kept for the
    pending file and a placeholder ruling, which no report ever reads."""

    answers: Mapping[str, Ruling]
    pending: dict[str, str] = field(default_factory=dict)

    def rule(self, report: FindingReport, evidence: Sequence[Evidence], item: PlantedItem | None) -> Ruling:
        prompt = judge_prompt(report, evidence, item)
        key = prompt_id(prompt)
        if key in self.answers:
            return self.answers[key]
        self.pending[key] = prompt
        return Ruling(same_problem=False, holds=False)


@dataclass(frozen=True)
class Judged:
    """Every run's audited score sheet and Gemini rescore, and the audits still owed."""

    audited: tuple[ScoreSheet, ...]
    gemini: tuple[LifeScore, ...]
    pending: int


def judge_runs(
    data: DataDir, lives: Sequence[Life], runs: Sequence[ArmRun], muse: CachedJudge, gemini: CachedJudge,
) -> Judged:
    """Score every run under the audited judge and rescore it under Gemini; write the audit's
    pending questions, blind, for Claude."""
    audit = BlindAudit(read_answers(data.audit / "answers.jsonl"))
    audited = AuditedJudge(primary=muse.judge, second=gemini.judge, auditor=audit)
    held = {life.id: life for life in lives}
    sheets = tuple(score_sheet(held[r.life_id], data.sealed(r.life_id), r, audited, view="audited") for r in runs)
    rescored = tuple(score_life(held[r.life_id], data.sealed(r.life_id), r, gemini.judge, view="gemini") for r in runs)
    pending = data.audit / "pending.jsonl"
    write_text(pending, "".join(json.dumps({"id": key, "prompt": audit.pending[key]}) + "\n"
                                for key in sorted(audit.pending)))
    return Judged(sheets, rescored, len(audit.pending))


def answer_pending(data: DataDir, auditor: Callable[[str], str]) -> int:
    """Answer every pending audit with ``auditor`` (``--fake`` only: live, Claude answers by hand)."""
    answers = data.audit / "answers.jsonl"
    rows = [json.loads(json.dumps(row)) for row in _rows(answers)]
    for row in _rows(data.audit / "pending.jsonl"):
        verdict = json.loads(auditor(str(row["prompt"])))
        rows.append({"id": row["id"], "same_problem": verdict["same_problem"], "holds": verdict["holds"]})
    write_text(answers, "".join(json.dumps(row) + "\n" for row in rows))
    return len(rows)
