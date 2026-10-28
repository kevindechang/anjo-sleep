"""Full practice (Task 7, from Oct 29): more scenes, a teacher candidate, a critic and pairs.

It keeps Task 5's cue-free scene prompt (``scenes_v1.md``, asked for ``SCENES_ASKED`` messages)
and the same F (``practice_f``), and runs ``scenes`` of them, 16 to 20. Each scene's message gets
four candidates on one shared prefix (F as the system message, then the message), so the server
caches it: three from her personal Glimmer (the adapter that serves her) and one from Spark Open,
the teacher. A candidate cut at max tokens is never a candidate. With no Glimmer candidate left
the scene is dropped, so the teacher alone never trains.

**Critic.** Spark Open, at temperature 0, scores the candidates blind (letters only, in an order
seeded per scene) and lists those that miss a constraint stated in F (``critic_v1.md``). The best
is the highest-scoring Glimmer candidate, the first on a tie. The teacher replaces it only when
it scores higher, all three Glimmer candidates miss a constraint and the teacher doesn't: Glimmer
wins ties. The worst is the lowest score, the teacher's on a tie. A critic reply that can't be
read keeps the first Glimmer candidate and no pair. ``teacher=False`` switches the teacher off.

**What trains.** The best candidate is the scene's target, on the pre-practice snapshot of C (or
nothing, under the dataset's dropout), and the transcript records whether it is the teacher's.
The (best, worst) pair is kept when the critic preferred one, for the ``sft_dpo`` objective. The
dataset decontaminates both sides. Selecting the teacher's reply doesn't make it on-policy.

**Side by side.** With ``concurrency`` above 1 the scenes run at once, and so do each scene's three
Glimmer candidates; its teacher runs only once one of them survived, and its critic after that, as in
series. At most ``concurrency`` calls are in flight (``parallel.Calls``), and every result is gathered
by scene and place, never by when it came back, so the run is the serial one given the same replies.
Every snapshot of C is taken first, on the calling thread. ``concurrency`` 1 runs as before: one call
at a time.

**Capped by role (§5.5).** Every Spark Open call is capped (``max_tokens``) at what its role writes
plus ``REASONING_ROOM``, the room the model takes to reason before it answers: on the hero's first
live night two brief folds cost $0.056, about 5.6k output tokens a call against a visible brief of
2,000 tokens at most, and the longest call wrote about 8.5k. What each role writes
(``VISIBLE_TOKENS``):

- the person: ``SCENES_ASKED`` messages of one to three sentences as JSON; twenty three-sentence
  messages of 25-word sentences estimate under 1,900 tokens, so 2,048;
- the critic: a score and a miss for each of at most four letters as JSON, under 30 tokens, so 256;
- the teacher: a reply, held to the bound its Glimmer rivals have (``serving.client.MAX_TOKENS``).
  Its reply is a training target, so it is kept whole or not at all: one cut at the cap is no
  candidate (as a cut Glimmer reply never is).

A cut critic gives no verdict (the scene keeps its first Glimmer candidate and no pair), and cut
scenes stop practice. ``PracticeRun.spark`` counts each role's calls, tokens and cut replies, so a
cap that bites shows in the night log.
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, Protocol

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.context.brief_store import union
from anjo.sleep.context.template import render
from anjo.sleep.parallel import Calls
from anjo.sleep.model_api import ChatResult
from anjo.sleep.practice.run import (
    NORMAL_STOPS,
    PRACTICE_F_CAP,
    PROMPT_PATH,
    SCENE_TEMPERATURE,
    SCENE_TIMEOUT_S,
    PracticeError,
    PracticeRun,
    Snapshot,
    _known,
    _scenes,
    _seed,
    has_cue,
    practice_f,
)
from anjo.sleep.practice.transcripts import Candidate, PracticeTranscript
from anjo.sleep.probes.clients import ModelClient
from anjo.sleep.provenance import GLIMMER
from anjo.sleep.serving.client import MAX_TOKENS as GLIMMER_REPLY_CAP
from anjo.sleep.types import CheckpointRef, Evidence

CRITIC_PATH: Final = Path(__file__).parent / "prompts" / "critic_v1.md"
SCENES_ASKED: Final = 20
MIN_SCENES: Final = 16
MAX_SCENES: Final = 20
GLIMMER_CANDIDATES: Final = 3
CANDIDATE_TEMPERATURE: Final = 1.0  # registered sampling, as for every arm
CRITIC_TEMPERATURE: Final = 0.0
LABELS: Final = "ABCD"
TEACHER_ALARM_SHARE: Final = 0.5  # a night whose targets are more often the teacher's is flagged
Adapter = Callable[[str], CheckpointRef | None]
SparkRole = Literal["person", "teacher", "critic"]
ROLES: Final[tuple[SparkRole, ...]] = ("person", "teacher", "critic")
CUT: Final = "length"  # the finish reason of a reply that reached its max_tokens
REASONING_ROOM: Final = 6_144
VISIBLE_TOKENS: Final[Mapping[str, int]] = MappingProxyType(
    {"person": 2_048, "teacher": GLIMMER_REPLY_CAP, "critic": 256})
SPARK_CAPS: Final[Mapping[str, int]] = MappingProxyType(
    {role: visible + REASONING_ROOM for role, visible in VISIBLE_TOKENS.items()})


class CappedChat(Protocol):
    """``baseten_api.chat``, capped per call (``max_tokens``)."""

    def __call__(
        self, model: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, timeout_s: float, *,
        temperature: float | None = None, max_tokens: int | None = None,
    ) -> ChatResult: ...


class _Spark:
    """One run's Spark Open calls, each capped by its role and counted, from any thread."""

    def __init__(self, chat: CappedChat, caps: Mapping[str, int]) -> None:
        self._chat, self._caps = chat, caps
        self._counts = {role: {"calls": 0, "in": 0, "out": 0, "cut": 0} for role in ROLES}
        self._lock = threading.Lock()

    def ask(self, role: SparkRole, messages: list[dict[str, Any]], temperature: float) -> ChatResult:
        result = self._chat(SPARK_OPEN, messages, None, SCENE_TIMEOUT_S, temperature=temperature,
                            max_tokens=self._caps[role])
        with self._lock:
            counts = self._counts[role]
            counts["calls"] += 1
            counts["in"] += result.input_tokens
            counts["out"] += result.output_tokens
            counts["cut"] += result.finish_reason == CUT
        return result

    def counts(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {role: dict(found) for role, found in self._counts.items()}


@dataclass(frozen=True)
class _Candidate:
    source: Candidate
    text: str
    place: int  # the order it was asked for: Glimmer 0 to 2, then the teacher


@dataclass(frozen=True)
class _Verdict:
    best: _Candidate
    worst: _Candidate | None  # None: no preference, so no pair
    unreadable: bool = False


@dataclass(frozen=True)
class _Scene:
    index: int
    message: str
    context: str | None  # the snapshot of C, taken before any candidate exists
    c_ancestry: frozenset[str] | None


@dataclass(frozen=True)
class _Night:
    user_id: str
    night: date
    known: str
    f_text: str
    f_ids: tuple[str, ...]
    f_ancestry: frozenset[str] | None
    adapter: CheckpointRef | None
    at: datetime
    spark: _Spark


def _marks(text: str, labels: dict[str, _Candidate]) -> tuple[dict[_Candidate, float], set[_Candidate]] | None:
    """The critic's scores and constraint misses by candidate, or None when they can't be read."""
    try:
        data: Any = json.loads(text.strip().removeprefix("```json").removeprefix("```").removesuffix("```"))
        raw_scores, raw_misses = data["scores"], data.get("misses_constraint", [])
    except (ValueError, KeyError, TypeError):
        return None
    if not isinstance(raw_scores, dict) or not isinstance(raw_misses, list):
        return None
    scores: dict[_Candidate, float] = {}
    for label, candidate in labels.items():
        score = raw_scores.get(label)
        if isinstance(score, bool) or not isinstance(score, int | float):
            return None
        scores[candidate] = float(score)
    return scores, {labels[label] for label in raw_misses if isinstance(label, str) and label in labels}


def _pick(candidates: Sequence[_Candidate], scores: dict[_Candidate, float], misses: set[_Candidate]) -> _Verdict:
    glimmer = [c for c in candidates if c.source == "glimmer"]
    teacher = next((c for c in candidates if c.source == "teacher"), None)
    best = max(glimmer, key=lambda c: (scores[c], -c.place))
    teacher_may_win = (
        teacher is not None and len(glimmer) == GLIMMER_CANDIDATES
        and all(c in misses for c in glimmer) and teacher not in misses
    )
    if teacher is not None and teacher_may_win and scores[teacher] > scores[best]:
        best = teacher
    worst = min(candidates, key=lambda c: (scores[c], c.source != "teacher", -c.place))
    return _Verdict(best, worst if scores[best] > scores[worst] else None)


class FullPractice:
    """Task 7's practice, the night's ``PracticeRunner``. ``adapter`` names her personal Glimmer
    for a person (the serving checkpoint; None before her first night, so the base model)."""

    def __init__(
        self, *, chat: CappedChat, client: ModelClient, adapter: Adapter, snapshot: Snapshot, seed: int,
        scenes: int = MIN_SCENES, teacher: bool = True, hosted_in_training: bool = False,
        f_cap: int = PRACTICE_F_CAP, concurrency: int = 1, spark_caps: Mapping[str, int] = SPARK_CAPS,
    ) -> None:
        if not MIN_SCENES <= scenes <= MAX_SCENES:
            raise ValueError(f"full practice runs {MIN_SCENES} to {MAX_SCENES} scenes a night, not {scenes}")
        if concurrency < 1:
            raise ValueError("practice runs at least one call at a time")
        self._chat = chat
        self._client = client
        self._adapter = adapter
        self._snapshot = snapshot
        self._seed = seed
        self._count = scenes
        self._teacher = teacher
        self._hosted = hosted_in_training
        self._f_cap = f_cap
        self._workers = concurrency
        self._caps = spark_caps

    @property
    def max_spark_calls(self) -> int:
        """The scenes, then per scene the teacher (when on) and the critic."""
        return 1 + self._count * (2 if self._teacher else 1)

    def practice(self, user_id: str, night: date, evidence: list[Evidence], hw: object | None) -> list[PracticeTranscript]:
        return list(self.run(user_id, night, evidence, hw, at=datetime.combine(night, datetime.min.time())).transcripts)

    def run(
        self, user_id: str, night: date, evidence: Sequence[Evidence], hw: object | None, *, at: datetime,
    ) -> PracticeRun:
        if hw is not None and not self._hosted:
            raise ValueError("homework never reaches practice while hosted outputs can't train (§3)")
        f = practice_f(evidence, hosted_in_training=self._hosted, cap_tokens=self._f_cap)
        if not f:
            return PracticeRun((), (), 0)
        prompt = render(PROMPT_PATH.read_text(encoding="utf-8"), {"KNOWN": _known(f), "COUNT": str(SCENES_ASKED)})
        spark = _Spark(self._chat, self._caps)
        written = spark.ask("person", [{"role": "user", "content": prompt}], SCENE_TEMPERATURE)
        if written.finish_reason == CUT:
            raise PracticeError(f"Spark Open's scenes ended at their cap of {self._caps['person']} tokens")
        messages = _scenes(written.text, SCENES_ASKED)
        cue_free = [message for message in messages if not has_cue(message)]
        ctx = _Night(user_id, night, _known(f), "## What you know about this person\n" + _known(f),
                     tuple(item.id for item in f), union(item.ancestry for item in f), self._adapter(user_id), at,
                     spark)
        scenes = [_Scene(index, message, *self._snapshot(user_id, message))  # C before any candidate
                  for index, message in enumerate(cue_free[: self._count])]
        calls = Calls(self._workers)
        done = calls.map(partial(self._scene, ctx, calls), scenes)
        kept = [(t, v) for t, v in done if t is not None and v is not None]
        return PracticeRun(
            tuple(t for t, _ in kept), ctx.f_ids, len(messages) - len(cue_free),
            dropped_cut=len(done) - len(kept), pairs=sum(v.worst is not None for _, v in kept),
            teacher_targets=sum(v.best.source == "teacher" for _, v in kept),
            critic_unreadable=sum(v.unreadable for _, v in kept), spark=spark.counts(),
        )

    def _glimmer(self, ctx: _Night, index: int, place: int, prefix: list[dict[str, str]]) -> _Candidate | None:
        answer = self._client.complete(
            ctx.adapter, prefix, temperature=CANDIDATE_TEMPERATURE,
            seed=_seed(self._seed, ctx.user_id, ctx.night.isoformat(), index, place),
        )
        if answer.finish_reason not in NORMAL_STOPS:  # a cut reply is never a candidate
            return None
        return _Candidate("glimmer", answer.text.strip(), place)

    def _taught(self, ctx: _Night, prefix: list[dict[str, str]]) -> _Candidate | None:
        taught = ctx.spark.ask("teacher", list(prefix), CANDIDATE_TEMPERATURE)
        if taught.finish_reason not in NORMAL_STOPS:  # cut at its cap: kept whole or not at all
            return None
        return _Candidate("teacher", taught.text.strip(), GLIMMER_CANDIDATES)

    def _candidates(self, ctx: _Night, calls: Calls, index: int, prefix: list[dict[str, str]]) -> list[_Candidate]:
        """The kept candidates by place: Glimmer's, side by side, then the teacher's only when a Glimmer
        one survived. Other scenes keep the server busy while this one waits for its teacher."""
        asks = [partial(self._glimmer, ctx, index, place, prefix) for place in range(GLIMMER_CANDIDATES)]
        glimmer = [c for c in calls.map(calls.call, asks) if c is not None]
        if not glimmer or not self._teacher:
            return glimmer
        taught = calls.call(partial(self._taught, ctx, prefix))
        return glimmer if taught is None else [*glimmer, taught]

    def _judge(self, ctx: _Night, calls: Calls, index: int, message: str, candidates: list[_Candidate]) -> _Verdict:
        if len(candidates) == 1:
            return _Verdict(candidates[0], None)
        order = sorted(candidates, key=lambda c: _seed(self._seed, ctx.user_id, ctx.night.isoformat(), index, "critic", c.place))
        labels = dict(zip(LABELS, order, strict=False))
        block = "\n".join(f"[{label}]\n{candidate.text}" for label, candidate in labels.items())
        prompt = render(CRITIC_PATH.read_text(encoding="utf-8"), {
            "COUNT": str(len(labels)), "KNOWN": ctx.known, "MESSAGE": message, "CANDIDATES": block})
        reply = calls.call(partial(ctx.spark.ask, "critic", [{"role": "user", "content": prompt}], CRITIC_TEMPERATURE))
        marks = None if reply.finish_reason == CUT else _marks(reply.text, labels)  # a cut verdict is never read
        if marks is None:
            return _Verdict(candidates[0], None, unreadable=True)
        return _pick(candidates, *marks)

    def _scene(self, ctx: _Night, calls: Calls, scene: _Scene) -> tuple[PracticeTranscript | None, _Verdict | None]:
        index, message = scene.index, scene.message
        prefix = [{"role": "system", "content": ctx.f_text}, {"role": "user", "content": message}]
        candidates = self._candidates(ctx, calls, index, prefix)
        if not any(c.source == "glimmer" for c in candidates):
            return None, None
        verdict = self._judge(ctx, calls, index, message, candidates)
        digest = hashlib.sha256(f"{ctx.user_id}\x1f{ctx.night.isoformat()}\x1f{index}".encode()).hexdigest()[:16]
        return PracticeTranscript(
            id=f"pr-{digest}", person_id=ctx.user_id, night=ctx.night, scene=index, message=message,
            reply=verdict.best.text, context=scene.context, f_ids=ctx.f_ids,
            ancestry=union((frozenset({SPARK_OPEN, GLIMMER}), ctx.f_ancestry, scene.c_ancestry)),
            created_at=ctx.at + timedelta(seconds=index), target=verdict.best.source,
            rejected=verdict.worst.text if verdict.worst else None,
            rejected_from=verdict.worst.source if verdict.worst else None,
        ), verdict
