"""Full practice (Task 7, from Oct 29): more scenes, a teacher candidate, a critic and pairs.

It keeps Task 5's cue-free scene prompt (``scenes_v1.md``, asked for ``SCENES_ASKED`` messages)
and the same F (``practice_f``), and runs ``scenes`` of them, 16 to 20 (a judge's night lowers the
floor, ``floor``). Each scene's message gets four candidates on one shared prefix (F as the system
message, then the message), so the server caches it: three from her personal Glimmer (the adapter
that serves her) and one from Spark Open, the teacher. A candidate cut at max tokens is never a
candidate. With no Glimmer candidate left the scene is dropped, so the teacher alone never trains.

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
2,000 tokens at most, and the longest call wrote about 8.5k, so about 6.5k of reasoning. The room is
twice that: a cap bounds a runaway call and doesn't bite a call like those measured (it lowers no
measured night's cost). What each role writes (``VISIBLE_TOKENS``):

- the person: ``SCENES_ASKED`` messages of one to three sentences as JSON; twenty three-sentence
  messages of 25-word sentences estimate under 1,900 tokens, so 2,048;
- the critic: a score and a miss for each of at most four letters as JSON, under 30 tokens, so 256;
- the teacher: a reply, held to the bound its Glimmer rivals have (``serving.client.MAX_TOKENS``):
  one cut at its cap, or longer than that bound by ``estimate_tokens`` (conservative, so it never
  lets a longer reply through), is no candidate, as a cut Glimmer reply never is.

A cap never changes training data silently. A cut critic is no verdict, and its scene is dropped:
it never trains on its first candidate without a pair (as an unreadable reply that stopped on its
own does). A cut list of scenes keeps its whole messages and drops the one the cut fell in.
``PracticeRun.spark`` counts each role's calls, tokens and cut replies (the teacher's past the
bound among them) and ``capped_scenes`` the scenes a cap dropped; the night log carries both.

**Who plays the roles** (``roles``, ``ANJO_SLEEP_PRACTICE_ROLES``; all Spark Open unless set):

- ``glimmer_person``: Glimmer's base model (no adapter) writes the scenes on the serving GPU, seeded per
  night, at the person's visible cap (it answers directly, with no reasoning to make room for).
- ``critic``: a third-party critic (``critics.ThirdPartyCritic``, DeepSeek V4.1 Flash) scores the
  candidates of a person ``third_party`` allows (synthetic people only); anyone else's critic is Spark
  Open. Its reply is read for scores and letters only, as Spark Open's is, and priced into
  ``PracticeRun.critic_usd``. A reply cut at its cap is never read, as Spark Open's: its scene is dropped.
- ``lazy_teacher``: the teacher is asked only where the critic found every Glimmer candidate missing a
  constraint, the only scene where it can become the target; the critic then judges the four again.
  Neither the teacher nor anything that trains ever reads a verdict: the teacher's prompt is F and the
  message, as before.

Glimmer's candidates are asked exactly as before, whoever plays the other roles, so whatever they read
(F, and MIND's soft tokens where the client carries them) is unchanged.
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

from anjo.sleep import spark_provider
from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.context.brief import CUT, REASONING_ROOM
from anjo.sleep.context.brief_store import union
from anjo.sleep.context.recall_common import estimate_tokens
from anjo.sleep.context.template import render
from anjo.sleep.mind.loop_focus import FocusTarget
from anjo.sleep.parallel import Calls
from anjo.sleep.model_api import ChatResult
from anjo.sleep.practice.critics import ThirdPartyCritic
from anjo.sleep.practice.run import (
    NORMAL_STOPS,
    PRACTICE_F_CAP,
    PROMPT_PATH,
    SCENE_TEMPERATURE,
    SCENE_TIMEOUT_S,
    PracticeRun,
    Snapshot,
    PracticeError,
    _known,
    _scenes,
    _seed,
    focus_count,
    focus_prompt,
    focus_scenes,
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
LOWEST_FLOOR: Final = 4  # a judge's night may lower the floor this far (``floor``; night/demo.py's DEMO_MIN_SCENES)
GLIMMER_CANDIDATES: Final = 3
CANDIDATE_TEMPERATURE: Final = 1.0  # registered sampling, as for every arm
CRITIC_TEMPERATURE: Final = 0.0
LABELS: Final = "ABCD"
TEACHER_ALARM_SHARE: Final = 0.5  # a night whose targets are more often the teacher's is flagged
Adapter = Callable[[str], CheckpointRef | None]
SparkRole = Literal["person", "teacher", "critic"]
ROLES: Final[tuple[SparkRole, ...]] = ("person", "teacher", "critic")
VISIBLE_TOKENS: Final[Mapping[str, int]] = MappingProxyType(
    {"person": 2_048, "teacher": GLIMMER_REPLY_CAP, "critic": 256})
SPARK_CAPS: Final[Mapping[str, int]] = MappingProxyType(
    {role: visible + REASONING_ROOM for role, visible in VISIBLE_TOKENS.items()})


class _CappedClient(Protocol):
    """Glimmer's client, capped per call (``VllmClient``): the person's scenes when Glimmer plays it."""

    def complete(
        self, adapter: CheckpointRef | None, messages: list[dict[str, str]], *, temperature: float,
        seed: int | None = None, max_tokens: int | None = None,
    ) -> ChatResult: ...


@dataclass(frozen=True)
class _Said:
    """A critic's reply as practice reads it: its text, and whether it reached its cap."""

    text: str
    cut: bool


class _ThirdParty:
    """One run's third-party critic calls, counted and priced from any thread."""

    def __init__(self, critic: ThirdPartyCritic) -> None:
        self.critic = critic
        self.counts = {"calls": 0, "in": 0, "out": 0, "cut": 0}
        self.usd = 0.0
        self._lock = threading.Lock()

    def ask(self, prompt: str) -> _Said:
        reply = self.critic.ask(prompt)
        cut = reply.finish_reason == CUT
        with self._lock:
            self.counts["calls"] += 1
            self.counts["in"] += reply.input_tokens
            self.counts["out"] += reply.output_tokens
            self.counts["cut"] += cut
            self.usd += self.critic.usd(reply.input_tokens, reply.output_tokens)
        return _Said(reply.text, cut)  # a cut verdict is never read: its scene is dropped


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

    def over_bound(self, role: SparkRole) -> None:
        """A reply that stopped on its own past its role's bound: counted with the cut ones."""
        with self._lock:
            self._counts[role]["cut"] += 1

    def counts(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {role: dict(found) for role, found in self._counts.items()}


def _salvaged(raw: str) -> list[str]:
    """The whole messages of a list of scenes cut at its cap: each complete string in ``scenes``,
    up to the one the cut fell in."""
    named = raw.find('"scenes"')
    at = raw.find("[", named) + 1 if named >= 0 else 0
    found: list[str] = []
    decoder = json.JSONDecoder()
    while at:
        while at < len(raw) and raw[at] in " \t\r\n,":
            at += 1
        try:
            value, at = decoder.raw_decode(raw, at)
        except ValueError:
            break
        if isinstance(value, str) and value.strip():
            found.append(value.strip())
    return found


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
    glimmer_all_miss: bool = False  # all three Glimmer candidates miss a constraint: the teacher may win
    flagged: int = 0  # candidates the critic listed as missing a constraint
    best_flagged: bool = False  # the critic listed the kept candidate among them


@dataclass(frozen=True)
class _Scene:
    index: int
    message: str
    context: str | None  # the snapshot of C, taken before any candidate exists
    c_ancestry: frozenset[str] | None
    focus: bool = False  # drawn from a focus target of her own questions (the MIND loop)
    target_id: str | None = None  # which one; None for every other scene


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
    critic: Callable[[str], _Said]
    flags: bool = False  # the MIND loop's night: record whether the critic flagged the kept reply


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
    all_miss = len(glimmer) == GLIMMER_CANDIDATES and all(c in misses for c in glimmer)
    return _Verdict(best, worst if scores[best] > scores[worst] else None, glimmer_all_miss=all_miss,
                    flagged=len(misses), best_flagged=best in misses)


def spark_calls(scenes: int, *, teacher: bool = True, glimmer_person: bool = False, lazy_teacher: bool = False) -> int:
    """The most Spark Open calls a night's practice makes: the scenes (unless Glimmer writes them), then
    per scene the teacher (when on) and the critic, twice with a lazy teacher. A third-party critic
    still counts: a real person's is Spark Open."""
    lazy = lazy_teacher and teacher
    return (0 if glimmer_person else 1) + scenes * ((1 if teacher else 0) + (2 if lazy else 1))


class FullPractice:
    """Task 7's practice, the night's ``PracticeRunner``. ``adapter`` names her personal Glimmer
    for a person (the serving checkpoint; None before her first night, so the base model)."""

    def __init__(
        self, *, chat: CappedChat, client: ModelClient, adapter: Adapter, snapshot: Snapshot, seed: int,
        scenes: int = MIN_SCENES, teacher: bool = True, hosted_in_training: bool = False,
        f_cap: int = PRACTICE_F_CAP, concurrency: int = 1, spark_caps: Mapping[str, int] = SPARK_CAPS,
        glimmer_person: bool = False, critic: ThirdPartyCritic | None = None,
        third_party: Callable[[str], bool] = lambda user_id: False, lazy_teacher: bool = False,
        floor: int = MIN_SCENES,
    ) -> None:
        if not LOWEST_FLOOR <= floor <= MIN_SCENES:
            raise ValueError(f"practice's floor is {LOWEST_FLOOR} to {MIN_SCENES} scenes, not {floor}")
        if not floor <= scenes <= MAX_SCENES:
            raise ValueError(f"full practice runs {floor} to {MAX_SCENES} scenes a night, not {scenes}")
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
        self._glimmer_person = glimmer_person
        self._critic = critic
        self._third_party = third_party
        self._lazy = lazy_teacher and teacher

    @property
    def max_spark_calls(self) -> int:
        return spark_calls(self._count, teacher=self._teacher, glimmer_person=self._glimmer_person,
                           lazy_teacher=self._lazy)

    def _roles(self, third: _ThirdParty | None) -> dict[str, str]:
        teacher = ("spark-lazy" if self._lazy else "spark") if self._teacher else "off"
        return {"person": "glimmer" if self._glimmer_person else "spark", "teacher": teacher,
                "critic": third.critic.model if third is not None else "spark"}

    def _write_scenes(self, spark: _Spark, user_id: str, night: date, prompt: str, *,
                      seed_key: str = "person") -> ChatResult:
        messages = [{"role": "user", "content": prompt}]
        if not self._glimmer_person:
            return spark.ask("person", messages, SCENE_TEMPERATURE)
        glimmer: Any = self._client  # VllmClient: the base model, capped for this call alone
        written: ChatResult = glimmer.complete(
            None, messages, temperature=SCENE_TEMPERATURE, seed=_seed(self._seed, user_id, night.isoformat(), seed_key),
            max_tokens=VISIBLE_TOKENS["person"])
        return written

    def practice(self, user_id: str, night: date, evidence: list[Evidence], hw: object | None) -> list[PracticeTranscript]:
        return list(self.run(user_id, night, evidence, hw, at=datetime.combine(night, datetime.min.time())).transcripts)

    def run(
        self, user_id: str, night: date, evidence: Sequence[Evidence], hw: object | None, *, at: datetime,
        focus: Sequence[FocusTarget] = (),
    ) -> PracticeRun:
        if hw is not None and not self._hosted:
            raise ValueError("homework never reaches practice while hosted outputs can't train (§3)")
        f = practice_f(evidence, hosted_in_training=self._hosted, cap_tokens=self._f_cap)
        if not f:
            return PracticeRun((), (), 0)
        known = _known(f)
        spark = _Spark(self._chat, self._caps)
        cut = False
        focus_pairs: list[tuple[str, str]] = []
        if focus:  # her focus scenes first, in their own call, so each names its target
            asked = max(1, focus_count(self._count))
            try:
                focus_text = focus_prompt(focus, asked, known)
            except (KeyError, OSError):  # the loop's prompt can't be read: tonight's scenes are today's
                focus_pairs = []
            else:
                focus_written = self._write_scenes(spark, user_id, night, focus_text, seed_key="focus")
                try:
                    focus_pairs = focus_scenes(focus_written.text, focus, asked)
                    cut = focus_written.finish_reason == CUT
                except PracticeError:
                    # Malformed focus scenes are the loop's failure, not the night's: the rest of
                    # tonight's scenes are today's, and the prediction resolves as it can.
                    focus_pairs = []
        rest = max(0, SCENES_ASKED - len(focus_pairs))
        if rest:
            prompt = render(PROMPT_PATH.read_text(encoding="utf-8"), {"KNOWN": known, "COUNT": str(rest)})
            written = self._write_scenes(spark, user_id, night, prompt)
            cut = cut or written.finish_reason == CUT  # the scene it was writing is dropped, the whole ones kept
            messages = _salvaged(written.text)[:rest] if written.finish_reason == CUT else _scenes(written.text, rest)
        else:
            messages = []
        drafts: list[tuple[str, bool, str | None]] = [(message, True, target_id) for target_id, message in focus_pairs]
        drafts += [(message, False, None) for message in messages]
        cue_free = [draft for draft in drafts if not has_cue(draft[0])]
        dropped = len(drafts) - len(cue_free)
        third = _ThirdParty(self._critic) if self._critic is not None and self._third_party(user_id) else None

        def critic(text: str) -> _Said:
            if third is not None:
                return third.ask(text)
            reply = spark.ask("critic", [{"role": "user", "content": text}], CRITIC_TEMPERATURE)
            return _Said(reply.text, reply.finish_reason == CUT)

        ctx = _Night(user_id, night, known, "## What you know about this person\n" + known,
                     tuple(item.id for item in f), union(item.ancestry for item in f), self._adapter(user_id), at,
                     spark, critic, flags=bool(focus))
        # Numbered after the cue filter, as practice always has: a scene's index seeds its candidates and
        # names its record, so with no focus a night's scenes are exactly today's. C before any
        # candidate, taken on the calling thread, for the kept scenes only.
        scenes = [_Scene(index, message, *self._snapshot(user_id, message), focus=is_focus, target_id=target_id)
                  for index, (message, is_focus, target_id) in enumerate(cue_free[: self._count])]
        calls = Calls(self._workers)
        done = calls.map(partial(self._scene, ctx, calls), scenes)
        kept = [(t, v) for t, v, _ in done if t is not None and v is not None]
        capped = sum(c for _, _, c in done)
        return PracticeRun(
            tuple(t for t, _ in kept), ctx.f_ids, dropped,
            dropped_cut=len(done) - len(kept) - capped, pairs=sum(v.worst is not None for _, v in kept),
            teacher_targets=sum(v.best.source == "teacher" for _, v in kept),
            critic_unreadable=sum(v.unreadable for _, v in kept), capped_scenes=capped + cut, spark=spark.counts(),
            critic_usd=third.usd if third is not None else 0.0, roles=self._roles(third),
            flagged=sum(v.flagged for _, v in kept),
            third_party=dict(third.counts) if third is not None else {},
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
        text = taught.text.strip()
        if estimate_tokens(text) > GLIMMER_REPLY_CAP:  # longer than any Glimmer rival can be
            ctx.spark.over_bound("teacher")
            return None
        return _Candidate("teacher", text, GLIMMER_CANDIDATES)

    def _candidates(self, ctx: _Night, calls: Calls, index: int, prefix: list[dict[str, str]]) -> list[_Candidate]:
        """The kept candidates by place: Glimmer's, side by side, then the teacher's only when a Glimmer
        one survived (a lazy teacher waits for the verdict). Other scenes keep the server busy while this
        one waits for its teacher."""
        asks = [partial(self._glimmer, ctx, index, place, prefix) for place in range(GLIMMER_CANDIDATES)]
        glimmer = [c for c in calls.map(calls.call, asks) if c is not None]
        if not glimmer or not self._teacher or self._lazy:
            return glimmer
        taught = calls.call(partial(self._taught, ctx, prefix))
        return glimmer if taught is None else [*glimmer, taught]

    def _judge(
        self, ctx: _Night, calls: Calls, index: int, message: str, candidates: list[_Candidate],
    ) -> _Verdict | None:
        """The verdict; None when the critic's reply was cut at its cap (the scene is dropped)."""
        if len(candidates) == 1:
            return _Verdict(candidates[0], None)
        order = sorted(candidates, key=lambda c: _seed(self._seed, ctx.user_id, ctx.night.isoformat(), index, "critic", c.place))
        labels = dict(zip(LABELS, order, strict=False))
        block = "\n".join(f"[{label}]\n{candidate.text}" for label, candidate in labels.items())
        prompt = render(CRITIC_PATH.read_text(encoding="utf-8"), {
            "COUNT": str(len(labels)), "KNOWN": ctx.known, "MESSAGE": message, "CANDIDATES": block})
        reply = calls.call(partial(ctx.critic, prompt))
        if reply.cut:  # a cut verdict is never read, and its scene never trains
            return None
        marks = _marks(reply.text, labels)
        if marks is None:
            return _Verdict(candidates[0], None, unreadable=True)
        return _pick(candidates, *marks)

    def _scene(
        self, ctx: _Night, calls: Calls, scene: _Scene,
    ) -> tuple[PracticeTranscript | None, _Verdict | None, bool]:
        """The scene's transcript and verdict, or Nones, and whether a cap dropped it."""
        index, message = scene.index, scene.message
        prefix = [{"role": "system", "content": ctx.f_text}, {"role": "user", "content": message}]
        candidates = self._candidates(ctx, calls, index, prefix)
        if not any(c.source == "glimmer" for c in candidates):
            return None, None, False
        verdict = self._judge(ctx, calls, index, message, candidates)
        if verdict is not None and self._lazy and verdict.glimmer_all_miss:  # only here can the teacher win
            taught = calls.call(partial(self._taught, ctx, prefix))  # F and the message: never the verdict
            if taught is not None:
                verdict = self._judge(ctx, calls, index, message, [*candidates, taught])
        if verdict is None:
            return None, None, True
        digest = hashlib.sha256(f"{ctx.user_id}\x1f{ctx.night.isoformat()}\x1f{index}".encode()).hexdigest()[:16]
        return PracticeTranscript(
            id=f"pr-{digest}", person_id=ctx.user_id, night=ctx.night, scene=index, message=message,
            reply=verdict.best.text, context=scene.context, f_ids=ctx.f_ids,
            ancestry=union((frozenset({spark_provider.current().model, GLIMMER}), ctx.f_ancestry, scene.c_ancestry)),
            created_at=ctx.at + timedelta(seconds=index), target=verdict.best.source,
            rejected=verdict.worst.text if verdict.worst else None,
            rejected_from=verdict.worst.source if verdict.worst else None,
            focus=scene.focus, target_id=scene.target_id,
            flagged=verdict.best_flagged and ctx.flags,  # today's bytes without the loop
        ), verdict, False
