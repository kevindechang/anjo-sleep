"""``sft_dpo_seed``'s extra rows, built on the host from what the night already holds (2026-10-29).

**Knowledge seeding** (``seed_rows``). The paper's consolidation: the teacher is the model with the
short-term memory in context, the student the model without it. Here the teacher is Glimmer with her
long-term adapter (the one the night starts from) and the consolidated day before the message: her notes on the person (the pre-practice
snapshot of C) and what they told her today (the day's own evidence: lessons, corrections,
preferences, tone requests). The student is Glimmer with tonight's adapter and the message alone.
Both score the same reply: tonight's practice reply from her own Glimmer, already a sample of the
serving adapter. A reply the critic took from the Spark Open teacher is no sample of hers and seeds
nothing. The day's items are what held across the day; MIND's live threads are never in the
teacher's context (they are the thought channel's, and the student must not learn to recite them).

A row is refused, and counted, when:

- its ancestry is not cleared (``provenance.may_train``): the union of the transcript's (message,
  reply, C and F) and every item of today's context. One item from Claude, Gemini or OpenAI, or of
  unknown origin, excludes every row whose context holds it;
- its context holds a sealed probe verbatim or by a 13-gram, the rule C is held to;
- it has no context (the teacher would be the student), or is longer than the training sequence.

The message and the reply passed the dataset's three rules already: seeding takes kept examples only.

**Dreams** (``dream_rows``): Glimmer's candidates from MIND's dream seeds (``night/dreams.py``),
each resting on at least one real event, cleared by its ancestry, its message and reply held to the
decontaminator's three rules and its thread (the student's input) to C's. **Anchors** (``anchor_rows``): the 32
hand-written items of ``sets/anchors.jsonl``, which the training job scores before and after the dream block. No gate
scores them: training must never see a gate item.

**Day dreams** (``day_dream_rows``, ``ANJO_SLEEP_DAY_DREAMS=1``): her Glimmer writes the messages the
person might send next from the same consolidated day the seed teacher reads, then answers each with
that day in context (``night/day_dreams.py``). The rows are knowledge-seeding rows, so the teacher's
context is the day and the student's the message alone; ``group="dream"`` marks them for the job's
keep-if-improves trials. The day is the newest transcript whose context is non-empty and whose
ancestry may train; a day that may not train, or that holds a sealed probe, asks nothing of her
Glimmer and spends nothing (``attach_night_terms``). Replay rows are earlier practice examples: the
same refusals as tonight's, newest first, at most ``SEED_REPLAY_MAX``, because the night's own kept
replies are too few to fill the capacity point's steps; a refusal is counted only while that cap
still leaves room, so rows past it never inflate the receipt. They stay ordinary seed rows; only the
dreams carry a group.

**Self-distillation** (``sdft_rows``, ``ANJO_SLEEP_SDFT=1``), after Shenfeld, Damani, Hübotter and Agrawal,
"Self-Distillation Enables Continual Learning" (arXiv 2601.19897): the same model is the teacher with a
demonstration in context and the student without it, trained on the student's own samples. Arm 2C holds
what she needs (``probes/arms.py``: its system turn is ``ContextBuilder.build``, the brief, preferences
and corrections since it, and recall for the message, before the probe's prompt), and arm 3 must hold it
in her weights. So the teacher reads the seeding row's consolidated day, the same material (the
snapshot of C, built as 2C's is, then today's lessons, corrections, preferences and tone requests), and
the student the query alone. No model writes a query: each is one of the seeding rows' own messages,
tonight's practice first, then her day dreams, then replay, at most ``SDFT_SOURCES_MAX`` distinct ones,
in two forms (``SDFT_FRAMES``): bare (``tone``: the work message as it came, as a ``tone_hold`` probe is
an ordinary work message, so length, opening and hedging show unasked; the day's new work is asked the
same way) and asking for a choice (``taste``: where her preferences decide). No frame names a preference,
a tone or a style, so the student learns to hold them unasked; the job samples every reply (``sleep_loss.py``). A row is refused, and
counted, when its query carries (``OverlapGuard``) a sealed probe, a held-out item, an item of a gate,
anchor or knowledge set, or her own preference or tone request (a cue would teach the request, not its
hold); when the decontaminator flags it; or when it is longer than the training sequence.
"""
from __future__ import annotations

import functools
import hashlib
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import date
from types import MappingProxyType
from typing import Any, Final

from anjo.sleep.context.recall_common import estimate_tokens
from anjo.sleep.night.day_dreams import DAY_DREAM_GROUP, DayDream, DreamDay
from anjo.sleep.night.deps import NightDeps
from anjo.sleep.night.dreams import SEED_KINDS, Dream
from anjo.sleep.night.meter import SpendMeter
from anjo.sleep.ports import Decontaminator, EvidenceStore
from anjo.sleep.practice.transcripts import PracticeTranscript, TranscriptStore
from anjo.sleep.probes.decontam import NGRAM, probe_texts, words
from anjo.sleep.provenance import may_train, union
from anjo.sleep.sets.loader import SETS_DIR, load_anchors, load_collapse, load_general
from anjo.sleep.train.dataset import (
    DEFAULT_MAX_SEQ_TOKENS,
    ROLE_OVERHEAD_TOKENS,
    AnchorRow,
    Dataset,
    DreamRow,
    Example,
    SdftRow,
    SeedRow,
    carries_verbatim,
)
from anjo.sleep.types import Evidence, Probe

DAY_KINDS: Final = ("lesson", "correction", "preference", "tone_request")  # what the person said today
ANCHORS: Final = 32
SEED_DROPS: Final = ("missing", "teacher_target", "no_context", "uncleared", "probe", "truncated")
SEED_REPLAY_MAX: Final = 24  # an earlier-practice cap: enough rows for capacity, never the whole log
DREAMS_NOT_WIRED: Final = "not wired"  # the dataset receipt's dreams while MIND's dream states are not
DREAM_DROPS: Final = ("unanchored", "uncleared", "probe", "truncated")
DAY_DREAMS_NOT_WIRED: Final = "not wired"  # the dataset receipt's day dreams without a serving client
DAY_DREAM_DROPS: Final = ("no_context", "uncleared", "probe", "truncated")
NOTES_HEAD: Final = "What you know about them, from your notes:"
TODAY_HEAD: Final = "What they told you today:"
SDFT_KINDS: Final = ("tone", "taste")
# What each kind adds to her message. None names a preference, a tone or a style (a probe's own rule); a
# tone query is the bare message, as a tone_hold probe is (a bare knowledge row would be the same query).
SDFT_FRAMES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType({
    "tone": ("",),
    "taste": ("Which way would you go, and why?", "What would you choose here?"),
})
# Distinct messages a night distils, a row a kind: tonight's practice (16 scenes), then day dreams, then
# replay all fit. A sampled visit is about 30 s on the A100 (192 tokens of Glimmer-30B in 4-bit at some
# 8 a second, then the teacher's and the student's passes), so 24 sources less the 2 held are some 11
# minutes of tone rows at one epoch, inside the term's 15 (``sleep_loss.SDFT_BUDGET_S``); past it a
# visit draws nothing.
SDFT_SOURCES_MAX: Final = 24
SDFT_REPLY_TOKENS: Final = 192  # the student's sample, which the length counts (``sleep_loss.SDFT_TOKENS``)
SDFT_DROPS: Final = ("probe", "held_out", "gate_set", "cue", "truncated")
CUE_KINDS: Final = ("preference", "tone_request")  # her own requests: a query never carries one
GUARD_EXACT_WORDS: Final = 3  # a needle this long or longer is refused whole wherever it appears
GUARD_NEAR_NGRAM: Final = 5
GUARD_NEAR_SHARE: Final = 0.5  # half of either side's word 5-grams in the other: a near duplicate


def today_items(day: Sequence[Evidence]) -> list[Evidence]:
    """The day's own evidence that may sit in the teacher's context: never a held-out item."""
    kept = [item for item in day if not item.held_out and item.kind in DAY_KINDS]
    return sorted(kept, key=lambda item: (item.created_at, item.id))


def consolidated_day(notes: str | None, items: Sequence[Evidence]) -> str:
    """The teacher's context: her notes on the person, then today's items, one a line."""
    parts = []
    if notes and notes.strip():
        parts.append(f"{NOTES_HEAD}\n{notes.strip()}")
    if items:
        parts.append(TODAY_HEAD + "\n" + "\n".join(f"- {item.text.strip()}" for item in items))
    return "\n\n".join(parts)


def _tokens(*texts: str) -> int:
    return sum(estimate_tokens(text) + ROLE_OVERHEAD_TOKENS for text in texts)


def _seed_row(
    example: Example, transcript: PracticeTranscript, items: Sequence[Evidence],
    day_ancestry: frozenset[str] | None, sealed: Sequence[Probe], *, hosted_in_training: bool,
    max_seq_tokens: int, group: str | None,
) -> tuple[SeedRow | None, str | None]:
    """One seeding row from ``example`` and its transcript, or the refusal's reason."""
    context = consolidated_day(transcript.context, items)
    tokens = _tokens(context, transcript.message, example.completion)
    reason = (
        "no_context" if not context
        else "uncleared" if not may_train(union((transcript.ancestry, day_ancestry)),
                                          hosted_in_training=hosted_in_training)
        else "probe" if carries_verbatim(context, list(sealed))
        else "truncated" if tokens > max_seq_tokens else None
    )
    if reason is not None:
        return None, reason
    return SeedRow(id=example.id, context=context, message=transcript.message, reply=example.completion,
                   tokens=tokens, core=example.core, group=group), None


def seed_rows(
    data: Dataset, transcripts: TranscriptStore, day: Sequence[Evidence], sealed: Sequence[Probe], *,
    hosted_in_training: bool, max_seq_tokens: int = DEFAULT_MAX_SEQ_TOKENS, replay: bool = False,
) -> tuple[tuple[SeedRow, ...], dict[str, int]]:
    """Tonight's seeding rows, from the dataset's kept new examples, and the refusals by reason. With
    ``replay``, earlier practice rows join them under the same refusals, newest first, at most
    ``SEED_REPLAY_MAX``: they are where a night's training volume comes from once its own rows run out."""
    dropped = dict.fromkeys(SEED_DROPS, 0)
    items = today_items(day)
    day_ancestry = union(item.ancestry for item in items)
    rows: list[SeedRow] = []
    for example in data.examples:
        if not example.new:
            continue
        if example.teacher:
            dropped["teacher_target"] += 1
            continue
        transcript = transcripts.get(example.id)
        if transcript is None:
            dropped["missing"] += 1
            continue
        row, reason = _seed_row(example, transcript, items, day_ancestry, sealed,
                                hosted_in_training=hosted_in_training, max_seq_tokens=max_seq_tokens, group=None)
        if reason is not None:
            dropped[reason] += 1
            continue
        rows.append(row)
    if not replay:
        return tuple(rows), dropped
    earlier: list[tuple[Example, PracticeTranscript]] = []
    undated: list[str] = []  # refusals with no transcript to date them by: considered after the dated rows
    for example in data.examples:
        if example.new:
            continue
        if example.teacher:
            undated.append("teacher_target")
            continue
        transcript = transcripts.get(example.id)
        if transcript is None:
            undated.append("missing")
            continue
        earlier.append((example, transcript))
    earlier.sort(key=lambda pair: (pair[1].created_at, pair[1].id), reverse=True)  # newest first
    kept = 0
    for example, transcript in earlier:
        if kept >= SEED_REPLAY_MAX:
            break
        row, reason = _seed_row(example, transcript, items, day_ancestry, sealed,
                                hosted_in_training=hosted_in_training, max_seq_tokens=max_seq_tokens, group=None)
        if reason is not None:
            dropped[reason] += 1
            continue
        rows.append(row)
        kept += 1
    for reason in undated:  # only while the cap leaves room: a refusal past the cap was never considered
        if kept >= SEED_REPLAY_MAX:
            break
        dropped[reason] += 1
    return tuple(rows), dropped


def day_notes(
    data: Dataset, transcripts: TranscriptStore, *, hosted_in_training: bool,
) -> tuple[str | None, frozenset[str] | None]:
    """The notes the night's practice read and their ancestry: the day dreams start from the same day
    the seed teacher reads. It is the newest new example's transcript whose context is non-empty and
    whose ancestry ``may_train`` clears, so an uncleared note never reaches Glimmer's context."""
    made = sorted((transcript for example in data.examples if example.new
                   and (transcript := transcripts.get(example.id)) is not None),
                  key=lambda transcript: (transcript.created_at, transcript.id), reverse=True)
    for transcript in made:
        if (transcript.context and transcript.context.strip()
                and may_train(transcript.ancestry, hosted_in_training=hosted_in_training)):
            return transcript.context, transcript.ancestry
    return None, frozenset()


def day_dream_context(
    data: Dataset, transcripts: TranscriptStore, day: Sequence[Evidence], *, hosted_in_training: bool,
) -> DreamDay:
    """What the day dreams read: the consolidated day and the ancestry of everything in it, her own
    replies and the day's items, so one uncleared item excludes the dreams built on it."""
    notes, notes_ancestry = day_notes(data, transcripts, hosted_in_training=hosted_in_training)
    items = today_items(day)
    return DreamDay(text=consolidated_day(notes, items),
                    ancestry=union((notes_ancestry, *(item.ancestry for item in items))))


def day_dream_rows(
    dreams: Sequence[DayDream], day: DreamDay, sealed: Sequence[Probe], decontam: Decontaminator, *,
    hosted_in_training: bool, max_seq_tokens: int = DEFAULT_MAX_SEQ_TOKENS,
) -> tuple[tuple[SeedRow, ...], dict[str, int]]:
    """The day dreams that may train, and the refusals by reason: the day's ancestry, then the
    decontaminator's rules on the message and reply, the day's 13-gram probe rule and the length."""
    dropped = dict.fromkeys(DAY_DREAM_DROPS, 0)
    if not day.text.strip():
        dropped["no_context"] = len(dreams)
        return (), dropped
    cleared = [dream for dream in dreams if may_train(dream.ancestry, hosted_in_training=hosted_in_training)]
    dropped["uncleared"] = len(dreams) - len(cleared)
    flags = decontam.check([f"{dream.message}\n{dream.reply}" for dream in cleared], list(sealed)) if cleared else ()
    if len(flags) != len(cleared):
        raise ValueError("the decontaminator must return one flag per text")
    leaks = carries_verbatim(day.text, list(sealed))
    rows: list[SeedRow] = []
    for dream, flag in zip(cleared, flags, strict=True):
        tokens = _tokens(day.text, dream.message, dream.reply)
        reason = ("probe" if flag or leaks else "truncated" if tokens > max_seq_tokens else None)
        if reason is not None:
            dropped[reason] += 1
            continue
        rows.append(SeedRow(id=dream.id, context=day.text, message=dream.message, reply=dream.reply,
                            tokens=tokens, group=DAY_DREAM_GROUP))
    return tuple(rows), dropped


def dream_rows(
    dreams: Sequence[Dream], sealed: Sequence[Probe], decontam: Decontaminator, *, hosted_in_training: bool,
    max_seq_tokens: int = DEFAULT_MAX_SEQ_TOKENS,
) -> tuple[tuple[DreamRow, ...], dict[str, int]]:
    """The dream candidates that may train, and the refusals by reason: one resting on no real event
    first (any source's, not only Glimmer's), then the ancestry, the probes and the length."""
    dropped = dict.fromkeys(DREAM_DROPS, 0)
    anchored = [dream for dream in dreams if dream.event_ids]
    cleared = [dream for dream in anchored if may_train(dream.ancestry, hosted_in_training=hosted_in_training)]
    dropped["unanchored"] = len(dreams) - len(anchored)
    dropped["uncleared"] = len(anchored) - len(cleared)
    flags = decontam.check([f"{dream.message}\n{dream.reply}" for dream in cleared], list(sealed)) if cleared else ()
    if len(flags) != len(cleared):
        raise ValueError("the decontaminator must return one flag per text")
    rows: list[DreamRow] = []
    for dream, flag in zip(cleared, flags, strict=True):
        tokens = _tokens(dream.thread, dream.message, dream.reply)
        reason = ("probe" if flag or carries_verbatim(dream.thread, list(sealed))
                  else "truncated" if tokens > max_seq_tokens else None)
        if reason is not None:
            dropped[reason] += 1
            continue
        rows.append(DreamRow(id=dream.id, thread=dream.thread, message=dream.message, reply=dream.reply, tokens=tokens,
                             kind=dream.kind, core=dream.core))
    return tuple(rows), dropped


def _grams(tokens: Sequence[str], n: int) -> frozenset[tuple[str, ...]]:
    return frozenset(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


class OverlapGuard:
    """What no self-distillation query may carry, each needle labelled by where it comes from (``probe``,
    ``held_out``, ``gate_set`` or ``cue``). After the decontaminator's folding (NFKC, case, letters and
    digits only), a query is refused when it holds a needle of ``GUARD_EXACT_WORDS`` words or more whole,
    shares a run of ``NGRAM`` (13) words with one (RF5's rule), or is a near duplicate of one: at least
    ``GUARD_NEAR_SHARE`` of the needle's word 5-grams are in the query, or of the query's in the needle
    (a query cut from a longer item). ``refusal`` names the first needle's label, probes first."""

    def __init__(self, needles: Iterable[tuple[str, str]]) -> None:
        self._needles = [(label, tokens, _grams(tokens, GUARD_NEAR_NGRAM), _grams(tokens, NGRAM))
                         for label, text in needles if (tokens := tuple(words(text)))]

    def refusal(self, text: str) -> str | None:
        hay = words(text)
        padded = " " + " ".join(hay) + " "
        near, long = _grams(hay, GUARD_NEAR_NGRAM), _grams(hay, NGRAM)
        for label, tokens, needle_near, needle_long in self._needles:
            if len(tokens) >= GUARD_EXACT_WORDS and " " + " ".join(tokens) + " " in padded:
                return label
            if needle_long & long:
                return label
            shared = len(needle_near & near)
            if shared and (shared >= GUARD_NEAR_SHARE * len(needle_near) or shared >= GUARD_NEAR_SHARE * len(near)):
                return label
        return None


@functools.cache
def set_texts() -> tuple[str, ...]:
    """Every item text of the gate sets (general and collapse), the anchors and the frozen knowledge sets:
    questions, prompts and choices. Read once a process; none of them is ever a training query."""
    texts = [text for item in (*load_general(), *load_anchors()) for text in (item.question, *item.choices)]
    texts += [item.prompt for item in load_collapse()]
    for path in sorted(SETS_DIR.glob("knowledge_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                texts += [str(row.get("prompt", "")), *(str(choice) for choice in row.get("choices", ()))]
    return tuple(text for text in texts if text.strip())


def held_out_items(store: EvidenceStore, night: date) -> list[Evidence]:
    """The person's held-out items on every day her stored evidence spans, and tonight's."""
    days = {item.day for item in store.stored(night)} | {night}
    return [item for day in sorted(days) for item in store.held_out(day)]


def sdft_guard(sealed: Sequence[Probe], held_out: Sequence[Evidence], said: Sequence[Evidence]) -> OverlapGuard:
    """The night's guard: her sealed probes (prompts and options), her held-out items, every set item,
    and her own preferences and tone requests (``said``'s, as cues)."""
    return OverlapGuard([
        *(("probe", text) for probe in sealed for text in probe_texts(probe)),
        *(("held_out", item.text) for item in held_out),
        *(("gate_set", text) for text in set_texts()),
        *(("cue", item.text) for item in said if item.kind in CUE_KINDS),
    ])


def sdft_query(message: str, kind: str, source: str, seed: int) -> str:
    """``message`` in ``kind``'s form: bare, or followed by one of its frames, picked by a seeded hash."""
    frames = SDFT_FRAMES[kind]
    frame = frames[int(hashlib.sha256(f"{seed}:sdft:{source}:{kind}".encode()).hexdigest()[:8], 16) % len(frames)]
    return f"{message.strip()}\n\n{frame}" if frame else message.strip()


def sdft_rows(
    seeds: Sequence[SeedRow], replay_ids: frozenset[str], guard: OverlapGuard, sealed: Sequence[Probe],
    decontam: Decontaminator, *, seed: int, max_seq_tokens: int = DEFAULT_MAX_SEQ_TOKENS,
) -> tuple[tuple[SdftRow, ...], dict[str, int]]:
    """The self-distillation rows from the seeding rows' messages, and the refusals by reason. Sources:
    tonight's rows, then her day dreams, then replay (``replay_ids``), each message once, the first
    ``SDFT_SOURCES_MAX``; each gives one row a kind, with its seeding row's context and core."""
    dropped = dict.fromkeys(SDFT_DROPS, 0)
    order = sorted(range(len(seeds)), key=lambda i: (
        2 if seeds[i].id in replay_ids else 1 if seeds[i].group == DAY_DREAM_GROUP else 0, i))
    sources: list[SeedRow] = []
    seen: set[str] = set()
    for i in order:
        if len(sources) < SDFT_SOURCES_MAX and (message := seeds[i].message.strip()) not in seen:
            seen.add(message)
            sources.append(seeds[i])
    candidates = [(row, kind, sdft_query(row.message, kind, row.id, seed)) for row in sources for kind in SDFT_KINDS]
    flags = decontam.check([query for _, _, query in candidates], list(sealed)) if candidates else ()
    if len(flags) != len(candidates):
        raise ValueError("the decontaminator must return one flag per text")
    rows: list[SdftRow] = []
    for (row, kind, query), flag in zip(candidates, flags, strict=True):
        tokens = _tokens(row.context, query) + SDFT_REPLY_TOKENS
        reason = guard.refusal(query) or ("probe" if flag else "truncated" if tokens > max_seq_tokens else None)
        if reason is not None:
            dropped[reason] += 1
            continue
        rows.append(SdftRow(id=f"{row.id}:{kind}", kind=kind, source=row.id, context=row.context, query=query,
                            tokens=tokens, core=row.core))
    return tuple(rows), dropped


def anchor_rows(seed: int, n: int = ANCHORS) -> tuple[AnchorRow, ...]:
    """A fixed, seeded ``n`` of the anchor set's items (disjoint from every gate): the same every night of a run."""
    ranked = sorted(load_anchors(), key=lambda item: hashlib.sha256(f"{seed}:anchor:{item.id}".encode()).hexdigest())
    return tuple(AnchorRow(id=item.id, question=item.question, choices=tuple(item.choices), answer=item.answer)
                 for item in ranked[:n])


def attach_night_terms(
    deps: NightDeps, uid: str, night: date, store: EvidenceStore, sealed: Sequence[Probe], data: Dataset,
    spend: SpendMeter,
) -> tuple[Dataset, dict[str, Any]]:
    """The night's dataset with the rows its terms ask for, and the dataset step's ``terms`` receipt
    (counts only). Dreams are asked of her Glimmer here, inside the night's GPU session."""
    seeds: tuple[SeedRow, ...] = ()
    done: dict[str, Any] = {}
    if deps.terms.seed:
        seeds, dropped = seed_rows(data, deps.transcripts(uid), store.for_day(night), sealed,
                                   hosted_in_training=deps.hosted_in_training, replay=deps.terms.day_dreams)
        done |= {"seeds": len(seeds), "seeds_dropped": dropped}
        if deps.terms.day_dreams:
            replay_ids = {example.id for example in data.examples if not example.new}
            done["seed_replay"] = sum(row.id in replay_ids for row in seeds)
    if deps.terms.day_dreams:
        if not deps.day_dreams.available:
            done["day_dreams"] = DAY_DREAMS_NOT_WIRED
        else:
            timer = deps.timer or time.monotonic
            day = day_dream_context(data, deps.transcripts(uid), store.for_day(night),
                                    hosted_in_training=deps.hosted_in_training)
            # Before any call: a day that may not train, or that holds a sealed probe, asks nothing of her
            # Glimmer and spends nothing; the dreams are zero and the receipt says why.
            refused_by = ("uncleared" if not may_train(day.ancestry, hosted_in_training=deps.hosted_in_training)
                          else "probe" if carries_verbatim(day.text, list(sealed)) else None)
            if refused_by is not None:
                done["day_dreams"] = {"generated": 0, "kept": 0, "dropped": dict.fromkeys(DAY_DREAM_DROPS, 0),
                                      "skipped": refused_by, "seconds": 0.0}
            else:
                started = timer()
                made = deps.day_dreams.dream(uid, night, deps.registry.last_promoted(uid), day)
                seconds = max(0.0, timer() - started)
                spend.add("practice", seconds / 3600.0 * deps.prices.serve_gpu_per_hour)  # her serving GPU's time
                rows, refused = day_dream_rows(made, day, sealed, deps.decontam,
                                               hosted_in_training=deps.hosted_in_training)
                seeds = (*seeds, *rows)
                done["day_dreams"] = {"generated": len(made), "kept": len(rows), "dropped": refused,
                                      "seconds": round(seconds, 3)}
    dreams: tuple[DreamRow, ...] = ()
    if deps.terms.dreams and not deps.dreams.available:
        done["dreams"] = DREAMS_NOT_WIRED
    elif deps.terms.dreams:
        timer = deps.timer or time.monotonic
        started = timer()
        made = deps.dreams.dream(uid, night, deps.registry.last_promoted(uid))
        seconds = max(0.0, timer() - started)
        spend.add("practice", seconds / 3600.0 * deps.prices.serve_gpu_per_hour)  # her serving GPU's time
        dreams, dropped = dream_rows(made, sealed, deps.decontam, hosted_in_training=deps.hosted_in_training)
        done |= {"dream_candidates": len(made), "dreams": len(dreams), "dreams_dropped": dropped,
                 "dreams_keep_rate": round(len(dreams) / len(made), 4) if made else None,  # the dream filter's
                 "dreams_by_seed": {kind: sum(row.kind == kind for row in dreams) for kind in SEED_KINDS},
                 "dream_seconds": round(seconds, 3)}
        if any(row.core is not None for row in dreams):  # dreamt through the thought channel
            done["dreams_in_channel"] = sum(row.core is not None for row in dreams)
    anchors = (anchor_rows(deps.settings.seed) if dreams or deps.terms.protect or deps.terms.day_dreams else ())
    done["anchors"] = len(anchors)
    sdft: tuple[SdftRow, ...] = ()
    if deps.terms.sdft:  # from the seeding rows above: no model is asked for anything
        guard = sdft_guard(sealed, held_out_items(store, night), store.stored(night))
        replay = frozenset(example.id for example in data.examples if not example.new)
        sdft, refused = sdft_rows(seeds, replay, guard, sealed, deps.decontam, seed=deps.settings.seed)
        done["sdft"] = {"rows": {kind: sum(row.kind == kind for row in sdft) for kind in SDFT_KINDS},
                        "sources": len({row.source for row in sdft}), "dropped": refused}
    return replace(data, seeds=seeds, dreams=dreams, anchors=anchors, sdft=sdft), done
