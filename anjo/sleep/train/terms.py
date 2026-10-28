"""``sft_dpo_seed``'s extra rows, built on the host from what the night already holds (2026-10-29).

**Knowledge seeding** (``seed_rows``). The paper's consolidation: the teacher is the model with the
short-term memory in context, the student the model without it. Here the teacher is Glimmer with the
adapter off and the consolidated day before the message: her notes on the person (the pre-practice
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
decontaminator's three rules and its thread (the student's input) to C's. **Anchors** (``anchor_rows``): a fixed, seeded 32 of the
general gate's 200 items, which the training job scores before and after the dream block.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import date
from typing import Any, Final

from anjo.sleep.context.recall_common import estimate_tokens
from anjo.sleep.night.deps import NightDeps
from anjo.sleep.night.dreams import SEED_KINDS, Dream
from anjo.sleep.night.meter import SpendMeter
from anjo.sleep.ports import Decontaminator, EvidenceStore
from anjo.sleep.practice.transcripts import TranscriptStore
from anjo.sleep.provenance import may_train, union
from anjo.sleep.sets.loader import load_general
from anjo.sleep.train.dataset import (
    DEFAULT_MAX_SEQ_TOKENS,
    ROLE_OVERHEAD_TOKENS,
    AnchorRow,
    Dataset,
    DreamRow,
    SeedRow,
    carries_verbatim,
)
from anjo.sleep.types import Evidence, Probe

DAY_KINDS: Final = ("lesson", "correction", "preference", "tone_request")  # what the person said today
ANCHORS: Final = 32
SEED_DROPS: Final = ("missing", "teacher_target", "no_context", "uncleared", "probe", "truncated")
DREAM_DROPS: Final = ("unanchored", "uncleared", "probe", "truncated")
NOTES_HEAD: Final = "What you know about them, from your notes:"
TODAY_HEAD: Final = "What they told you today:"


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


def seed_rows(
    data: Dataset, transcripts: TranscriptStore, day: Sequence[Evidence], sealed: Sequence[Probe], *,
    hosted_in_training: bool, max_seq_tokens: int = DEFAULT_MAX_SEQ_TOKENS,
) -> tuple[tuple[SeedRow, ...], dict[str, int]]:
    """Tonight's seeding rows, from the dataset's kept new examples, and the refusals by reason."""
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
            dropped[reason] += 1
            continue
        rows.append(SeedRow(id=example.id, context=context, message=transcript.message, reply=example.completion,
                            tokens=tokens))
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
                             kind=dream.kind))
    return tuple(rows), dropped


def anchor_rows(seed: int, n: int = ANCHORS) -> tuple[AnchorRow, ...]:
    """A fixed, seeded ``n`` of the general gate's items: the same every night of a run."""
    ranked = sorted(load_general(), key=lambda item: hashlib.sha256(f"{seed}:anchor:{item.id}".encode()).hexdigest())
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
                                   hosted_in_training=deps.hosted_in_training)
        done |= {"seeds": len(seeds), "seeds_dropped": dropped}
    dreams: tuple[DreamRow, ...] = ()
    if deps.terms.dreams and not deps.dreams.available:
        done["dreams"] = "no dream source: MIND's dream states aren't wired"
    elif deps.terms.dreams:
        timer = deps.timer or time.monotonic
        started = timer()
        made = deps.dreams.dream(uid, night, deps.registry.last_promoted(uid))
        seconds = max(0.0, timer() - started)
        spend.add("practice", seconds / 3600.0 * deps.prices.serve_gpu_per_hour)  # her serving GPU's time
        dreams, dropped = dream_rows(made, sealed, deps.decontam, hosted_in_training=deps.hosted_in_training)
        done |= {"dream_candidates": len(made), "dreams": len(dreams), "dreams_dropped": dropped,
                 "dreams_by_seed": {kind: sum(row.kind == kind for row in dreams) for kind in SEED_KINDS},
                 "dream_seconds": round(seconds, 3)}
    anchors = anchor_rows(deps.settings.seed) if dreams else ()
    done["anchors"] = len(anchors)
    return replace(data, seeds=seeds, dreams=dreams, anchors=anchors), done
