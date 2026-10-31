"""The night's SFT dataset (Task 5, ``build_night_dataset``) and its preference pairs (Task 7).

Every example is one practice transcript: its input is the snapshot of context C taken before
tonight's practice plus the person's message, and its target is her reply, with the loss on the
reply only (the rows are TRL's conversational prompt/completion form). Each row names the
target's provenance (``glimmer`` or ``teacher``). Daytime replies never become examples (§3).

A new transcript that kept the critic's (best, worst) pair also gives a ``Pair`` on the same prompt
as its example. Both sides go through the decontaminator: a probe in the chosen side drops the
example and its pair, one in the rejected side drops the pair only (``pairs_dropped``). Pair rows
are written only for the ``sft_dpo`` objective (``to_jsonl(pairs=True)``).

With the thought channel on, a transcript carries the core state its reply was made under, and its
example and pair rows carry it as ``core`` (``night.core_snapshot``): a replayed row keeps the state
of the night it was made on, and the training job splices that state's soft tokens into its prompt.
A seeding row carries its example's (``SeedRow.core``), and a dream row the imagined state's when its
source dreamt through the channel (``DreamRow.core``), for ``sft_dpo_seed``'s one job.

In order, the builder:

1. takes the stored practice items, never held-out ones, and refuses (and counts) any that
   ``trainable(until)`` doesn't return: unknown or hosted ancestry while hosted outputs can't train;
2. puts the new ones (stamped after ``since``, the last trained night) first, newest first, then a
   replay sample of earlier ones in an order seeded by the run seed, the person and the night;
3. refuses any example whose transcript's ancestry isn't cleared, anywhere in its input, context
   or target (§3), and counts it;
4. drops every example that carries a sealed probe, checked through the ``Decontaminator`` port,
   and counts it (RF5), and every pair with a probe on either side;
5. drops, and counts, any example longer than the training sequence, so nothing trains cut;
6. keeps examples while they fit ``token_cap`` together;
7. drops the whole of C from a seeded share of the examples (context dropout), chosen only among
   examples whose targets quote no URL or date.

The context (the person's own brief, never a training target: the loss is on the reply) is held to
exact and 13-gram only, stricter-for-training than input but looser than the paraphrase rule, which
dropped every example when the shared brief paraphrased two probes (2026-10-28). The message, the
reply and each DPO side keep all three rules. ``context_paraphrase_kept`` counts the examples that
only passed because of it.

Token counts use the conservative estimate (``estimate_tokens``), which over-counts English, so
the cap and the sequence length are never exceeded by the estimate's error. The training script
checks the length again with the real tokenizer.

Under ``sft_dpo_seed`` the dataset also carries the seeding, dream and anchor rows ``train/terms.py``
builds (``to_jsonl(terms=True)``, each row named by its ``kind``); any other objective never writes them.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from types import MappingProxyType
from typing import Final, Literal

from anjo.sleep.context.recall_common import estimate_tokens
from anjo.sleep.ports import Decontaminator, EvidenceStore
from anjo.sleep.practice.transcripts import PracticeTranscript, TranscriptStore
from anjo.sleep.probes.decontam import NGRAM, overlaps, probe_texts, words
from anjo.sleep.provenance import may_train
from anjo.sleep.serving import framing
from anjo.sleep.types import Evidence, Probe

DEFAULT_DROPOUT: Final = 0.6
DROPOUT_CHOICES: Final = (0.3, 0.6, 0.9)  # chosen on development, registered at the freeze
DEFAULT_MAX_SEQ_TOKENS: Final = 8192
ROLE_OVERHEAD_TOKENS: Final = 8  # the chat template's markers around each message, over-counted
DropReason = Literal["probe", "hosted", "truncated", "over_cap", "missing"]
DROP_REASONS: Final[tuple[DropReason, ...]] = ("probe", "hosted", "truncated", "over_cap", "missing")
# ``example``: its example was dropped; ``identical``: chosen and rejected are the same text
PairDropReason = Literal["probe", "truncated", "example", "identical"]
PAIR_DROP_REASONS: Final[tuple[PairDropReason, ...]] = ("probe", "truncated", "example", "identical")
_URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+|\b[a-z0-9-]+\.(?:com|org|net|io|edu|gov|dev|ai)\b")
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
    "|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_DATE = re.compile(
    rf"(?i)\b\d{{4}}-\d{{1,2}}-\d{{1,2}}\b|\b\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?\b"
    rf"|\b(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTHS})\b"
)

Message = tuple[str, str]  # (role, content)


@dataclass(frozen=True)
class Example:
    """One SFT row: ``prompt`` (an optional system C and the person's message) and the target."""

    id: str
    prompt: tuple[Message, ...]
    completion: str
    context_dropped: bool
    tokens: int
    new: bool
    teacher: bool = False  # the target is the Spark Open teacher's candidate (full practice)
    current_date: date | None = None  # the date its prompt was served on, when not the run's (Task 17b)
    weight: float = 1.0  # its loss weight: MIND's surprise at the evidence it was made from (the Sleep link)
    core: str | None = None  # the core state its reply was made under (``night.core_snapshot``, JSON), channel on


@dataclass(frozen=True)
class Pair:
    """One preference pair: the critic's best (``chosen``) and worst (``rejected``) replies on the
    example's prompt, and where each came from."""

    id: str
    prompt: tuple[Message, ...]
    chosen: str
    rejected: str
    chosen_from: str
    rejected_from: str
    tokens: int  # the prompt and the longer side
    core: str | None = None  # its example's core state (JSON), channel on


@dataclass(frozen=True)
class SeedRow:
    """One knowledge-seeding row: the teacher reads ``context`` (the consolidated day) before the
    message, the student the message alone, and both score ``reply``, her own practice reply.
    ``group`` marks one of her day dreams (``dream``); None is every other row, replay included."""

    id: str
    context: str
    message: str
    reply: str
    tokens: int
    core: str | None = None  # its example's core state (JSON), channel on: student and teacher both read it
    group: str | None = None


@dataclass(frozen=True)
class DreamRow:
    """One dream candidate: ``thread`` (MIND's state as text) is her system turn, the loss is on ``reply``.
    ``kind`` is its seed's (``boundary``, ``picked`` or ``random``), for the job's keep rate by kind."""

    id: str
    thread: str
    message: str
    reply: str
    tokens: int
    kind: str
    core: str | None = None  # the imagined state's snapshot (JSON), when the dream source dreamt through the channel


@dataclass(frozen=True)
class AnchorRow:
    """One of the general gate's items, for the training job's anchor check (never trained on)."""

    id: str
    question: str
    choices: tuple[str, ...]
    answer: int


def _messages(prompt: tuple[Message, ...]) -> list[dict[str, str]]:
    """A prompt as the server is sent it: its text cleaned of Glimmer's special tokens, and its
    system turn of the template's reasoning phrase (``framing.clean_messages``)."""
    return framing.clean_messages([{"role": role, "content": text} for role, text in prompt])


def _reply(text: str) -> list[dict[str, str]]:
    return [{"role": "assistant", "content": framing.clean_text(text)}]


def _core(core: str | None) -> dict[str, object]:
    """A row's ``core`` field: the state it was made under, channel on; nothing otherwise."""
    return {} if core is None else {"core": json.loads(core)}


@dataclass(frozen=True)
class Dataset:
    """The night's examples and pairs, and how many of each were dropped for each reason."""

    examples: tuple[Example, ...]
    tokens: int
    dropped: Mapping[str, int] = field(hash=False)
    pairs: tuple[Pair, ...] = ()
    pairs_dropped: Mapping[str, int] = field(default_factory=dict, hash=False)
    context_paraphrase_kept: int = 0  # examples kept although their context matched a probe by paraphrase only
    seeds: tuple[SeedRow, ...] = ()  # sft_dpo_seed's rows (train/terms.py); written only with terms=True
    dreams: tuple[DreamRow, ...] = ()
    anchors: tuple[AnchorRow, ...] = ()

    def __post_init__(self) -> None:
        counts: dict[str, int] = {reason: 0 for reason in DROP_REASONS}
        counts.update(self.dropped)
        object.__setattr__(self, "dropped", MappingProxyType(counts))
        pair_counts: dict[str, int] = {reason: 0 for reason in PAIR_DROP_REASONS}
        pair_counts.update(self.pairs_dropped)
        object.__setattr__(self, "pairs_dropped", MappingProxyType(pair_counts))

    def to_jsonl(self, *, pairs: bool = False, terms: bool = False) -> str:
        """The rows the training script reads: conversational prompt/completion, so TRL puts the
        loss on the completion only, then with ``pairs`` the preference rows (prompt, chosen,
        rejected) for the DPO pass, and with ``terms`` the seeding, dream and anchor rows. Every
        text is cleaned as the server's prompt is. A row with its own ``current_date`` carries it,
        and is rendered on it. A row MIND weighted carries its loss ``weight`` (1.0, the trainer's
        default, is left out)."""
        rows: list[dict[str, object]] = [
            {"id": example.id, "provenance": "teacher" if example.teacher else "glimmer",
             "prompt": _messages(example.prompt), "completion": _reply(example.completion),
             **({} if example.current_date is None else {"current_date": example.current_date.isoformat()}),
             **({} if example.weight == 1.0 else {"weight": example.weight}),
             **_core(example.core)}
            for example in self.examples
        ]
        if pairs:
            rows += [
                {"id": pair.id, "prompt": _messages(pair.prompt), "chosen": _reply(pair.chosen),
                 "rejected": _reply(pair.rejected), **_core(pair.core)}
                for pair in self.pairs
            ]
        if terms:
            rows += self._term_rows()
        return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)

    def weighted(self, weights: Mapping[str, float]) -> Dataset:
        """A copy with each example in ``weights`` carrying its loss weight (MIND's Sleep link)."""
        if any(not w > 0 for w in weights.values()):
            raise ValueError("a row's loss weight is positive")
        return replace(self, examples=tuple(replace(e, weight=weights.get(e.id, e.weight)) for e in self.examples))

    def _term_rows(self) -> list[dict[str, object]]:
        seeds: list[dict[str, object]] = [{"id": row.id, "kind": "seed", "teacher": _messages((("system", row.context), ("user", row.message))),
                  "prompt": _messages((("user", row.message),)), "completion": _reply(row.reply),
                  **({} if row.group is None else {"group": row.group}), **_core(row.core)}
                  for row in self.seeds]
        dreams: list[dict[str, object]] = [{"id": row.id, "kind": "dream", "seed_kind": row.kind,
                   "prompt": _messages((("system", row.thread), ("user", row.message))),
                   "completion": _reply(row.reply), **_core(row.core)} for row in self.dreams]
        anchors: list[dict[str, object]] = [{"id": row.id, "kind": "anchor", "prompt": _messages((("user", row.question),)),
                    "choices": [framing.clean_text(choice) for choice in row.choices], "answer": row.answer}
                   for row in self.anchors]
        return [*seeds, *dreams, *anchors]


def quotes_url_or_date(text: str) -> bool:
    """Whether ``text`` quotes a URL or a date, which only context can carry exactly."""
    return bool(_URL.search(text) or _DATE.search(text))


def _rank(seed: int, label: str, person: str, until: date, item_id: str) -> str:
    return hashlib.sha256(f"{seed}:{label}:{person}:{until.isoformat()}:{item_id}".encode()).hexdigest()


def _drops_context(seed: int, person: str, until: date, item_id: str, fraction: float) -> bool:
    digest = _rank(seed, "dropout", person, until, item_id)
    return int(digest[:12], 16) / float(16**12) < fraction


def _ordered(items: Sequence[Evidence], since: date | None, seed: int, until: date) -> list[tuple[Evidence, bool]]:
    new = [item for item in items if since is None or item.day > since]
    old = [item for item in items if since is not None and item.day <= since]
    new.sort(key=lambda item: (item.created_at, item.id), reverse=True)
    old.sort(key=lambda item: _rank(seed, "replay", item.person_id, until, item.id))
    return [(item, True) for item in new] + [(item, False) for item in old]


def _example(transcript: PracticeTranscript, *, new: bool, drop_context: bool) -> Example:
    prompt: tuple[Message, ...] = (("user", transcript.message),)
    if transcript.context and not drop_context:
        prompt = (("system", transcript.context), *prompt)
    tokens = sum(estimate_tokens(text) + ROLE_OVERHEAD_TOKENS for _, text in prompt)
    tokens += estimate_tokens(transcript.reply) + ROLE_OVERHEAD_TOKENS
    if transcript.core is not None and json.loads(transcript.core).get("dreamt") is True:
        raise ValueError(f"transcript {transcript.id} carries a dreamt state; a practice reply is made under her real one")
    return Example(
        id=transcript.id, prompt=prompt, completion=transcript.reply,
        context_dropped=transcript.context is not None and drop_context, tokens=tokens, new=new,
        teacher=transcript.target == "teacher", core=transcript.core,
    )


def _pair(transcript: PracticeTranscript, example: Example) -> Pair:
    assert transcript.rejected is not None
    prompt = sum(estimate_tokens(text) + ROLE_OVERHEAD_TOKENS for _, text in example.prompt)
    longer = max(estimate_tokens(transcript.reply), estimate_tokens(transcript.rejected))
    return Pair(
        id=transcript.id, prompt=example.prompt, chosen=transcript.reply, rejected=transcript.rejected,
        chosen_from=transcript.target, rejected_from=transcript.rejected_from or "glimmer",
        tokens=prompt + longer + ROLE_OVERHEAD_TOKENS, core=example.core,
    )


def _trained_text(transcript: PracticeTranscript, reply: str | None = None) -> str:
    """What trains: the message and a reply. All three decontamination rules apply to it."""
    parts = (transcript.message, transcript.reply if reply is None else reply)
    return "\n".join(part for part in parts if part)


def carries_verbatim(text: str, sealed: Sequence[Probe]) -> bool:
    """Whether ``text`` holds a whole probe, or a run of ``NGRAM`` words of one: the context's only rules."""
    hay = words(text)
    padded = " " + " ".join(hay) + " "
    runs = {" ".join(hay[i:i + NGRAM]) for i in range(len(hay) - NGRAM + 1)}
    for probe in sealed:
        for needle in probe_texts(probe):
            tokens = words(needle)
            if not tokens:
                continue
            if " " + " ".join(tokens) + " " in padded:
                return True
            if len(tokens) >= NGRAM and any(" ".join(tokens[i:i + NGRAM]) in runs
                                            for i in range(len(tokens) - NGRAM + 1)):
                return True
    return False


def _paraphrases(text: str, sealed: Sequence[Probe]) -> bool:
    return any(overlaps(text, needle) for probe in sealed for needle in probe_texts(probe) if needle.strip())


def _leaks(
    candidates: Sequence[tuple[PracticeTranscript, bool]], sealed: list[Probe], decontam: Decontaminator,
) -> tuple[list[bool], list[bool], list[bool]]:
    """Per candidate: whether its example carries a sealed probe (in what trains, or verbatim in its
    context), whether its pair's rejected side does, and whether its context matches a probe by
    paraphrase alone (kept, and counted). Every text that trains goes through the decontaminator."""
    texts = [_trained_text(t) for t, _ in candidates]
    sides = [i for i, (t, new) in enumerate(candidates) if new and t.rejected is not None]
    texts += [_trained_text(candidates[i][0], candidates[i][0].rejected) for i in sides]
    flags = decontam.check(texts, sealed) if texts else ()
    if len(flags) != len(texts):
        raise ValueError("the decontaminator must return one flag per text")
    rejected = [False] * len(candidates)
    for i, flag in zip(sides, flags[len(candidates):], strict=True):
        rejected[i] = bool(flag)
    verbatim: dict[str, bool] = {}  # the brief is the same for most transcripts: check each once
    soft: dict[str, bool] = {}
    leaked, paraphrased = [], []
    for (t, _), flag in zip(candidates, flags, strict=False):
        ctx = t.context or ""
        if ctx not in verbatim:
            verbatim[ctx] = bool(ctx) and carries_verbatim(ctx, sealed)
            soft[ctx] = bool(ctx) and not verbatim[ctx] and _paraphrases(ctx, sealed)
        leaked.append(bool(flag) or verbatim[ctx])
        paraphrased.append(soft[ctx])
    return leaked, rejected, paraphrased


def build_night_dataset(
    store: EvidenceStore,
    since: date | None,
    until: date,
    sealed: list[Probe],
    decontam: Decontaminator,
    token_cap: int,
    *,
    transcripts: TranscriptStore,
    seed: int,
    dropout: float = DEFAULT_DROPOUT,
    max_seq_tokens: int = DEFAULT_MAX_SEQ_TOKENS,
    hosted_in_training: bool = False,
) -> tuple[Dataset, int]:
    """The night's dataset and how many examples were dropped as probe matches."""
    if not 0.0 <= dropout < 1.0:
        raise ValueError("the context dropout fraction is in [0, 1)")
    if token_cap <= 0:
        raise ValueError("the night's token cap is positive")
    dropped: dict[str, int] = {reason: 0 for reason in DROP_REASONS}
    trainable = {item.id for item in store.trainable(until)}
    practice = [item for item in store.stored(until) if item.kind == "practice" and not item.held_out]
    candidates: list[tuple[PracticeTranscript, bool]] = []
    for item, new in _ordered(practice, since, seed, until):
        transcript = transcripts.get(item.id)
        if transcript is None or transcript.person_id != item.person_id:
            dropped["missing"] += 1
        elif item.id not in trainable or not may_train(
            transcript.ancestry, hosted_in_training=hosted_in_training,
        ):
            dropped["hosted"] += 1  # hosted or unknown ancestry, in the item or its transcript
        else:
            candidates.append((transcript, new))
    leaked, rejected_leaked, soft_context = _leaks(candidates, sealed, decontam)
    kept: list[Example] = []
    pairs: list[Pair] = []
    pairs_dropped: dict[str, int] = {reason: 0 for reason in PAIR_DROP_REASONS}
    used = 0
    context_paraphrase_kept = 0
    for index, (transcript, new) in enumerate(candidates):
        paired = new and transcript.rejected is not None  # only the night's new practice gives pairs
        drop = (
            transcript.context is not None and not quotes_url_or_date(transcript.reply)
            and _drops_context(seed, transcript.person_id, until, transcript.id, dropout)
        )
        example = _example(transcript, new=new, drop_context=drop)
        reason: DropReason | None = (
            "probe" if leaked[index]
            else "truncated" if _example(transcript, new=new, drop_context=False).tokens > max_seq_tokens
            else "over_cap" if used + example.tokens > token_cap else None
        )
        if reason is not None:
            dropped[reason] += 1
            if paired:
                pairs_dropped["probe" if reason == "probe" else "example"] += 1
            continue
        kept.append(example)
        context_paraphrase_kept += soft_context[index]
        used += example.tokens
        if not paired:
            continue
        pair = _pair(transcript, example)
        why: PairDropReason | None = (
            "probe" if rejected_leaked[index] else "identical" if pair.chosen == pair.rejected
            else "truncated" if pair.tokens > max_seq_tokens else None
        )
        if why is not None:
            pairs_dropped[why] += 1
        else:
            pairs.append(pair)
    data = Dataset(examples=tuple(kept), tokens=used, dropped=dropped, pairs=tuple(pairs), pairs_dropped=pairs_dropped,
                   context_paraphrase_kept=context_paraphrase_kept)
    return data, dropped["probe"]
