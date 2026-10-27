"""The night's SFT dataset (Task 5, ``build_night_dataset``).

Every example is one practice transcript: its input is the snapshot of context C taken before
tonight's practice plus the person's message, and its target is her reply, with the loss on the
reply only (the rows are TRL's conversational prompt/completion form). Daytime replies never
become examples (§3).

In order, the builder:

1. takes the stored practice items, never held-out ones, and refuses (and counts) any that
   ``trainable(until)`` doesn't return: unknown or hosted ancestry while hosted outputs can't train;
2. puts the new ones (stamped after ``since``, the last trained night) first, newest first, then a
   replay sample of earlier ones in an order seeded by the run seed, the person and the night;
3. refuses any example whose transcript's ancestry isn't cleared, anywhere in its input, context
   or target (§3), and counts it;
4. drops every example that carries a sealed probe, checked through the ``Decontaminator`` port,
   and counts it (RF5);
5. drops, and counts, any example longer than the training sequence, so nothing trains cut;
6. keeps examples while they fit ``token_cap`` together;
7. drops the whole of C from a seeded share of the examples (context dropout), chosen only among
   examples whose targets quote no URL or date.

Token counts use the conservative estimate (``estimate_tokens``), which over-counts English, so
the cap and the sequence length are never exceeded by the estimate's error. The training script
checks the length again with the real tokenizer.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType
from typing import Final, Literal

from anjo.sleep.context.recall_common import estimate_tokens
from anjo.sleep.ports import Decontaminator, EvidenceStore
from anjo.sleep.practice.transcripts import PracticeTranscript, TranscriptStore
from anjo.sleep.provenance import may_train
from anjo.sleep.types import Evidence, Probe

DEFAULT_DROPOUT: Final = 0.6
DROPOUT_CHOICES: Final = (0.3, 0.6, 0.9)  # chosen on development, registered at the freeze
DEFAULT_MAX_SEQ_TOKENS: Final = 8192
ROLE_OVERHEAD_TOKENS: Final = 8  # the chat template's markers around each message, over-counted
DropReason = Literal["probe", "hosted", "truncated", "over_cap", "missing"]
DROP_REASONS: Final[tuple[DropReason, ...]] = ("probe", "hosted", "truncated", "over_cap", "missing")
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


@dataclass(frozen=True)
class Dataset:
    """The night's examples and how many were dropped for each reason."""

    examples: tuple[Example, ...]
    tokens: int
    dropped: Mapping[str, int] = field(hash=False)

    def __post_init__(self) -> None:
        counts: dict[str, int] = {reason: 0 for reason in DROP_REASONS}
        counts.update(self.dropped)
        object.__setattr__(self, "dropped", MappingProxyType(counts))

    def to_jsonl(self) -> str:
        """The rows the training script reads: conversational prompt/completion, so TRL puts the
        loss on the completion only."""
        return "".join(
            json.dumps({
                "id": example.id,
                "prompt": [{"role": role, "content": text} for role, text in example.prompt],
                "completion": [{"role": "assistant", "content": example.completion}],
            }, ensure_ascii=False) + "\n"
            for example in self.examples
        )


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
    return Example(
        id=transcript.id, prompt=prompt, completion=transcript.reply,
        context_dropped=transcript.context is not None and drop_context, tokens=tokens, new=new,
    )


def _full_text(transcript: PracticeTranscript) -> str:
    return "\n".join(part for part in (transcript.context, transcript.message, transcript.reply) if part)


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
    flags = decontam.check([_full_text(t) for t, _ in candidates], sealed) if candidates else ()
    if len(flags) != len(candidates):
        raise ValueError("the decontaminator must return one flag per text")
    kept: list[Example] = []
    used = 0
    for (transcript, new), leaked in zip(candidates, flags, strict=True):
        if leaked:
            dropped["probe"] += 1
            continue
        drop = (
            transcript.context is not None and not quotes_url_or_date(transcript.reply)
            and _drops_context(seed, transcript.person_id, until, transcript.id, dropout)
        )
        full = _example(transcript, new=new, drop_context=False)
        if full.tokens > max_seq_tokens:
            dropped["truncated"] += 1
            continue
        example = _example(transcript, new=new, drop_context=drop)
        if used + example.tokens > token_cap:
            dropped["over_cap"] += 1
            continue
        kept.append(example)
        used += example.tokens
    return Dataset(examples=tuple(kept), tokens=used, dropped=dropped), dropped["probe"]
