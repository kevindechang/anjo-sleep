"""Replay's labels (Task 17b): which of a start state's frozen hypotheses held.

One frozen proposition per start state: its hypothesis set, the catch-all last. Spark Open
(``resolve_v1.md``) reads those hypotheses, each with a letter, and only the evidence that arrived
after the start state, up to the life's end (the surface-day reveal included), and answers the
letter that holds or ``undetermined``. Its prompt is built from the start state and the life alone,
so it never sees a branch. A start state with no later evidence is undetermined without a call.
A call that fails or answers out of form is undetermined too, marked ``failed``. Every label is
written once into a ``LabelBook``, which every branch of the start state is scored against.

**The label's model is the API's report.** The labeller calls its ``ChatFn`` itself, not through
``spark_inquiry.call``, which records an unreported model as Spark Open: a label keeps the model
the reply named, as named, and a failed call names none (``""``). Only a label Spark Open's reply
names may branch or train (``Label.usable``); any other is dropped as ``provenance``.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from anjo.sleep.mind.apd.arms import json_values, night_as_of
from anjo.sleep.mind.apd.types import Life
from anjo.sleep.mind.replay.types import UNDETERMINED, Label, StartState, check_synthetic
from anjo.sleep.mind.replay.types import LabelBook as LabelBook
from anjo.sleep.mind.replay.types import LabelRewrite as LabelRewrite
from anjo.sleep.model_api import ModelApiError
from anjo.sleep.spark_inquiry import ChatFn
from anjo.sleep.types import Evidence

logger = logging.getLogger(__name__)

RESOLVE_PROMPT: Final = (Path(__file__).parent / "prompts" / "resolve_v1.md").read_text(encoding="utf-8")
LABEL_MAX_TOKENS: Final = 200
LETTERS: Final = "ABCDE"  # at most four alternatives and the catch-all


def later_evidence(start: StartState, life: Life) -> tuple[Evidence, ...]:
    """What arrived after the start state, up to the end of the life's last day, in order."""
    end = night_as_of(life.days[-1])
    later = [item for item in life.evidence if not item.held_out and start.as_of < item.created_at <= end]
    return tuple(sorted(later, key=lambda item: (item.created_at, item.id)))


def _line(text: str) -> str:
    return " ".join(text.split())


def label_messages(start: StartState, life: Life) -> list[dict[str, Any]]:
    """The labeller's prompt: the frozen hypotheses by letter, then the later evidence by day."""
    hypotheses = "\n".join(f"{LETTERS[n]}. {_line(statement)}" for n, (_, statement) in enumerate(start.hypotheses))
    lines = "\n".join(f"- ({item.day.isoformat()}) {_line(item.text)}" for item in later_evidence(start, life))
    return [
        {"role": "system", "content": RESOLVE_PROMPT},
        {"role": "user", "content": f"Hypotheses:\n{hypotheses}\n\nEvidence:\n{lines}"},
    ]


def parse_label(text: str, start: StartState) -> tuple[str | None, bool]:
    """The frozen hypothesis the reply names, or ``None``; and whether the reply was out of form."""
    found = next((v["holds"] for v in json_values(text) if isinstance(v, dict) and "holds" in v), None)
    if found == UNDETERMINED:
        return None, False
    letters = LETTERS[: len(start.hypotheses)]
    if isinstance(found, str) and len(found) == 1 and found in letters:
        return start.frozen_ids[letters.index(found)], False
    return None, True


@dataclass(frozen=True)
class SparkLabeler:
    """The labeller on Spark Open (a ``ChatFn``: ``spark_inquiry.spark_open`` live, a fake in tests)."""

    chat: ChatFn

    def label(self, start: StartState, life: Life) -> Label:
        """No later evidence is undetermined with no call; a failed call is charged its cap."""
        if not later_evidence(start, life):
            return Label(start.id, None, "")
        try:
            result = self.chat(label_messages(start, life), LABEL_MAX_TOKENS)
        except ModelApiError as error:
            logger.warning("replay's label call failed, charged its cap of %d tokens: %s", LABEL_MAX_TOKENS, error)
            return Label(start.id, None, "", tokens=LABEL_MAX_TOKENS, failed=True)
        holds, failed = parse_label(result.text, start)
        return Label(start.id, holds, result.model, tokens=result.output_tokens, failed=failed)


def label_all(starts: Sequence[StartState], lives: Mapping[str, Life], labeler: SparkLabeler) -> LabelBook:
    """One label per start state, each written once."""
    book = LabelBook()
    for start in starts:
        life = lives[start.life_id]
        check_synthetic(life)
        book = book.with_label(labeler.label(start, life))
    return book

