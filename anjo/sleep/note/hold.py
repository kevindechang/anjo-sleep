"""``note.hold`` (Task 5, step 10): the one note a night may leave.

A note is written only for a ``hold_note`` choice, and never on a failed night (RF1: no note claims
a night that didn't finish). The writer (Glimmer as her, with ``note_v1.md``) may decline, which
the night records as ``note_declined``. Every fact from homework keeps its source: a source the
writer didn't cite inline is listed under the note. On a rejected night the note carries one
honest line that the update was held back (§2.4).

The night queues the note through the app's outreach (``Notifier.queue``), shown at the next open
(for a synthetic person, the next uncleared one), and pushes it only to a person who opted in.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final, Literal, Protocol

from anjo.sleep.context.template import render
from anjo.sleep.probes.clients import ModelClient
from anjo.sleep.types import CheckpointRef, Choice, NightResult

PROMPT_PATH: Final = Path(__file__).parent / "prompts" / "note_v1.md"
SOURCE: Final = "sleep_note"
TEMPERATURE: Final = 0.7
DECLINE: Final = "NOTHING"
# Copy: the honest line on a rejected night, and the sources heading.
HELD_BACK_LINE: Final = (
    "One more thing: I also tried to learn from our recent conversations overnight, and I held "
    "that update back because it didn't pass my own checks."
)
SOURCES_HEADING: Final = "Sources:"
NoteOutcome = Literal["held", "declined", "not_chosen", "failed_night"]


@dataclass(frozen=True)
class Fact:
    """One thing homework found, and where (a URL or a citation)."""

    text: str
    source: str


@dataclass(frozen=True)
class HeldNote:
    id: str
    user_id: str
    night: date
    target_id: str
    text: str
    sources: tuple[str, ...]
    held_back: bool


class NoteWriter(Protocol):
    def write(self, user_id: str, topic: str, facts: Sequence[Fact]) -> str | None:
        """Her note about ``topic`` from ``facts``, or None to decline."""


def _with_sources(text: str, facts: Sequence[Fact]) -> str:
    missing = list(dict.fromkeys(fact.source for fact in facts if fact.source not in text))
    if not missing:
        return text
    return text + "\n\n" + SOURCES_HEADING + "\n" + "\n".join(f"- {source}" for source in missing)


def _passes(text: str) -> bool:
    return True


def write_note(
    user_id: str, choice: Choice, result: NightResult, *, writer: NoteWriter, topic: str,
    facts: Sequence[Fact], night: date, check: Callable[[str], bool] = _passes,
) -> tuple[HeldNote | None, NoteOutcome]:
    """The note and what happened: held, declined (by the writer, or by ``check`` on the final
    text, sources and held-back line included), not chosen, or a failed night."""
    if choice.action != "hold_note" or choice.target_id is None:
        return None, "not_chosen"
    if result.status == "failed":
        return None, "failed_night"
    if any(not fact.source.strip() for fact in facts):
        raise ValueError("every homework fact keeps its source")
    written = writer.write(user_id, topic, facts)
    if written is None or not written.strip() or written.strip() == DECLINE:
        return None, "declined"
    text = _with_sources(written.strip(), facts)
    held_back = result.status == "rejected"
    if held_back:
        text = f"{text}\n\n{HELD_BACK_LINE}"
    if not check(text):
        return None, "declined"
    digest = hashlib.sha256(f"{user_id}\x1f{night.isoformat()}\x1f{choice.target_id}".encode()).hexdigest()[:16]
    note = HeldNote(
        id=f"note-{digest}", user_id=user_id, night=night, target_id=choice.target_id, text=text,
        sources=tuple(dict.fromkeys(fact.source for fact in facts)), held_back=held_back,
    )
    return note, "held"


def hold(
    user_id: str, choice: Choice, result: NightResult, *, writer: NoteWriter, topic: str,
    facts: Sequence[Fact], night: date, check: Callable[[str], bool] = _passes,
) -> HeldNote | None:
    """The spec's ``note.hold``: the note, or None."""
    return write_note(user_id, choice, result, writer=writer, topic=topic, facts=facts, night=night,
                      check=check)[0]


class GlimmerNoteWriter:
    """Her own model writes the note: the person's serving checkpoint, the base model before one."""

    def __init__(
        self, client: ModelClient, adapter: Callable[[str], CheckpointRef | None], *, seed: int,
    ) -> None:
        self._client = client
        self._adapter = adapter
        self._seed = seed
        self._template = PROMPT_PATH.read_text(encoding="utf-8")

    def write(self, user_id: str, topic: str, facts: Sequence[Fact]) -> str | None:
        lines = "\n".join(f"- {fact.text} [{fact.source}]" for fact in facts) or "- nothing"
        prompt = render(self._template, {"TOPIC": topic, "FACTS": lines})
        reply = self._client.complete(
            self._adapter(user_id), [{"role": "user", "content": prompt}], temperature=TEMPERATURE,
            seed=self._seed,
        ).text.strip()
        return None if not reply or reply == DECLINE else reply
