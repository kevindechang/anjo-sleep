"""What the live page never shows (Task 13a, full step): text only the scorer may read.

The mind panel shows her own words: her targets' labels, her questions, hypotheses and predictions,
the ledger's notes and the MIND report. None of its sources should hold held-out text (her thinker,
her targets and her receipts never see a held-out item or a probe), and this filter makes sure on
the server, before anything is sent. ``HeldOutText`` is built from every held-out text there is for
the person:

- her persona file's held-out part (``PersonFile.held_out``): the preferences no assistant saw (the
  judge brief's "Preferences no assistant saw" section), the ``taste_ab`` items with their options,
  and the held-out new work the sealed probes and the relevance labels are written from;
- every evidence item stored with ``held_out=True``.

A shown text is withheld when it carries one of them by the engine's own decontamination rules
(``probes.decontam.overlaps``: the whole item, a 13-word run, or most of its content words close
together), or when the two share any run of ``SHARED_RUN`` words in either direction. Both rules err
toward withholding. The sealed probe files themselves are never opened here: the web process has no
business reading them, and what they are written from is covered above.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Final

from anjo.sleep.personas.model import PersonFile
from anjo.sleep.probes.decontam import overlaps, words

SHARED_RUN: Final = 8  # words in a row that a shown text may never share with a held-out one


def persona_texts(person: PersonFile) -> tuple[str, ...]:
    """Every held-out text of a persona or life file."""
    held = person.held_out
    taste = [text for item in held.taste_ab for text in (item.prompt, *item.options)]
    return (*(s.text for s in held.preferences), *taste, *(w.text for w in held.new_work))


def _runs(tokens: Sequence[str]) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + SHARED_RUN]) for i in range(len(tokens) - SHARED_RUN + 1)}


class HeldOutText:
    """The filter. ``shown`` returns a text as it may be shown, or None when it is withheld."""

    def __init__(self, texts: Iterable[str]) -> None:
        self._texts = tuple(text for text in texts if text.strip())
        self._runs: set[tuple[str, ...]] = set().union(*(_runs(words(text)) for text in self._texts))
        self._seen: dict[str, bool] = {}

    def carries(self, text: str) -> bool:
        if text not in self._seen:
            self._seen[text] = bool(self._runs & _runs(words(text))) or any(
                overlaps(text, held) for held in self._texts)
        return self._seen[text]

    def shown(self, text: str | None) -> str | None:
        if text is None or self.carries(text):
            return None
        return text

    def item(self, text: str | None) -> dict[str, Any] | None:
        """``text`` as the page receives it, ``{"text": ..., "withheld": ...}``; None stays None."""
        if text is None:
            return None
        shown = self.shown(text)
        return {"text": shown, "withheld": shown is None}
