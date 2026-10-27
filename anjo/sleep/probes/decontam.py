"""The engine core's decontamination check (Task 5, RF5): no training text may carry a sealed probe.

``OverlapDecontaminator`` is the ``Decontaminator`` port with public parts only, so the standalone
example runs it too. It is simpler than Anjo's own check, and says so. A text overlaps a probe when
any of these holds, after NFKC, case folding and collapsing everything but letters and digits:

- **exact:** the whole probe appears in the text;
- **13-gram:** the text shares a run of 13 words with the probe;
- **paraphrase:** inside a window of the text about three times the probe's length, it holds at
  least 60% of the probe's content words (and at least three). Content words are words of three
  or more letters that aren't on a short stop list, with a plural ``s`` folded.

The paraphrase rule errs toward dropping: a dropped example costs one training row, a leaked probe
costs the test. The app adds a semantic check on top (``anjo.sleep.app.decontam``). Nothing here
logs or returns a probe's text: the result is one flag per text.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from typing import Final

from anjo.sleep.types import Probe

NGRAM: Final = 13
PARAPHRASE_SHARE: Final = 0.6
PARAPHRASE_MIN_WORDS: Final = 3
WINDOW_FACTOR: Final = 3
MIN_WINDOW: Final = 24
_WORD = re.compile(r"[^\W_]+")
_STOP: Final = frozenset("""
a an and are as at be been but by can could did do does for from had has have how i if in into is it
its just me my of on or our should so than that the their them then there these they this to too us
was we were what when where which who why will with would you your about after any around best each
much some way what's whats she her hers his him
""".split())


def words(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _WORD.findall(folded)


def _content(word: str) -> str | None:
    if len(word) < 3 or word in _STOP or word.isdigit():
        return None
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def _ngrams(tokens: Sequence[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _contains(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    first = needle[0]
    size = len(needle)
    return any(
        haystack[i] == first and list(haystack[i:i + size]) == list(needle)
        for i in range(len(haystack) - size + 1)
    )


def _paraphrased(text_words: Sequence[str], probe_words: Sequence[str]) -> bool:
    wanted = {word for word in (_content(w) for w in probe_words) if word is not None}
    if len(wanted) < PARAPHRASE_MIN_WORDS:
        return False
    needed = max(PARAPHRASE_MIN_WORDS, math.ceil(PARAPHRASE_SHARE * len(wanted)))
    marks = [_content(word) for word in text_words]
    window = max(MIN_WINDOW, WINDOW_FACTOR * len(probe_words))
    inside: Counter[str] = Counter()
    for index, mark in enumerate(marks):
        if mark in wanted:
            inside[mark] += 1
        leaving = index - window
        if leaving >= 0 and marks[leaving] in wanted:
            gone = marks[leaving]
            assert gone is not None
            inside[gone] -= 1
            if inside[gone] == 0:
                del inside[gone]
        if len(inside) >= needed:
            return True
    return False


def overlaps(text: str, probe_text: str) -> bool:
    """Whether ``text`` carries ``probe_text`` by any of the three rules."""
    text_words, probe_words = words(text), words(probe_text)
    if not probe_words:
        return False
    if _contains(text_words, probe_words):
        return True
    if len(probe_words) >= NGRAM and _ngrams(text_words, NGRAM) & _ngrams(probe_words, NGRAM):
        return True
    return _paraphrased(text_words, probe_words)


def probe_texts(probe: Probe) -> tuple[str, ...]:
    """Every text of a probe a training row could carry: the prompt and, for ``taste_ab``, its
    options."""
    return (probe.prompt, *(probe.options or ()))


class OverlapDecontaminator:
    """The ``Decontaminator`` port: exact, 13-gram and paraphrase overlap."""

    def check(self, texts: Sequence[str], sealed: Sequence[Probe]) -> tuple[bool, ...]:
        needles = [text for probe in sealed for text in probe_texts(probe) if text.strip()]
        return tuple(any(overlaps(text, needle) for needle in needles) for text in texts)
