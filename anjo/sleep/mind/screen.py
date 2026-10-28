"""MIND's domain screen on Spark Open, the LLM side of ``ScreenPort`` (Task 17). The night builds
it per account with their names (``app/screen.py``'s ``screen_for``); APD's synthetic lives have
none.

The kernel's denylist runs first, as data and a pure check (``kernel.DENYLIST``); this screen reads
what passed it. It limits what she investigates, not what she says.

- **A research query** first meets fixed checks that cost no call: homework's
  ``screen_query`` (no ``@``, URL or domain, run of five digits, or word of the person's names)
  and no date (a weekday, a month's name, a numeric date). "May" and "March" are left to the
  model, since they are also ordinary words.
- **Then Spark Open** (``screen_v1.md``) reads the purpose and the text, and answers
  ``{"allowed": true|false}``. The person's names are never sent. Each call is capped at
  ``SCREEN_MAX_TOKENS`` under the allowance the loop passes.
- **Anything short of a clear yes refuses:** a failed call (charged its cap, as in
  ``anjo.sleep.spark_inquiry``), a reply out of form, or no allowance left.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from anjo.sleep.homework.agent import screen_query
from anjo.sleep.mind.inquiry.parse import field_value
from anjo.sleep.mind.inquiry.types import Screened, ScreenPurpose
from anjo.sleep.spark_inquiry import PROMPTS, ChatFn, call, spark_open

SCREEN_PROMPT: Final = (PROMPTS / "screen_v1.md").read_text(encoding="utf-8")
SCREEN_MAX_TOKENS: Final = 200
_WEEKDAYS: Final = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_MONTHS: Final = (
    "january|february|april|june|july|august|september|october|november|december"
    "|jan|feb|apr|jun|jul|aug|sept|sep|oct|nov|dec"
)
_DATE: Final = re.compile(
    rf"(?i)\b(?:{_WEEKDAYS}|{_MONTHS})\b|\b\d{{4}}-\d{{1,2}}-\d{{1,2}}\b|\b\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?\b"
)


def personal_terms(query: str, names: Sequence[str]) -> bool:
    """Whether a research query carries a fixed personal term: what ``screen_query`` refuses,
    or a date."""
    return screen_query(query, names) is not None or _DATE.search(query) is not None


@dataclass(frozen=True)
class SparkScreen:
    """The ``ScreenPort`` for one person: ``names`` are theirs, for the research-query check."""

    names: tuple[str, ...]
    chat: ChatFn = spark_open

    def screen(self, text: str, purpose: ScreenPurpose, max_tokens: int) -> Screened:
        if purpose == "research_query" and personal_terms(text, self.names):
            return Screened(allowed=False, tokens=0)
        cap = min(max_tokens, SCREEN_MAX_TOKENS)
        if cap <= 0:
            return Screened(allowed=False, tokens=0)
        messages = [
            {"role": "system", "content": SCREEN_PROMPT},
            {"role": "user", "content": f"Purpose: {purpose}\nText: {text}"},
        ]
        reply = call(self.chat, messages, cap)
        return Screened(allowed=field_value(reply.text, "allowed") is True, tokens=reply.tokens)

