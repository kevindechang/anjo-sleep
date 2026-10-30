"""The practice loop's marks on MIND's log, exact and pure (owner decision 2026-10-31).

The night's practice loop writes its own ``RunStart`` and opens a question about her own learning
on a focus target. Both are hers, but neither is a notice pass or an inquiry: a reader that resumes
a run must skip the start, and MIND's resolver must never settle the predictions on that question
from unrelated evidence. The words below are the one definition of both, so the writer and every
reader agree exactly.
"""
from __future__ import annotations

from typing import Final

LOOP_DECIDER: Final = "mind-loop"  # what the loop's RunStart records; no inquiry resumes it
SUBJECT_PREFIX: Final = "Does practising "
SUBJECT_SUFFIX: Final = " make my own replies hold up?"


def is_loop_start(entry: object) -> bool:
    """Whether a log entry is the practice loop's own ``RunStart`` (not a notice or inquiry run)."""
    return getattr(entry, "decider", None) == LOOP_DECIDER


def is_self_practice(subject: str) -> bool:
    """Whether a question's subject is the loop's question about her own practice."""
    return subject.startswith(SUBJECT_PREFIX) and subject.endswith(SUBJECT_SUFFIX)
