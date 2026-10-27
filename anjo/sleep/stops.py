"""Why a night stops early (Task 5, RF1): its budget ran out, or it was cancelled (Task 9).

Both end the night as ``failed`` at the step they hit. The previous checkpoint keeps serving and
no note is held, as for any other failure.
"""
from __future__ import annotations


class BudgetStop(RuntimeError):
    """A paid job could not reserve its worst case: a pool would pass its cap."""


class Cancelled(RuntimeError):
    """The night's cancel flag was set between steps."""
